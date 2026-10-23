// ============================================================================
//  lattice_shim.hpp - Lattice 桥共享内核（全 inline，仅 shim 三个 .cpp 包含）
//
//  职责边界：句柄注册表、错误态、shape/dtype 解析、异常边界宏。逻辑分块：
//  lattice_core.cpp（生命周期/版本）/ lattice_tensor.cpp（张量域）/
//  lattice_train.cpp（训练与 JIT）。
//
//  ABI 契约（单一事实源；lattice.sg 与文档均引用此处，不复制）：
//    - 句柄 = long long，注册表从 1 递增发放；0 = 失败或 null。
//      free 仅注销注册表条目：from_data 借用视图的宿主内存归 Scourge，
//      注销只析构 torch 侧 wrapper，绝不触碰宿主内存。
//    - shape = "d1xd2x..." 字符串（0-dim 用 ""）；dtype = "int"|"float"
//      （Scourge i64/f64 ↔ torch kInt64/kFloat64，零转换对齐）。
//    - 每导出以 LATTICE_TRY/CATCH 包裹：异常 → 置 err 返回 0/""，
//      C++ 异常绝不逃出 DLL 边界（@extern 调用桩无 unwind 通道）。
//    - str 返回 = 每函数独立 static 缓冲（或 static string 的 c_str），
//      调用方在 FFI 边界即时深拷贝，顺序调用安全；跨函数共享缓冲会串值。
//    - 数值读失败（get_f/get_i）返回 0.0/0 且置 err——先 clear 再调用，
//      读后查 last_err 才能区分合法 0 与失败。
//    - 单线程契约：错误态与注册表无锁（Scourge 无多线程内建）。
// ============================================================================
#pragma once
#include <torch/script.h>          // 聚合头：ATen C++ API + jit(compile/load/Module)
#include <ATen/CPUGeneratorImpl.h> // 每调用独立 CPU 生成器 → 双模确定性基石
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define LATTICE_EXPORT extern "C" __declspec(dllexport)

// shim 自身版本（与生态包 pkg.json 的 version 同步）
#define LATTICE_SHIM_VERSION "0.1.0"
// 开发锁定版本：build_shim.bat 以 -DTORCH_VERSION_EXPECTED=2.13.0 注入；
// pp-number 词法使 2.13.0 成为单 token，无需引号转义。二段串化：单层 #x
// 会把宏名原样字符串化，先展开再串化才得到 "2.13.0"。
#define LATTICE_STR_(x) #x
#define LATTICE_STR(x) LATTICE_STR_(x)
#ifndef TORCH_VERSION_EXPECTED
#define TORCH_VERSION_EXPECTED 0.0.0
#endif
#define LATTICE_PIN_STR LATTICE_STR(TORCH_VERSION_EXPECTED)

namespace lattice {

// ---- 句柄注册表 ------------------------------------------------------------
// 句柄即 shared_ptr 所有权：free = 注销条目（wrapper 析构，torch 堆内存随
// 之释放）；from_data 借用视图的 wrapper 析构不影响宿主（无 deleter）。
enum class Kind { Tensor, ParamGroup, Optimizer, JitUnit, JitModule };

struct Entry {
    std::shared_ptr<void> obj;
    Kind kind;
};

inline std::unordered_map<long long, Entry>& reg() {
    static std::unordered_map<long long, Entry> m;
    return m;
}

inline long long put(std::shared_ptr<void> p, Kind k) {
    static long long next = 0; // 单线程契约，从 1 递增（0 保留为失败）
    long long id = ++next;
    reg().emplace(id, Entry{std::move(p), k});
    return id;
}

// 取句柄载荷；无效/类型不符抛异常，由 LATTICE_CATCH 转为 err 态
inline std::shared_ptr<void> get(long long h, Kind k) {
    auto it = reg().find(h);
    if (it == reg().end())
        throw std::runtime_error("句柄不存在或已释放");
    if (it->second.kind != k)
        throw std::runtime_error("句柄类型不符");
    return it->second.obj;
}

inline at::Tensor& tensor(long long h) {
    return *static_cast<at::Tensor*>(get(h, Kind::Tensor).get());
}

inline long long put_tensor(at::Tensor t) {
    return put(std::shared_ptr<void>(new at::Tensor(std::move(t))),
               Kind::Tensor);
}

// ---- 错误态（单线程契约，无锁） --------------------------------------------
inline std::string& err_buf() {
    static std::string s;
    return s;
}
inline void set_err(const std::string& m) { err_buf() = m; }

// ---- 解析工具 --------------------------------------------------------------
// "2x3" → {2,3}；"" → {}（0-dim 标量张量）。解析失败抛异常 → err。
inline std::vector<int64_t> parse_shape(const char* s) {
    std::vector<int64_t> v;
    if (s == nullptr || *s == '\0') return v;
    const char* p = s;
    while (*p) {
        if (!(*p >= '0' && *p <= '9'))
            throw std::runtime_error("shape 非法（应为 d1xd2x... 数字）");
        int64_t d = 0;
        while (*p >= '0' && *p <= '9') d = d * 10 + (*p++ - '0');
        if (*p == 'x') ++p;
        else if (*p != '\0')
            throw std::runtime_error("shape 非法（分隔符应为 x）");
        v.push_back(d);
    }
    return v;
}

// "int"|"float" → torch dtype；与 Scourge 张量 dtype 名逐字一致
inline at::TensorOptions opt_of(const char* dtype) {
    if (dtype != nullptr && std::strcmp(dtype, "int") == 0)
        return at::TensorOptions().dtype(at::kLong);
    if (dtype != nullptr && std::strcmp(dtype, "float") == 0)
        return at::TensorOptions().dtype(at::kDouble);
    throw std::runtime_error("dtype 仅支持 int/float");
}

} // namespace lattice

// 异常边界宏：所有导出函数体必须包裹（C++ 异常不逃出 DLL 边界）
#define LATTICE_TRY try {
#define LATTICE_CATCH                                                          \
    }                                                                          \
    catch (const std::exception& e) {                                          \
        lattice::set_err(e.what());                                            \
        return 0;                                                              \
    }                                                                          \
    catch (...) {                                                              \
        lattice::set_err("未知 C++ 异常");                                     \
        return 0;                                                              \
    }
#define LATTICE_CATCH_STR                                                      \
    }                                                                          \
    catch (const std::exception& e) {                                          \
        lattice::set_err(e.what());                                            \
        return "";                                                             \
    }                                                                          \
    catch (...) {                                                              \
        lattice::set_err("未知 C++ 异常");                                     \
        return "";                                                             \
    }
