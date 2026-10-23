// ============================================================================
//  lattice_tensor.cpp - Lattice 桥·张量域（28 导出）
//
//  职责边界：张量创建（zeros/ones/randn/标量/零拷贝借用）、元信息
//  （shape/dtype/numel/data_ptr）、Scourge 内存双通道（from_data 借入 /
//  copy_to_ptr 落回）、逐元素算术与规约、变形、激活。
//  生命周期/版本域见 lattice_core.cpp，训练/JIT 域见 lattice_train.cpp；
//  ABI 契约与共享内核见 lattice_shim.hpp（单一事实源）。
// ============================================================================
#include "lattice_shim.hpp"

#include <algorithm>

// ---- 创建 ------------------------------------------------------------------

LATTICE_EXPORT long long lattice_zeros(const char* shape, const char* dtype) {
    LATTICE_TRY
    return lattice::put_tensor(
        at::zeros(lattice::parse_shape(shape), lattice::opt_of(dtype)));
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_ones(const char* shape, const char* dtype) {
    LATTICE_TRY
    return lattice::put_tensor(
        at::ones(lattice::parse_shape(shape), lattice::opt_of(dtype)));
    LATTICE_CATCH
}

// randn：empty(kDouble) + normal_（独立 CPU 生成器 + 种子）。每调用独享
// 生成器 → 同一种子逐位可复现，VM/AOT 双模各调一次也逐位一致——这是双模
// 确定性的基石，禁用全局默认生成器（其状态随时序漂移）。
// seed 按无符号语义使用：负数回绕，只保证同值同结果。
LATTICE_EXPORT long long lattice_randn(const char* shape, long long seed) {
    LATTICE_TRY
    auto gen = at::make_generator<at::CPUGeneratorImpl>();
    gen.get<at::CPUGeneratorImpl>()->set_current_seed(
        static_cast<uint64_t>(seed));
    auto t = at::empty(lattice::parse_shape(shape),
                       at::TensorOptions().dtype(at::kDouble));
    t.normal_(0.0, 1.0, gen);
    return lattice::put_tensor(std::move(t));
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_scalar_f(double v) {
    LATTICE_TRY
    return lattice::put_tensor(
        at::scalar_tensor(v, at::TensorOptions().dtype(at::kDouble)));
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_scalar_i(long long v) {
    LATTICE_TRY
    return lattice::put_tensor(at::scalar_tensor(
        static_cast<int64_t>(v), at::TensorOptions().dtype(at::kLong)));
    LATTICE_CATCH
}

// from_data：借用 Scourge 侧宿主内存做零拷贝视图（无 deleter）。
// 契约：宿主必须是存活的命名变量（表达式临时量会空悬）；free 本句柄
// 不触碰宿主。算术/激活结果均为 torch 新堆内存，与宿主脱钩。
LATTICE_EXPORT long long lattice_from_data(long long ptr, const char* shape,
                                            const char* dtype) {
    LATTICE_TRY
    return lattice::put_tensor(at::from_blob(
        reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)),
        lattice::parse_shape(shape), lattice::opt_of(dtype)));
    LATTICE_CATCH
}

// ---- 元信息 ----------------------------------------------------------------

LATTICE_EXPORT const char* lattice_shape(long long h) {
    // 本函数独占 static 缓冲（ABI 契约：str 返回不跨函数共享）
    static std::string buf;
    LATTICE_TRY
    buf.clear();
    auto& t = lattice::tensor(h);
    for (int64_t i = 0; i < t.dim(); ++i) {
        if (i) buf += 'x';
        buf += std::to_string(t.size(i));
    }
    return buf.c_str();
    LATTICE_CATCH_STR
}

LATTICE_EXPORT const char* lattice_dtype(long long h) {
    LATTICE_TRY
    auto st = lattice::tensor(h).scalar_type();
    if (st == at::kLong) return "int";
    if (st == at::kDouble) return "float";
    // 严格域检查：from_data 按 f64/i64 解释宿主内存，宽放（如 kFloat 报
    // "float"）会让调用方把 4 字节元素读成 8 字节——宁报错不损坏。
    throw std::runtime_error("张量 dtype 非 int/float（仅支持零转换对齐域）");
    LATTICE_CATCH_STR
}

LATTICE_EXPORT long long lattice_numel(long long h) {
    LATTICE_TRY
    return static_cast<long long>(lattice::tensor(h).numel());
    LATTICE_CATCH
}

// 底层地址（导回 Scourge 侧做落回目标/零拷贝诊断）。要求张量连续——
// 本 shim 的算术结果与 from_data 视图恒连续。
LATTICE_EXPORT long long lattice_data_ptr(long long h) {
    LATTICE_TRY
    return static_cast<long long>(
        reinterpret_cast<uintptr_t>(lattice::tensor(h).data_ptr()));
    LATTICE_CATCH
}

// ---- Scourge 内存双通道 ----------------------------------------------------

// 落回：张量 → Scourge 裸地址（from_data 的反向通道）。非连续先物化，
// 按 numel*itemsize 拷贝；dst 指向 Scourge 侧命名数组。
LATTICE_EXPORT long long lattice_copy_to_ptr(long long dst, long long h) {
    LATTICE_TRY
    auto c = lattice::tensor(h).contiguous();
    const size_t bytes = c.numel() * c.itemsize();
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)),
                c.data_ptr(), bytes);
    return 1;
    LATTICE_CATCH
}

// 线性读：double 视角（int 张量经 Scalar 转换读取）。越界抛错；失败
// 返回 0 且置 err——先 clear 再调用，读后查 last_err 区分合法 0。
LATTICE_EXPORT double lattice_get_f(long long h, long long i) {
    LATTICE_TRY
    auto& t = lattice::tensor(h);
    if (i < 0 || i >= static_cast<long long>(t.numel()))
        throw std::runtime_error("线性下标越界");
    return t.contiguous().flatten()[static_cast<int64_t>(i)].item<double>();
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_get_i(long long h, long long i) {
    LATTICE_TRY
    auto& t = lattice::tensor(h);
    if (i < 0 || i >= static_cast<long long>(t.numel()))
        throw std::runtime_error("线性下标越界");
    return static_cast<long long>(
        t.contiguous().flatten()[static_cast<int64_t>(i)].item<int64_t>());
    LATTICE_CATCH
}

// ---- 逐元素算术与矩阵乘（ATen 类型提升：int+float → float 域） -------------

#define LATTICE_BIN(NAME, EXPR)                                                \
    LATTICE_EXPORT long long lattice_##NAME(long long a, long long b) {        \
        LATTICE_TRY                                                            \
        return lattice::put_tensor(EXPR);                                      \
        LATTICE_CATCH                                                          \
    }

LATTICE_BIN(add, at::add(lattice::tensor(a), lattice::tensor(b)))
LATTICE_BIN(sub, at::sub(lattice::tensor(a), lattice::tensor(b)))
LATTICE_BIN(mul, at::mul(lattice::tensor(a), lattice::tensor(b)))
LATTICE_BIN(div, at::div(lattice::tensor(a), lattice::tensor(b)))
LATTICE_BIN(matmul, at::matmul(lattice::tensor(a), lattice::tensor(b)))
LATTICE_BIN(pow, at::pow(lattice::tensor(a), lattice::tensor(b)))

#undef LATTICE_BIN

// ---- 规约（double 视角，int 张量经转换读取；失败返回 0 且置 err） -----------

LATTICE_EXPORT double lattice_sum(long long h) {
    LATTICE_TRY
    return lattice::tensor(h).sum().item<double>();
    LATTICE_CATCH
}

LATTICE_EXPORT double lattice_mean(long long h) {
    LATTICE_TRY
    return lattice::tensor(h).mean().item<double>();
    LATTICE_CATCH
}

// 张量均值（0-dim 结果句柄）：与 lattice_mean(double 值) 的分工——backward
// 需要损失在计算图内（0-dim 张量），double 版只供读值，接不了图。
LATTICE_EXPORT long long lattice_mean_tensor(long long h) {
    LATTICE_TRY
    return lattice::put_tensor(lattice::tensor(h).mean());
    LATTICE_CATCH
}

LATTICE_EXPORT double lattice_max(long long h) {
    LATTICE_TRY
    return lattice::tensor(h).max().item<double>();
    LATTICE_CATCH
}

// ---- 变形 ------------------------------------------------------------------

LATTICE_EXPORT long long lattice_reshape(long long h, const char* shape) {
    LATTICE_TRY
    return lattice::put_tensor(
        lattice::tensor(h).reshape(lattice::parse_shape(shape)));
    LATTICE_CATCH
}

// 二维转置（sg 层的 @ 矩阵乘与转置配套的最小变形集）。
LATTICE_EXPORT long long lattice_transpose2(long long h) {
    LATTICE_TRY
    auto& t = lattice::tensor(h);
    if (t.dim() != 2) throw std::runtime_error("transpose2 仅支持二维张量");
    return lattice::put_tensor(t.transpose(0, 1));
    LATTICE_CATCH
}

// ---- 激活 ------------------------------------------------------------------

#define LATTICE_ACT(NAME, EXPR)                                                \
    LATTICE_EXPORT long long lattice_##NAME(long long h) {                     \
        LATTICE_TRY                                                            \
        return lattice::put_tensor(EXPR);                                      \
        LATTICE_CATCH                                                          \
    }

LATTICE_ACT(relu, at::relu(lattice::tensor(h)))
LATTICE_ACT(tanh, at::tanh(lattice::tensor(h)))
LATTICE_ACT(sigmoid, at::sigmoid(lattice::tensor(h)))

LATTICE_EXPORT long long lattice_softmax(long long h, long long dim) {
    LATTICE_TRY
    return lattice::put_tensor(
        at::softmax(lattice::tensor(h), static_cast<int64_t>(dim)));
    LATTICE_CATCH
}

#undef LATTICE_ACT
