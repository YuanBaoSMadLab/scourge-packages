// ============================================================================
//  lattice_core.cpp - Lattice 桥·生命周期与版本域（9 导出）
//
//  职责边界：DLL 版本协商（编译期 pin / 数值版）、init 冒烟（版本守卫第二层）、
//  错误态读写、句柄释放。张量域见 lattice_tensor.cpp，训练/JIT 域见
//  lattice_train.cpp；ABI 契约与共享内核见 lattice_shim.hpp（单一事实源）。
// ============================================================================
#include "lattice_shim.hpp"

#include <torch/version.h>  // TORCH_VERSION：编译期 SDK 版本串（进 build_info）

namespace {

// init 冒烟：真实拉起一次 ATen 计算并校验结果。若 shim 编译 SDK 与运行期
// torch.dll 版本错配，在此暴露为异常/算错，而不是业务中途的静默数据损坏。
int smoke() {
    auto t = at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
    const double s = t.sum().item<double>();
    if (s != 6.0)
        throw std::runtime_error("init 冒烟失败：ones(2x3).sum() != 6（ABI 错配？）");
    return 1;
}

} // namespace

// ---- 版本域 ----------------------------------------------------------------
// 返回编译期锁定 torch 版本（如 "2.13.0"）。.sg 侧 init(expected) 用它逐字
// 校验；运行期实际加载的 torch 信息见 lattice_build_info。宏常量，无异常路径。
LATTICE_EXPORT const char* lattice_version() { return LATTICE_PIN_STR; }

// 数值版：major*10000 + minor*100 + patch（2.13.0 → 21300），便于程序化比较。
LATTICE_EXPORT long long lattice_version_int() {
    int ma = 0, mi = 0, pa = 0;
    std::sscanf(TORCH_VERSION, "%d.%d.%d", &ma, &mi, &pa);
    return ma * 10000LL + mi * 100LL + pa;
}

// 版本守卫第二层：创建张量 + 求和冒烟。成功返回 1，失败置 err 返回 0。
LATTICE_EXPORT long long lattice_init() {
    LATTICE_TRY
    return smoke();
    LATTICE_CATCH
}

// 编译全景：shim 版本 / pin / SDK 版本 / 构建时间。诊断"挨骂"现场用。
LATTICE_EXPORT const char* lattice_build_info() {
    static const std::string info =
        std::string("lattice " LATTICE_SHIM_VERSION " | torch pin " LATTICE_PIN_STR
                    " | torch sdk " TORCH_VERSION " | built " __DATE__ " " __TIME__);
    return info.c_str();
}

// ---- 错误态（纯缓冲读写，无抛点，不包裹） ----------------------------------
LATTICE_EXPORT const char* lattice_last_err() {
    return lattice::err_buf().c_str();
}
LATTICE_EXPORT void lattice_clear_err() { lattice::err_buf().clear(); }

// ---- 句柄生命周期 ----------------------------------------------------------
// free 仅注销注册表条目：torch 堆内存随之析构；from_data 借用视图的宿主
// 内存归 Scourge，绝不触碰。已释放再 free 置 err 返回 0（幂等保护）。
LATTICE_EXPORT long long lattice_free(long long h) {
    LATTICE_TRY
    auto& m = lattice::reg();
    auto it = m.find(h);
    if (it == m.end()) {
        lattice::set_err("句柄不存在或已释放");
        return 0;
    }
    m.erase(it);
    return 1;
    LATTICE_CATCH
}

// 存活查询：1/0，查询语义不置 err。
LATTICE_EXPORT long long lattice_handle_alive(long long h) {
    return lattice::reg().count(h) ? 1LL : 0LL;
}

// 全量释放（脚本退出前清扫），返回释放条数。
LATTICE_EXPORT long long lattice_free_all() {
    LATTICE_TRY
    const long long n = static_cast<long long>(lattice::reg().size());
    lattice::reg().clear();
    return n;
    LATTICE_CATCH
}
