// ============================================================================
//  lattice_train.cpp - Lattice 桥·训练与 JIT 域（15 导出）
//
//  职责边界：autograd 开关/反传/梯度快照、参数组、SGD/Adam 优化器、
//  TorchScript 内存编译执行（无 Python）、磁盘模型加载、张量序列化。
//  张量域见 lattice_tensor.cpp，生命周期/版本域见 lattice_core.cpp；
//  ABI 契约与共享内核见 lattice_shim.hpp（单一事实源）。
// ============================================================================
#include "lattice_shim.hpp"

// script.h 不携带 optim/serialize/api-jit（实测聚合头清单），train 域显式补齐：
#include <torch/jit.h>       // torch::jit::compile（内存 TorchScript）
#include <torch/optim.h>     // torch::optim::SGD/Adam（legacy 原地更新契约）
#include <torch/serialize.h> // torch::save/load（张量级序列化往返）

namespace lattice {

// 参数组：train 域私有值模型（仅本 TU 可见；句柄经 shared_ptr<void> 不透明
// 传递）。params 拷贝进优化器时为 Tensor 浅拷贝（共享 TensorImpl），
// opt.step() 的原地更新对各句柄与参数组本身均可见。
struct ParamGroup {
    std::vector<at::Tensor> params;
};

} // namespace lattice

// ---- autograd --------------------------------------------------------------

// 开关叶子张量的求导（from_data 借用视图也是叶子，可直接开启）。
LATTICE_EXPORT long long lattice_requires_grad(long long h, long long flag) {
    LATTICE_TRY
    lattice::tensor(h).requires_grad_(flag != 0);
    return 1;
    LATTICE_CATCH
}

// 反传。仅标量损失允许无参调用（ATen 语义）；非标量报错——调用方显式
// 传全 1 梯度即可，shim 不做隐式广播。
LATTICE_EXPORT long long lattice_backward(long long h) {
    LATTICE_TRY
    lattice::tensor(h).backward();
    return 1;
    LATTICE_CATCH
}

// 梯度快照：clone 语义——取出后 zero_grad/再次 backward 不影响已取句柄；
// 需要最新值就重新取。未反传或未开求导时报错（而非返回 0 句柄）。
LATTICE_EXPORT long long lattice_grad(long long h) {
    LATTICE_TRY
    auto g = lattice::tensor(h).grad();
    if (!g.defined())
        throw std::runtime_error("无梯度：尚未 backward 或未 requires_grad");
    return lattice::put_tensor(g.clone());
    LATTICE_CATCH
}

// ---- 参数组与优化器 --------------------------------------------------------

LATTICE_EXPORT long long lattice_param_group_new() {
    LATTICE_TRY
    return lattice::put(std::make_shared<lattice::ParamGroup>(),
                        lattice::Kind::ParamGroup);
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_param_group_add(long long pg, long long t) {
    LATTICE_TRY
    auto p = std::static_pointer_cast<lattice::ParamGroup>(
        lattice::get(pg, lattice::Kind::ParamGroup));
    p->params.push_back(lattice::tensor(t));
    return 1;
    LATTICE_CATCH
}

// legacy 优化器（torch::optim::SGD/Adam）：step() 对参数原地更新——
// 若参数来自 from_data 借用视图，更新直接写入 Scourge 宿主内存（f2t/t2f
// 双通道语义下所见即所得）。
LATTICE_EXPORT long long lattice_sgd_new(long long pg, double lr) {
    LATTICE_TRY
    auto p = std::static_pointer_cast<lattice::ParamGroup>(
        lattice::get(pg, lattice::Kind::ParamGroup));
    return lattice::put(
        std::make_shared<torch::optim::SGD>(p->params,
                                            torch::optim::SGDOptions(lr)),
        lattice::Kind::Optimizer);
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_adam_new(long long pg, double lr) {
    LATTICE_TRY
    auto p = std::static_pointer_cast<lattice::ParamGroup>(
        lattice::get(pg, lattice::Kind::ParamGroup));
    return lattice::put(
        std::make_shared<torch::optim::Adam>(p->params,
                                             torch::optim::AdamOptions(lr)),
        lattice::Kind::Optimizer);
    LATTICE_CATCH
}

// zero_grad（set_to_none 默认语义）：旧梯度句柄是 clone 快照，不受影响。
LATTICE_EXPORT long long lattice_opt_zero_grad(long long opt) {
    LATTICE_TRY
    std::static_pointer_cast<torch::optim::Optimizer>(
        lattice::get(opt, lattice::Kind::Optimizer))
        ->zero_grad();
    return 1;
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_opt_step(long long opt) {
    LATTICE_TRY
    std::static_pointer_cast<torch::optim::Optimizer>(
        lattice::get(opt, lattice::Kind::Optimizer))
        ->step();
    return 1;
    LATTICE_CATCH
}

// ---- TorchScript（内存编译，无 Python） ------------------------------------

// 编译 TorchScript 源码串 → CompilationUnit。源码内可用 torch.* 内建；
// 返回值 IValue 非 Tensor 时 run 侧报错。
LATTICE_EXPORT long long lattice_jit_compile(const char* src) {
    LATTICE_TRY
    if (src == nullptr || *src == '\0')
        throw std::runtime_error("TorchScript 源码为空");
    auto cu = torch::jit::compile(src);
    if (!cu) throw std::runtime_error("TorchScript 编译返回空单元");
    return lattice::put(cu, lattice::Kind::JitUnit);
    LATTICE_CATCH
}

// 执行编译单元中的方法：实参张量 → IValue 栈 → 结果须为 Tensor。
LATTICE_EXPORT long long lattice_jit_run(long long cu, const char* method,
                                         long long x) {
    LATTICE_TRY
    auto unit = std::static_pointer_cast<torch::jit::CompilationUnit>(
        lattice::get(cu, lattice::Kind::JitUnit));
    auto out = unit->run_method(c10::QualifiedName(method),
                                lattice::tensor(x))
                   .toTensor();
    return lattice::put_tensor(std::move(out));
    LATTICE_CATCH
}

// 加载磁盘序列化模块（Python 侧 torch.save 的 .pt 资产）。探针覆盖内存
// compile 路径；本入口面向外部模型资产分发。torch 2.13 的 jit::load 返回
// Module 值（非 shared_ptr），句柄注册需再包一层堆所有权。
LATTICE_EXPORT long long lattice_jit_load(const char* path) {
    LATTICE_TRY
    if (path == nullptr || *path == '\0') throw std::runtime_error("模型路径为空");
    auto m = torch::jit::load(path);
    return lattice::put(std::make_shared<torch::jit::Module>(std::move(m)),
                        lattice::Kind::JitModule);
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_jit_forward(long long m, long long x) {
    LATTICE_TRY
    auto mod = std::static_pointer_cast<torch::jit::Module>(
        lattice::get(m, lattice::Kind::JitModule));
    return lattice::put_tensor(
        mod->forward({lattice::tensor(x)}).toTensor());
    LATTICE_CATCH
}

// ---- 张量序列化（torch::save/load 往返） -----------------------------------

LATTICE_EXPORT long long lattice_save(long long h, const char* path) {
    LATTICE_TRY
    if (path == nullptr || *path == '\0') throw std::runtime_error("保存路径为空");
    torch::save(lattice::tensor(h), path);
    return 1;
    LATTICE_CATCH
}

LATTICE_EXPORT long long lattice_load(const char* path) {
    LATTICE_TRY
    if (path == nullptr || *path == '\0') throw std::runtime_error("加载路径为空");
    at::Tensor t;
    torch::load(t, path);
    return lattice::put_tensor(std::move(t));
    LATTICE_CATCH
}
