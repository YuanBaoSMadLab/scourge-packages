# lattice — PyTorch (LibTorch) FFI 桥

Scourge 语言直接调用 PyTorch 的官方桥：**纯 LibTorch C++ API，无 Python 依赖**。张量算子、自动求导、SGD/Adam 优化器、TorchScript 推理与序列化，全部经 C ABI 导出，`import lattice` 即用。

## 能力面（53 导出）

| 分块 | 数量 | 内容 |
|---|---|---|
| core | 9 | version / init / 错误态协议 / 句柄生命周期 / build_info |
| tensor | 29 | 创建（zeros/ones/randn/from_data）、算术与 matmul、归约（sum/mean/max）、形状（reshape/transpose）、激活（relu/tanh/sigmoid/softmax）、零拷贝互转 |
| train | 15 | autograd（requires_grad/backward/grad）、参数组与 SGD/Adam、JIT 编译/运行/加载、torch::save/load 往返 |

## 安装与使用

```text
sget install lattice
```

```scourge
import lattice

capability ffi

def main() -> int:
    if lattice.init_guard("2.13.0") != 1:
        return 1
    a: int = lattice.ones("3x3", "float")
    s: int = lattice.add(a, lattice.scalar_f(1.0))
    print(lattice.t_sum(s))          # 18.0（逐元素广播）
    lattice.free_all()
    return 0
```

## 运行前提

- `packages/lattice/bin/lattice.dll` 已随安装部署到项目根（`sg build` 自动拷贝产物目录）。
- 本机 LibTorch 的 `lib` 目录（含 `torch_cpu.dll`/`c10.dll`/`torch.dll`）在 `PATH`。
- **torch 版本锁定**：shim 按编译期 pin 的版本链接，版本守卫三层拦截错配——编译期宏、DLL 内 `ones(2x3).sum()==6` 冒烟、`init_guard` 逐字比对并醒目打印。升级 torch 后重编 shim 即可（见下）。

## 关键契约

- **句柄**：所有张量/优化器/JIT 实体经 `int` 句柄引用，从 1 递增；`free` 释放单个，`free_all` 全清（含优化器/JIT）。
- **shape 传字符串**：`"3"`、`"2x3"`；**dtype 只有 `"int"`（int64）与 `"float"`（float64）**。
- **零拷贝双向**：`from_data(data_ptr(arr), ...)` 借入 Scourge 宿主数组不拷贝；`copy_to_ptr` 把 torch 侧结果写回宿主。张量字面量语法 `Tensor[float, (2, 3)] = [[...], [...]]`。
- **错误态协议**：失败返回 `0` 并置错误缓冲，`last_err()` 取回，`clear_err()` 清除；单线程契约。
- **内建名冲突**：`sum/mean/max/reshape/tanh` 是语言内建，绑定侧用 `t_` 前缀（`t_sum/t_mean/t_mean_t/t_max/t_reshape/t_tanh`）；梯度快照需 0-dim 损失张量接图时用 `t_mean_t`（返回句柄）而非 `t_mean`（返回 float）。

## 从源码重编 shim

```text
cd shim && build_shim.bat
```

环境变量：`TORCH_ROOT`（默认探测 miniconda torch 包）、`LATTICE_PIN`（默认 `2.13.0`）。只需 `clang++` 与本机 LibTorch，无 Python 参与。

## 许可证

MIT（作者 YuanBaoSMadLab）。升级与问题反馈见 `pkg.json` 的 `repo` 字段。
