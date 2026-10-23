# scourge-packages — Scourge 生态包仓库

Scourge 语言的官方包生态仓库。`sget` 默认从这里发现与拉取包：**任何人上传到本仓库的包，安装 sget 后即可直接使用**。

## 使用

```text
sget search <词>        # 浏览 packages/ 下的包
sget install <pkg>      # 安装（deps 自动先装，lock 记账）
sget list               # 已安装包（含版本/来源/作者/许可证）
```

安装落位：`packages/<pkg>/<pkg>.sg`（import 即用）+ `packages/<pkg>/bin/*.dll`（运行期动态加载，`sg build` 自动拷到产物目录）。

## 包发现机制

仓库无中心索引——每个包自带清单：

```text
packages/<pkg>/pkg.json   # 包清单：name/desc/author/repo/license/version/binding/deps/url/sha256
packages/<pkg>/<pkg>.sg   # 绑定源码（import <pkg> 即用）
packages/<pkg>/LICENSE    # 包作者自设许可证（必填）
```

sget 扫描 `packages/` 目录读取各 `pkg.json`；离线时回退到 sget 内置 registry。

## 权益与协议（超宽松）

- 仓库基建 MIT（见 [LICENSE](LICENSE)）；**每个包的许可证由作者自设**，`pkg.json` 的 `license` 字段（SPDX）与包内 `LICENSE` 文件必填。
- `author` / `repo` 字段必填：署名权随清单永久保留，`sget list/search` 展示。
- 包名先到先得（PR 合入即占名）。
- Release 二进制资产记 SHA256，sget 下载后校验，防供应链篡改。

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 首个官方包

| 包 | 说明 |
|---|---|
| [lattice](packages/lattice/) | PyTorch (LibTorch) 互操作桥：张量/JIT 推理/训练（autograd + SGD/Adam），纯 C++ API，无 Python 依赖 |
