# 贡献指南

欢迎向 scourge-packages 上传包。协议刻意保持超宽松：不签 CLA/DCO，靠清单声明与署名保障权益。

## 五条约定

1. **许可证自设**：包内 `LICENSE` 文件 + `pkg.json` 的 `license`（SPDX 标识）必填。选哪种协议由作者决定，sget 只负责展示与透传。
2. **署名保留**：`pkg.json` 的 `author`（作者/组织）与 `repo`（源仓库 URL）必填，合入后不可移除——这是贡献者署名权的存档。
3. **包名先到先得**：PR 合入即占名；改名 = 删旧包 + 新包 PR。
4. **轻审核**：PR 满足下面清单即可合入——
   - 目录布局合规（见下）；
   - `license` 字段与 `LICENSE` 文件在；
   - 至少一个探针或示例（`<pkg>_demo.sg`）可跑通。
5. **供应链完整性**：二进制走 GitHub Release 资产，`pkg.json` 记 `url` + `sha256`；sget 下载后校验。

## 包布局

```text
packages/<pkg>/
  pkg.json          # 清单（字段见下）
  LICENSE           # 作者自设许可证
  <pkg>.sg          # 绑定源码：import <pkg> 即用；capability 只授最小集
  README.md         # 用法与 API 契约（句柄生命周期/错误约定/线程契约）
  <pkg>_demo.sg     # 示例（审核最低要求）
  shim/             # C/C++ shim 源码（api=cpp/com 的库必备）+ build_shim.bat
  bin/              # 预编译产物（不入 git，走 Release 资产）
```

## pkg.json 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 包名（=目录名，先到先得） |
| `desc` | 是 | 一句话描述（sget search 展示） |
| `author` | 是 | 作者/组织（署名存档） |
| `repo` | 是 | 源仓库 URL |
| `license` | 是 | SPDX 标识（与包内 LICENSE 一致） |
| `version` | 是 | semver |
| `binding` | 是 | `builtin`（绑定在 sget 内）/ `bundled`（包内自带 `<pkg>.sg`）/ `ecosystem`（同 bundled，包仓库驱动） |
| `deps` | 否 | 依赖包名数组，安装时先装 |
| `local_env` | 否 | 环境变量名：指向本机库根目录时跳过下载（本地模式） |
| `version_query` | 否 | 线上安装的版本探测策略（如 `libtorch-cpu-latest`） |
| `pin` | 否 | 探测失败时的回退锁定版本 |
| `url` | 否 | 二进制资产 URL（Release） |
| `sha256` | 否 | 资产校验和 |
| `files` | 否 | 需部署的源文件（缺省 `<pkg>.sg`） |

## shim 选型判据

- 库是**纯 C API**：优先 bindgen（`<pkg>.bind` 声明，单表达式体）——工具自动生成 `.sg` 绑定与代理 shim。
- 库是 **C++ ABI / COM**：手写 shim（extern "C" 导出 + 句柄注册表 + try/catch 边界），参考 [lattice](packages/lattice/shim/)。bindgen 的单表达式 DSL 表达不了异常边界与句柄生命周期管理。
- 约定：句柄 `long long`（0 = 失败）；shape/复合参数用字符串；每函数 try/catch，错误态经 `last_err()` 查询；str 返回用独立 static 缓冲。

## 版本升级流程（ABI 耦合库，如 lattice）

改 `pin` → 重编 shim（`shim/build_shim.bat`）→ 探针回归全绿 → PR 改 `url`/`sha256`。
