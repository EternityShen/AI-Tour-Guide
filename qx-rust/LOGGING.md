# 运行日志

主程序 `qx-desktop` 已接入 [tracing](https://docs.rs/tracing)，日志默认落盘，GUI 没有控制台也能排查现场问题。

## 日志位置

```
%LOCALAPPDATA%\QiaoxiangGuide\logs\qx-guide.log.YYYY-MM-DD
```

例如：

```
C:\Users\<用户名>\AppData\Local\QiaoxiangGuide\logs\qx-guide.log.2026-11-16
```

- 按天滚动，一个进程一天一个文件；
- 启动时自动删除 14 天前的旧日志；
- 「自检」面板会显示当前日志文件路径（`/api/stats` 的 `log_file` 字段）；
- 首选目录不可写时自动退到 `%TEMP%\QiaoxiangGuide\logs`。

## 环境变量

| 变量 | 作用 | 示例 |
|---|---|---|
| `QX_LOG_DIR` | 指定日志目录（便携版可指到 U 盘） | `D:\qx-logs` |
| `QX_LOG` | 过滤级别，语法同 `RUST_LOG`，优先级最高 | `debug`、`qx_desktop=debug,warn` |
| `RUST_LOG` | 备用过滤级别，`QX_LOG` 未设置时生效 | `info` |

默认过滤规则：

```
warn,qx_desktop=info,qx_core=info
```

即：本项目 `info` 及以上，依赖库只保留 `warn`，避免刷屏。

调试示例（PowerShell）：

```powershell
$env:QX_LOG = "debug"
.\QiaoxiangGuide-Local.exe
```

> `debug` 级别会记录观众提问原文，仅排障时开启；默认 `info` 只记录字数、图片数、耗时等元数据。

## 记录了什么

- 启动/退出、版本、日志目录、端口；
- 主题包切换、LLM 后端与模型名、模型不可用原因；
- ASR / Piper / WinRT 小伊 / 内置模型 / 视觉旁路的加载结果与耗时；
- 每次 HTTP 请求：方法、路径、状态码、耗时；
  - `/api/*` 正常请求记 `info`；
  - 静态资源记 `debug`（默认不写文件）；
  - 4xx 记 `warn`，5xx 记 `error`；
- 每次提问：字数、图片数、展品、深度、动作、总耗时、回复字数、音频字节数；
- 大模型调用失败、TTS 全链路失败、限流等异常；
- 前端未捕获异常、未处理的 Promise 拒绝、资源加载失败（经 `/api/log` 回传）；
- panic：线程名、文件位置、backtrace。

## 前端错误回传

`web/src/log.js` 在 `index.html` 中优先加载，监听：

- `window.onerror`（含脚本/图片/模型等资源加载失败）；
- `unhandledrejection`；
- 业务代码里主动调用的 `window.qxLog(level, message, { source, stack })`。

回传地址是本机 `POST /api/log`，只在 `127.0.0.1` 上可用；服务端会截断超长消息与堆栈，前端 60 秒内最多发 30 条并做 10 秒去重。日志失败静默，不影响讲解。

## 相关代码

- `crates/qx-desktop/src/logging.rs`：初始化、滚动文件、过滤、panic hook；
- `crates/qx-desktop/src/http.rs`：访问日志、`/api/log`、问答/TTS 日志；
- `crates/qx-core/src/nlu.rs`、`llm.rs`：大模型失败与调用耗时；
- `web/src/log.js`：前端错误采集。

## 注意

- 时间为 UTC（日志行尾的 `Z`）。
- 首次构建会新增 `tracing-subscriber` / `tracing-appender` 依赖并更新 `Cargo.lock`，需要能访问 crates.io（或已配置镜像）。
