# 侨乡 AI 智能讲解员（qx-rust）

开平碉楼文化的 AI 导览桌面应用：GPUI 原生窗口 + WebView 渲染 Three.js + VRM 数字人前端，内置离线语音识别（Whisper）与离线神经语音合成（VITS），断网也能完整导览。

## 架构

```
qx-rust/
├── crates/
│   ├── qx-core      # 架构无关核心：主题包 / 六级路由 / Ollama 客户端 / 离线知识库
│   ├── qx-desktop   # 主交付：gpui-kit 原生窗口 + WebView（跨平台语音链路都在这里）
│   └── embed-model  # 内嵌模型打包工具
├── web/             # 前端页面（Three.js + VRM 数字人 + 聊天 UI），编译期打进 exe
├── themes/          # 主题包（讲解词、知识库、测验题）
├── tts/             # VITS 模型（zh_CN-huayan-medium）+ espeak-ng G2P 数据
├── asr/             # Whisper 模型（ggml-base-q5_1）
├── prebuilt-voices/ # 预烘焙的断网语音（tts-cache 覆盖）
└── tools/           # voice_bake 等辅助脚本
```

- **web/ 与 themes/ 编译期打包进 exe，运行时释放到临时目录**——单个可执行文件自包含，不依赖 Python 或任何运行时环境。
- Windows 用 WebView2，Linux 用 WebKitGTK；画面与浏览器版完全一致。

## 语音链路（全部跨平台）

| 环节 | 实现 | 说明 |
|------|------|------|
| 语音识别（ASR） | `whisper-rs`（静态编译 whisper.cpp） | ggml-base-q5_1 模型，16kHz mono PCM，30s 上限，静音门限过滤 |
| 离线 TTS | `sherpa-rs`（sherpa-onnx VITS） | zh_CN-huayan-medium 模型，内置 C++ espeak-ng 做 G2P，静态链接进 exe |
| 在线 TTS | `msedge-tts`（纯 Rust） | 微软 Edge 在线语音（小伊），`QX_TTS_EDGE=off` 可强制全程离线 |
| 断网兜底 | `voice_bake` 预烘焙 | 赛前烘焙高频讲解词到 tts-cache，断网时按缓存键命中 |

环境变量调音（离线 VITS）：`QX_TTS_NOISE_SCALE` / `QX_TTS_LENGTH_SCALE` / `QX_TTS_NOISE_W`。

## 构建与运行

### 依赖

- Rust（edition 2024，建议最新 stable）
- CMake + C/C++ 工具链（首次编译静态链接 sherpa-onnx 与 whisper.cpp 用，各需几分钟）
- Linux 附加：WebKitGTK 运行库（Arch: `pacman -S webkit2gtk-4.1`）、clang/libclang
- Windows 附加：MSVC 工具链 + WebView2（Win10/11 自带）

### 命令

```bash
cargo run -p qx-desktop          # 运行桌面端
cargo test --workspace           # 全部测试（含 TTS/ASR 冒烟）
QX_HEADLESS=1 cargo run -p qx-desktop   # 无头模式（仅起 HTTP 服务，不开窗口）
```

LLM 后端默认指向本地 Ollama（`qwen3.5:4b`），无 Ollama 时自动回退到离线知识库问答。

## 平台注意事项

- **Linux + NVIDIA / Wayland**：程序启动时会自动设置 `WEBKIT_DISABLE_DMABUF_RENDERER=1`（WebKitGTK 的 DMABUF 渲染器在 NVIDIA 下分配 GBM buffer 失败会导致黑屏）并固定 X11 窗口模式（XWayland 下正常工作），无需手动干预。
- **Linux 静态链接**：`.cargo/config.toml` 为 Linux 目标设置 `relocation-model=dynamic-no-pic`（sherpa-onnx 静态库要求），对 Windows 构建无影响。
- 首次构建时 sherpa-rs 会用 CMake 源码编译 sherpa-onnx（含 onnxruntime），结果静态链接进 exe——不要使用其默认的 `download-binaries` 预编译包（缺静态库，且内置的裁剪版 onnxruntime 会触发 `GetElementType is not implemented` 崩溃）。

## 日志

按天滚动、保留 14 天，写入用户数据目录（Windows：`%LOCALAPPDATA%\QiaoxiangGuide\logs`；Linux：`~/.local/share/QiaoxiangGuide/logs`，可用 `QX_LOG_DIR` 覆盖）。文件落盘由自研异步日志库 [shen_eternity_log](https://github.com/EternityShen/shen_eternity_log) 承担（后台 tokio 写入线程，逐条 flush）。详见 [LOGGING.md](LOGGING.md)。
