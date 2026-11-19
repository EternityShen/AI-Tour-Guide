# 侨乡 AI 智能讲解员（qx-rust）

开平碉楼文化的 AI 导览桌面应用：GPUI 原生窗口 + WebView 渲染 Three.js + VRM 数字人前端，内置离线语音识别（Qwen3-ASR）与离线神经语音合成（VITS），断网也能完整导览。

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
├── asr/             # Qwen3-ASR 0.6B 模型（qwen3-asr-0.6b/）
├── prebuilt-voices/ # 预烘焙的断网语音（tts-cache 覆盖）
└── tools/           # voice_bake 等辅助脚本
```

- **web/ 与 themes/ 编译期打包进 exe，运行时释放到临时目录**——单个可执行文件自包含，不依赖 Python 或任何运行时环境。
- Windows 用 WebView2，Linux 用 WebKitGTK；画面与浏览器版完全一致。

## 语音链路（全部跨平台）

| 环节 | 实现 | 说明 |
|------|------|------|
| 语音识别（ASR） | `cpal` 采集 + `valle-asr`（Qwen3-ASR 0.6B，纯 Rust CPU 推理） | 后端直接打开系统默认麦克风，16kHz mono，能量 VAD 自动分段；模型完全本地加载，不联网、不调用外部 exe |
| 离线 TTS | `sherpa-rs`（sherpa-onnx VITS） | zh_CN-huayan-medium 模型，内置 C++ espeak-ng 做 G2P，静态链接进 exe |
| 在线 TTS | `msedge-tts`（纯 Rust） | 微软 Edge 在线语音（小伊），`QX_TTS_EDGE=off` 可强制全程离线 |
| 断网兜底 | `voice_bake` 预烘焙 | 赛前烘焙高频讲解词到 tts-cache，断网时按缓存键命中 |

环境变量调音（离线 VITS）：`QX_TTS_NOISE_SCALE` / `QX_TTS_LENGTH_SCALE` / `QX_TTS_NOISE_W`。

离线 ASR 模型默认从 `asr/qwen3-asr-0.6b`、`asr/Qwen3-ASR-0.6B` 或环境变量 `QX_ASR_MODEL_DIR` 指定的目录加载；目录内需有 `config.json`、`model.safetensors`，以及 `tokenizer.json`（或旧式 `vocab.json` + `merges.txt` + `tokenizer_config.json`）。模型不会在运行时下载。启动时默认用 0.5 秒静音预热权重，`QX_ASR_NO_WARMUP=1` 可跳过预热（首次说话会变慢）。

分发时有两种放法：把模型目录放在 exe 同级 `asr/qwen3-asr-0.6b`，exe 保持较小；或把模型放进源码树 `qx-rust/asr/` 并用 `QX_EMBED_ASR=1` 构建，打进 exe（单文件自包含，但 exe 会增大约 1.9 GB，链接更慢，且 >2 GB 的 exe 在部分 Windows 上会报「不是有效的应用程序」无法启动，因此默认不嵌入）。

## 构建与运行

### 依赖

- Rust（edition 2024，建议最新 stable）
- CMake + C/C++ 工具链（首次编译静态链接 sherpa-onnx 用，需几分钟；ASR 为纯 Rust，无需额外工具链）
- Linux 附加：WebKitGTK 运行库（Arch: `pacman -S webkit2gtk-4.1`）、ALSA 开发库（Arch: `alsa-lib`；Debian/Ubuntu: `libasound2-dev`）、clang/libclang
- Windows 附加：MSVC 工具链 + WebView2（Win10/11 自带）

### 命令

```bash
cargo run -p qx-desktop          # 运行桌面端
cargo test --workspace           # 全部测试（含 TTS/ASR 冒烟）
QX_HEADLESS=1 cargo run -p qx-desktop   # 无头模式（仅起 HTTP 服务，不开窗口）
```

LLM 后端默认指向本地 Ollama（`qwen3.5:4b`），无 Ollama 时自动回退到离线知识库问答。

## 本机性能实测（2026-11-19）

测试模型为 `qwen3.5:4b`，使用 50 张内容不同、统一为 768×768 的图片，在模型预热后逐张发送同一问题「请用一句话描述这张图片。」。计时从客户端发送 `POST /api/ask` 开始，到收到完整回答结束；50 次均成功。

| 指标 | 耗时 |
|------|-----:|
| 平均端到端耗时 | **2029.9 ms** |
| 服务端平均处理耗时 | 2026.8 ms |
| 中位数 P50 | 2023.0 ms |
| P95 | 2419.7 ms |
| 最小值 | 1557.7 ms |
| 最大值 | 2615.6 ms |

以上是本机、当前模型与测试图片条件下的结果，不作为其他硬件的性能保证。导览册中的「识图50图均值」对应本次预热后实测；单次冷启动不包含在 50 次均值内。

## 平台注意事项

- **Linux + NVIDIA / Wayland**：程序启动时会自动设置 `WEBKIT_DISABLE_DMABUF_RENDERER=1`（WebKitGTK 的 DMABUF 渲染器在 NVIDIA 下分配 GBM buffer 失败会导致黑屏）并固定 X11 窗口模式（XWayland 下正常工作），无需手动干预。
- **Linux 静态链接**：`.cargo/config.toml` 为 Linux 目标设置 `relocation-model=dynamic-no-pic`（sherpa-onnx 静态库要求），对 Windows 构建无影响。
- **macOS 麦克风权限**：cpal 直接访问系统输入设备；打包 .app 时需在 `Info.plist` 写入 `NSMicrophoneUsageDescription`，否则系统会拒绝采集。
- 首次构建时 sherpa-rs 会用 CMake 源码编译 sherpa-onnx（含 onnxruntime），结果静态链接进 exe——不要使用其默认的 `download-binaries` 预编译包（缺静态库，且内置的裁剪版 onnxruntime 会触发 `GetElementType is not implemented` 崩溃）。

## 日志

按天滚动、保留 14 天，写入用户数据目录（Windows：`%LOCALAPPDATA%\QiaoxiangGuide\logs`；Linux：`~/.local/share/QiaoxiangGuide/logs`，可用 `QX_LOG_DIR` 覆盖）。文件落盘由自研异步日志库 [shen_eternity_log](https://github.com/EternityShen/shen_eternity_log) 承担（后台 tokio 写入线程，逐条 flush）。详见 [LOGGING.md](LOGGING.md)。
