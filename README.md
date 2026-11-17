# AI-Tour-Guide · 侨乡·江门 AI 智能讲解员

开平碉楼文化的 AI 导览桌面应用：gpui-kit 原生窗口 + WebView 渲染 **Three.js + VRM 数字人** 前端，内置离线语音识别（Whisper）与离线神经语音合成（VITS），**断网也能完整导览**。

单个可执行文件自包含，前端页面与主题包在编译期打进 exe，运行时释放到临时目录，不依赖 Python 或任何运行时环境。

---

## 一、功能特点

| 特性 | 说明 |
|------|------|
| **跨平台桌面端** | Windows 使用 WebView2，Linux 使用 WebKitGTK；画面与浏览器网页版完全一致 |
| **Three.js 3D 场景** | 开平碉楼、赤坎骑楼街、梁启超故居、新会陈皮、侨批银信、小鸟天堂等展品原创 GLB 模型 |
| **VRM 数字人** | 内置「小侨」AI 智能讲解员，支持表情与手势，随讲解内容动态变化 |
| **离线语音识别 (ASR)** | whisper-rs（静态编译 whisper.cpp），ggml-base-q5_1，16 kHz mono PCM，30 s 上限，静音门限过滤 |
| **离线语音合成 (TTS)** | sherpa-rs（sherpa-onnx VITS），zh_CN-huayan-medium 女声，内置 espeak-ng G2P，静态链接进 exe |
| **在线 TTS 兜底** | msedge-tts（纯 Rust），微软 Edge 在线语音（小伊）；`QX_TTS_EDGE=off` 可强制全程离线 |
| **断网预烘焙语音** | `voice_bake` 赛前把高频讲解词固化为本地 WAV，断网时按缓存键命中，无需联网 |
| **本地大模型 (LLM)** | 默认走本地 Ollama（`qwen3.5:4b`）；支持 `--model <GGUF>` 本地内嵌推理（Qwen2.5-VL-3B，无需 Ollama） |
| **离线知识库回退** | 无 Ollama 时自动回到离线知识库问答，脱网也能答展馆相关问题 |
| **零 Python / 零运行时** | `web/` 与 `themes/` 编译期打包进单文件 exe，拷贝即可运行 |

---

## 二、项目结构

```
AI-Tour-Guide/
├── README.md                 # 本文件
├── LICENSES.md               # 开源组件与素材授权清单
├── .gitignore
├── qx-rust/                  # Rust 桌面端工作空间
│   ├── Cargo.toml            # 工作空间根：qx-core / qx-desktop / embed-model
│   ├── crates/
│   │   ├── qx-core           # 架构无关核心层：主题包 / 六级路由 / Ollama 客户端 / 离线知识库
│   │   ├── qx-desktop        # 主交付：gpui-kit 原生窗口 + WebView（跨平台语音链路都在这里）
│   │   └── embed-model       # 内嵌 GGUF + mmproj 追加到 exe 尾部（PE overlay）的打包工具
│   ├── web/                  # 前端页面（Three.js + VRM 数字人 + 聊天 UI），编译期打进 exe
│   ├── themes/               # 主题包（讲解词、知识库、测验题）
│   ├── asr/                  # Whisper 模型（ggml-base-q5_1，Git LFS / 本地存放，不进源码）
│   ├── tts/                  # VITS 模型（zh_CN-huayan-medium）+ espeak-ng G2P 数据（同上）
│   ├── prebuilt-voices/      # 预烘焙的断网语音（tts-cache 覆盖）
│   ├── bevy-assets/models/   # 原创 3D 展品模型（GLB）
│   ├── target/               # 构建产物（不进源码）
│   ├── voice_bake_report.txt # 语音烘焙报告（运行时生成，不进源码）
│   └── LOGGING.md            # 日志系统说明
```

---

## 三、架构概览

```
┌─────────────────────────────────────────────────────────┐
│              qx-desktop (Rust gpui-kit + WebView)       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  原生窗口    │  │   WebView   │  │   本地 HTTP 服务   │ │
│  │ (Windows:   │  │ (WebView2 / │  │  theme / kb /   │ │
│  │  WebView2)  │  │  WebKitGTK) │  │  Ollama / LLM   │ │
│  └─────────────┘  └─────────────┘  │  ASR / TTS /    │ │
│                                    │  voice_bake ... │ │
│  渲染 web/ 页面（Three.js + VRM）   └─────────────────┘ │
│        (编译期打进 exe，运行时释放到临时目录)           │
└─────────────────────────────────────────────────────────┘
```

---

## 四、语音链路（全部跨平台）

| 环节 | 实现 | 说明 |
|------|------|------|
| 语音识别（ASR） | `whisper-rs`（静态编译 whisper.cpp） | ggml-base-q5_1 模型，16kHz mono PCM，30 s 上限，静音门限过滤 |
| 离线 TTS | `sherpa-rs`（sherpa-onnx VITS） | zh_CN-huayan-medium 模型，内置 C++ espeak-ng 做 G2P，静态链接进 exe |
| 在线 TTS | `msedge-tts`（纯 Rust） | 微软 Edge 在线语音（小伊），`QX_TTS_EDGE=off` 可强制全程离线 |
| 断网兜底 | `voice_bake` 预烘焙 | 赛前烘焙高频讲解词到 `tts-cache`，断网时按缓存键命中 |

环境变量调音（离线 VITS）：`QX_TTS_NOISE_SCALE` / `QX_TTS_LENGTH_SCALE` / `QX_TTS_NOISE_W`。

---

## 五、快速开始

### 1. 依赖

- Rust（edition 2024，建议最新 stable）
- CMake + C/C++ 工具链（首次编译静态链接 sherpa-onnx 与 whisper.cpp 用，各需几分钟）

**Linux 附加：**
- WebKitGTK 运行库（Arch: `pacman -S webkit2gtk-4.1`）
- clang / libclang

**Windows 附加：**
- MSVC 工具链 + WebView2（Win10/11 自带）

### 2. 构建与运行

```bash
# 进入 Rust 工作空间
cd qx-rust

# 运行桌面端（默认端口 8765，走本地 Ollama / qwen3.5:4b）
cargo run -p qx-desktop

# 全部测试（含 TTS/ASR 冒烟）
cargo test --workspace

# 无头模式（仅起 HTTP 服务，不开窗口，便于 curl /api/ask 验证）
QX_HEADLESS=1 cargo run -p qx-desktop

# 发布构建（轻量优化，缩短编译时间并减小 exe 体积）
cargo build --release -p qx-desktop
```

### 3. 大模型后端

LLM 后端默认指向本地 Ollama（`qwen3.5:4b`），无 Ollama 时自动回退到离线知识库问答。

启用本地 GGUF 内嵌推理（需 `cargo build --features local-llm`，构建机需装 C/C++ 工具链）：

```bash
cargo run -p qx-desktop --features local-llm -- --model /path/to/model.gguf
```

---

## 六、命令行参数

运行 `qx-desktop` 时支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bake-voices` | 赛前联网跑一次：把现场会说的每句话固化成语音并写报告文件 | — |
| `--model <路径>` | 使用本地 GGUF 模型（需 `--features local-llm` 构建） | — |
| `--ollama-url <URL>` | Ollama / llama-server 兼容协议地址 | `http://127.0.0.1:9931` |
| `--ollama-model <名>` | 模型名 | `qwen3.5:4b` |
| `--llm-backend <auto\|ollama\|llama\|local\|none>` | LLM 后端选择 | `auto` |

---

## 七、环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `QX_PORT` | HTTP 服务端口（被占用时自动顺延 8765~8772） | `8765` |
| `QX_HEADLESS=1` | 无头模式，只起 HTTP 不创建窗口 | — |
| `QX_LOG_DIR` | 日志目录 | Windows: `%LOCALAPPDATA%\QiaoxiangGuide\logs`<br>Linux: `~/.local/share/QiaoxiangGuide/logs` |
| `QX_TTS_EDGE=off` | 强制全程离线 TTS（禁用微软 Edge 在线语音） | — |
| `QX_TTS_NOISE_SCALE` / `QX_TTS_LENGTH_SCALE` / `QX_TTS_NOISE_W` | 离线 VITS 调音参数 | — |
| `QX_NO_EMBEDDED_MODEL=1` | 关闭内嵌 Qwen2.5-VL-3B 模型，回退到 Ollama / `--model` | — |
| `QX_LLM_SELFTEST=1` | 本地模型自检（仅 `local-llm` 构建可用） | — |
| `QX_VOICE_BAKE_DIR` | 语音烘焙输出目录 | 编译时源码根 |

---

## 八、平台注意事项

### Linux + NVIDIA / Wayland

程序启动时会自动设置 `WEBKIT_DISABLE_DMABUF_RENDERER=1`（WebKitGTK 的 DMABUF 渲染器在 NVIDIA 下分配 GBM buffer 失败会导致黑屏），并固定 X11 窗口模式（XWayland 下正常工作），无需手动干预。

### Linux 静态链接

`.cargo/config.toml` 为 Linux 目标设置 `relocation-model=dynamic-no-pic`（sherpa-onnx 静态库要求），对 Windows 构建无影响。

### 首次构建

sherpa-rs 会用 CMake 源码编译 sherpa-onnx（含 onnxruntime），结果静态链接进 exe。**不要使用其默认的 `download-binaries` 预编译包**（缺静态库，且内置的裁剪版 onnxruntime 会触发 `GetElementType is not implemented` 崩溃）。

---

## 九、日志

按天滚动、保留 14 天，写入用户数据目录（可用 `QX_LOG_DIR` 覆盖）。文件落盘由自研异步日志库 [`shen_eternity_log`](https://github.com/EternityShen/shen_eternity_log) 承担（后台 tokio 写入线程，逐条 flush）。详见 `qx-rust/LOGGING.md`。

---

## 十、授权与第三方组件

本项目使用的开源组件与素材授权清单见 [`LICENSES.md`](LICENSES.md)，主要包括：

- **Three.js** (MIT) — 3D 渲染
- **@pixiv/three-vrm** (MIT) — VRM 数字人加载与表情
- **sherpa-onnx** (Apache-2.0) — 离线语音识别
- **Piper / VITS** (MIT) — 离线神经语音合成
- **ONNX Runtime** (MIT) — 神经网络推理运行时
- **espeak-ng** (GPL-3.0) — 中文音素化
- **msedge-tts** (LGPL-3.0) — 构建期批量生成讲解语音（仅赛前使用，未随 exe 分发）
- **whisper.cpp** — 离线语音识别底层

展品三维模型、讲解词与问答库、场景与展台均为团队原创，详见 `LICENSES.md`。

---

## 十一、许可证

本项目采用 [MIT License](LICENSE) 开源协议。

第三方开源组件及素材的授权详见 [`LICENSES.md`](LICENSES.md)，其中部分组件（如 espeak-ng GPL-3.0、msedge-tts LGPL-3.0）遵循其各自原始许可证，项目本身代码按 MIT 分发。

---

> 侨乡·江门 AI 智能讲解员 —— 让碉楼、骑楼、陈皮、侨批，在离线也能说出口。
