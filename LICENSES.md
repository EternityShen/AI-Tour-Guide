# 开源组件与素材授权清单

本作品「侨乡·江门 AI 智能讲解员」由参赛团队自主开发。以下清单对应当前 Rust 桌面端的实际依赖与素材，供知识产权核验；各组件保持其原始许可证。

## 一、当前开源组件

| 组件 | 版本 | 当前用途 | 许可证 | 项目地址 |
|---|---|---|---|---|
| Three.js | r187 | Three.js 三维场景与展项渲染 | MIT | https://github.com/mrdoob/three.js |
| @pixiv/three-vrm | 3.5.5 | VRM 骨架、表情与动作驱动 | MIT | https://github.com/pixiv/three-vrm |
| valle-asr | 0.2.0 | Qwen3-ASR 纯 Rust 本地推理 | Apache-2.0 | https://crates.io/crates/valle-asr |
| cpal | 0.18 | 系统麦克风采集 | Apache-2.0 | https://github.com/RustAudio/cpal |
| Qwen3-ASR-0.6B | — | 离线语音识别模型权重 | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-ASR-0.6B |
| Piper | — | VITS 声学模型与语音前端格式 | MIT | https://github.com/rhasspy/piper |
| ONNX Runtime | 1.24.1 | Piper 声学模型推理，动态加载随包 DLL | MIT | https://github.com/microsoft/onnxruntime |
| espeak-ng | 1.52 | 中文音素化，随包使用 `espeak-ng.dll` 与数据目录 | GPL-3.0 | https://github.com/espeak-ng/espeak-ng |
| Ollama | — | 可选的本地大模型运行时 | MIT | https://github.com/ollama/ollama |
| Qwen3.5 4B | — | 默认本地自由问答与视觉模型 | Apache-2.0 | https://ollama.com/library/qwen3.5 |
| llama.cpp（可选） | — | `local-llm` feature 的 GGUF 内嵌推理 | MIT | https://github.com/ggml-org/llama.cpp |
| shen_eternity_log | — | 后台异步日志落盘 | 见源仓库 | https://github.com/EternityShen/shen_eternity_log |

当前正式运行链路不依赖 Python、FastAPI、Uvicorn、jieba、Whisper、Edge TTS 或外部 `piper.exe`。源码中保留的旧文件或未启用可选 feature，不应被理解为当前默认运行依赖。

## 二、自建内容（团队原创）

| 内容 | 说明 |
|---|---|
| 展品三维模型 | 开平碉楼、赤坎古镇、梁启超故居、新会陈皮、侨批银信、小鸟天堂等 GLB 展项模型 |
| 讲解员外观 | `mascot.glb` 吉祥物外观按部位绑定到 VRM 骨架，复用同一套口型、手势、待机与视线逻辑 |
| 讲解词与知识库 | 展项介绍、FAQ 和互动题面依据公开资料撰写并写入 `qx-rust/themes/qiaoxiang/theme.json` |
| 场景与界面 | Three.js 场景、展台、灯光、交互界面与演示流程由团队设计实现 |

## 三、VRM 模型

`qx-rust/web/assets/models/avatar.vrm` 用于讲解员的基础骨架与动作承载，遵循其原始授权条款，仅用于本作品演示，不用于商业发行；默认视觉外观由 `mascot.glb` 接管。

## 四、字体与音频

- 界面字体使用操作系统自带字体，未另行打包分发商业字体文件。
- `qx-rust/prebuilt-voices/` 中的 130 条 WAV 使用本地 Piper `zh_CN-huayan-medium` 离线生成，随程序分发。
- 缓存未命中的新回答仍由同一本地 Piper 模型实时合成：espeak-ng 音素化后交由 ONNX Runtime 推理，输出 22050 Hz 单声道 WAV。
- 当前运行时不调用在线 Edge TTS；预生成语音也不是比赛现场联网合成的结果。
- 未使用第三方录音素材。语音模型权重沿用其原始授权条款，本作品仅用于非商业教学与竞赛演示。

## 五、数据与隐私

- 麦克风由本机 Rust 进程通过 cpal 直接采集，音频仅在内存中用于离线识别，不写入磁盘、不上传互联网。
- 对话记录保存在浏览器 `sessionStorage`：刷新保留，关闭标签页后由浏览器清除；「重置全场」也会主动清除记录。
- 明确的天气、新闻或比分意图会尝试访问相应公开接口；失败时退回本地话术。
- 不含第三方统计或埋点脚本。

## 六、声明

本作品所用开源组件均遵循其各自许可证；模型、DLL 与素材的实际授权文件随对应来源或分发包保留。如发现任何权利瑕疵，团队将及时整改。

参赛团队　2026 年 11 月
