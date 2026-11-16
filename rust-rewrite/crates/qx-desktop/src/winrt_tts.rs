//! 离线「小伊」：Windows 11 OneCore 自然语音（XiaoyiNeural）。
//!
//! Edge TTS 的「小伊」是 Azure 神经语音的在线版，权重不公开、离线复刻不了。
//! 但 Windows 11 内置了它的**离线嵌入式版**（Narrator / OneCore 自然语音，
//! 同为 Azure XiaoyiNeural 发音人），通过 WinRT `SpeechSynthesizer` 调用，
//! 全程零模型、零显存、零 Python、微软官方合规——这就是「断网也是小伊声音」
//! 的正解，比语音克隆（9 GB 模型 + 合规风险 + 音色近似）好得多。
//!
//! 前提：系统已装「中文（简体）神经语音」包（设置 → 时间和语言 → 语言和区域 →
//! 中文 → 语言选项 → 语音 → 下载）。本机没装时 `load()` 返回 None，回退到静音。
//!
//! 【重要 · COM 单元隔离】WinRT 需要先初始化 COM。若在**主线程**调用
//! `RoInitialize(RO_INIT_MULTITHREADED)`，会把主线程锁成 MTA；而 gpui 建窗要求
//! STA，随后 `application().run()` 就会以 0x80010106 (RPC_E_CHANGED_MODE) panic，
//! 表现为「双击没反应、窗口出不来」（Local 版曾因此从出生就跑不起来）。
//! 因此这里把 WinRT 的 COM 初始化与 SpeechSynthesizer 全部关进**专用后台线程**，
//! 主线程只通过 mpsc 收发文本/WAV，绝不接触 COM。WinRT 对象有单元亲和性，
//! 固定在同一后台线程创建和使用，也顺带避免了跨线程调用 WinRT 的隐患。

use std::future::{Future, IntoFuture};
use std::pin::pin;
use std::sync::mpsc::{channel, Sender};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use windows::core::HSTRING;
use windows::Media::SpeechSynthesis::{SpeechSynthesizer, VoiceInformation};
use windows::Storage::Streams::DataReader;
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};

/// 后台线程里初始化 WinRT（MTA）。只在专用线程调用，绝不污染主线程。
fn ensure_winrt() {
    unsafe {
        let _ = RoInitialize(RO_INIT_MULTITHREADED);
    }
}

/// 极简 block_on：no-op waker + 短轮询。windows-future 的 AsyncFuture 每次
/// poll 都会先查 `Status`，完成即返回 Ready，所以即使 waker 不唤醒，靠轮询也能
/// 拿到结果（TTS 合成几百 ms，2 ms 轮询代价可忽略）。不用 park/unpark + 栈 token，
/// 避免 Completed 回调持有悬垂指针导致 use-after-free。
fn block_on<T>(fut: impl Future<Output = T>) -> T {
    fn raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            raw_waker()
        }
        fn wake(_: *const ()) {}
        fn wake_by_ref(_: *const ()) {}
        fn drop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(2)),
        }
    }
}

/// 主线程持有的轻量句柄。真正的 SpeechSynthesizer 常驻后台专用线程。
pub struct WinrtTts {
    req: Sender<(String, Sender<Option<Vec<u8>>>)>,
    ready: bool,
}

impl WinrtTts {
    /// 在专用后台线程初始化并选中 XiaoyiNeural。主线程零 COM 接触。
    /// 找不到该语音包时返回 None（离线小伊不可用）。
    pub fn load() -> Option<Self> {
        // 与后台线程握手：拿到「初始化是否成功」的结果，避免主线程阻塞过久。
        let (ready_tx, ready_rx) = channel::<bool>();
        let (req_tx, req_rx) = channel::<(String, Sender<Option<Vec<u8>>>)>();

        std::thread::Builder::new()
            .name("winrt-tts".into())
            .spawn(move || {
                // ① 只在后台线程初始化 COM（MTA）
                ensure_winrt();
                let synth = match SpeechSynthesizer::new() {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "SpeechSynthesizer 创建失败，离线小伊不可用");
                        let _ = ready_tx.send(false);
                        return;
                    }
                };
                let voices = match SpeechSynthesizer::AllVoices() {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "获取系统语音列表失败，离线小伊不可用");
                        let _ = ready_tx.send(false);
                        return;
                    }
                };
                let name_of = |v: &VoiceInformation| -> String {
                    v.DisplayName().map(|n| n.to_string()).unwrap_or_default()
                };
                // IVectorView 用 GetAt + Size 遍历（0.62 没有 .iter()）
                let mut target: Option<VoiceInformation> = None;
                if let Ok(n) = voices.Size() {
                    for i in 0..n {
                        if let Ok(v) = voices.GetAt(i) {
                            if name_of(&v).to_lowercase().contains("xiaoyi") {
                                target = Some(v);
                                break;
                            }
                        }
                    }
                }
                let ready = match target {
                    Some(v) => {
                        if synth.SetVoice(&v).is_err() {
                            tracing::warn!("选中 Xiaoyi 语音失败，离线小伊不可用");
                            false
                        } else {
                            tracing::info!(voice = %name_of(&v), "离线小伊语音已就绪");
                            true
                        }
                    }
                    None => {
                        // 诊断开关：本机没装 Xiaoyi 时，用默认音色跑通合成链路（音色非小伊）。
                        let fallback = std::env::var("QX_WINRT_FALLBACK")
                            .is_ok_and(|v| !v.is_empty() && v != "0");
                        if fallback {
                            tracing::warn!("未找到 Xiaoyi，QX_WINRT_FALLBACK 用默认音色（仅诊断）");
                            true
                        } else {
                            tracing::warn!(
                                "未找到 Xiaoyi 自然语音。请在 设置→时间和语言→语言和区域→\
                                 中文(简体)→语言选项→语音 里下载「神经语音」，之后断网也能用小伊音色。"
                            );
                            false
                        }
                    }
                };
                let _ = ready_tx.send(ready);
                if !ready {
                    return;
                }
                // ② 消息循环：WinRT 对象固定在本线程使用
                for (text, reply) in req_rx {
                    let _ = reply.send(synth_wav(&synth, &text));
                }
            })
            .ok()?;

        // 等待后台线程完成初始化（最多给 3 秒，避免语音包异常时卡住启动）。
        let ready = match ready_rx.recv_timeout(std::time::Duration::from_secs(3)) {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!("离线小伊初始化超时（3 s），本次降级");
                false
            }
        };
        Some(Self { req: req_tx, ready })
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    /// 合成中文文本为 WAV。语速 +10% 对齐在线小伊。
    /// 通过 channel 交给后台专用线程执行；本方法不接触任何 COM/WinRT 对象。
    pub fn synth(&self, text: &str) -> Option<Vec<u8>> {
        if !self.ready {
            return None;
        }
        let (reply_tx, reply_rx) = channel::<Option<Vec<u8>>>();
        self.req.send((text.to_string(), reply_tx)).ok()?;
        // 合成通常几百 ms；给 15 秒上限，异常时不至于永久卡住 HTTP 线程。
        reply_rx.recv_timeout(std::time::Duration::from_secs(15)).ok()?
    }
}

/// 在后台专用线程上执行真正的 WinRT 合成。
fn synth_wav(synth: &SpeechSynthesizer, text: &str) -> Option<Vec<u8>> {
    if let Ok(opts) = synth.Options() {
        // 1.1 = +10%，与 Edge 小伊的 rate 设置一致
        let _ = opts.SetSpeakingRate(1.1);
    }
    let op = synth.SynthesizeTextToStreamAsync(&HSTRING::from(text)).ok()?;
    let stream = block_on(op.into_future()).ok()?;
    let size = stream.Size().ok()? as u32;
    if size == 0 {
        return None;
    }
    // SpeechSynthesisStream 是 IRandomAccessStream，转 IInputStream 后交给 DataReader
    let input = stream.GetInputStreamAt(0).ok()?;
    let reader = DataReader::CreateDataReader(&input).ok()?;
    block_on(reader.LoadAsync(size).ok()?.into_future()).ok()?;
    let mut buf = vec![0u8; size as usize];
    reader.ReadBytes(&mut buf).ok()?; // 读满 buf；ReadBytes 返回 Result<()>
    if buf.len() < 44 {
        return None;
    }
    Some(buf)
}
