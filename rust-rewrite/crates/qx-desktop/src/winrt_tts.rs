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
//! 注意：`SpeechSynthesizer` 只能在有消息循环的线程上工作；这里被 HTTP 请求线程
//! 调用，需确保 MTA 初始化（Rust 进程默认 STA? 见 load() 里的说明）。

use std::future::{Future, IntoFuture};
use std::pin::pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use windows::core::HSTRING;
use windows::Media::SpeechSynthesis::{SpeechSynthesizer, VoiceInformation};
use windows::Storage::Streams::DataReader;
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};

/// WinRT 类需要先初始化运行时（MTA）。重复初始化返回 RPC_E_CHANGED_MODE 可忽略。
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

pub struct WinrtTts {
    synth: SpeechSynthesizer,
    ready: bool,
}

impl WinrtTts {
    /// 初始化并选中 XiaoyiNeural。找不到该语音包时返回 None（离线小伊不可用）。
    pub fn load() -> Option<Self> {
        ensure_winrt();
        let synth = SpeechSynthesizer::new().ok()?;
        let voices = SpeechSynthesizer::AllVoices().ok()?;
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
        match target {
            Some(v) => {
                if synth.SetVoice(&v).is_err() {
                    eprintln!("[winrt-tts] 选中 Xiaoyi 语音失败");
                    return None;
                }
                eprintln!("[winrt-tts] 离线小伊已就绪：{}", name_of(&v));
                Some(Self { synth, ready: true })
            }
            None => {
                // 诊断开关：本机没装 Xiaoyi 时，用默认音色跑通合成链路（音色非小伊）。
                let fallback = std::env::var("QX_WINRT_FALLBACK")
                    .is_ok_and(|v| !v.is_empty() && v != "0");
                if fallback {
                    eprintln!("[winrt-tts] 未找到 Xiaoyi，QX_WINRT_FALLBACK 用默认音色（仅诊断）");
                    Some(Self { synth, ready: true })
                } else {
                    eprintln!(
                        "[winrt-tts] 未找到 Xiaoyi 自然语音。请在 设置→时间和语言→语言和区域→\
                         中文(简体)→语言选项→语音 里下载「神经语音」，之后断网也能用小伊音色。"
                    );
                    None
                }
            }
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    /// 合成中文文本为 WAV。语速 +10% 对齐在线小伊。
    pub fn synth(&self, text: &str) -> Option<Vec<u8>> {
        if !self.ready {
            return None;
        }
        if let Ok(opts) = self.synth.Options() {
            // 1.1 = +10%，与 Edge 小伊的 rate 设置一致
            let _ = opts.SetSpeakingRate(1.1);
        }
        let op = self
            .synth
            .SynthesizeTextToStreamAsync(&HSTRING::from(text))
            .ok()?;
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
}
