//! 运行时在线「小伊」语音合成（纯 Rust，可选增强，默认自动）。
//!
//! 背景：知识库的固定讲解词是赛前用 Edge TTS `zh-CN-XiaoyiNeural` 预生成的 46 条 wav
//! （`prebuilt-voices/`），按文本哈希命中磁盘缓存；而识图 / 自由问答由 LLM 现场生成，
//! 文本每次都不同、无法预先枚举，缓存必然 miss 后回退本地 Piper
//! （`zh_CN-huayan-medium`）——于是同一个讲解员出现两种声音。本模块补上这条缺口：
//! 动态回答现场在线合成小伊音色，结果立刻写进 tts-cache，同一句话第二次秒播。
//!
//! 为什么用 `msedge-tts` crate：微软 2026 年 10 月起强制 `Sec-MS-GEC` v2 签名
//! （自定义 CS64 + SHA256 变体，旧版签名握手直接 403），自己复刻既长又易随协议升级失效。
//! 该 crate 是**纯 Rust 同步阻塞客户端**、自带 v2 签名，与本项目手写同步 HTTP 服务一致，
//! 不需要 Python、tokio 或任何外部运行时，exe 仍是单个自包含文件。
//!
//! 输出格式：Edge 端点只给 `audio-24khz-48kbitrate-mono-mp3`（CBR），不是 wav。
//! WebView2(Chromium) 的 decodeAudioData 原生支持 mp3，因此不做解码、不引额外依赖；
//! `wav_duration()` 对 mp3 按固定码率估算时长，口型/字幕照常工作。
//!
//! 断网行为：短超时 TCP 探测 + 失败冷却，避免每问一句都卡住；回退顺序仍是
//! Edge(小伊) → Piper(华艳) → System.Speech → 静音。赛前把演示问题跑一遍即可把
//! 音频固化在 tts-cache，现场断网时演示过的句子仍是小伊音色。
//!
//! 环境变量：`QX_TTS_EDGE=off|auto|on`（默认 auto；off 强制全程离线）。

use msedge_tts::tts::{SpeechConfig, client::connect};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const VOICE: &str = "zh-CN-XiaoyiNeural";
const FORMAT: &str = "audio-24khz-48kbitrate-mono-mp3";
/// 与 tools/prebuild_voices.py 的 `+10%` 保持一致；crate 会格式化成 `+10%`。
const RATE: i32 = 10;
/// 单次在线合成最长等待（握手 + 生成）。超时后线程自行结束，不阻塞问答请求。
const SYNTH_TIMEOUT: Duration = Duration::from_secs(14);
/// 失败冷却：断网现场不会每问一句都重复等待。
const COOLDOWN_SECS: i64 = 120;
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Auto,
    On,
}

pub struct EdgeTts {
    mode: Mode,
    cooldown_until: AtomicI64,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// crate 的 `build_ssml_message` 直接把文本插进 XML，不做转义；讲解词里若出现
/// `&`、`<` 等会破坏 SSML。这里先转成实体，服务端解析后仍是原文本、发音不变。
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

pub(crate) fn is_mp3(b: &[u8]) -> bool {
    b.len() > 1024 && (b.starts_with(b"ID3") || (b[0] == 0xff && (b[1] & 0xe0) == 0xe0))
}

fn synth_once(text: &str) -> Option<Vec<u8>> {
    let cfg = SpeechConfig {
        voice_name: VOICE.to_string(),
        audio_format: FORMAT.to_string(),
        pitch: 0,
        rate: RATE,
        volume: 0,
    };
    let mut client = connect().ok()?;
    let audio = client.synthesize(&xml_escape(text), &cfg).ok()?;
    let bytes = audio.audio_bytes;
    if is_mp3(&bytes) {
        Some(bytes)
    } else {
        None
    }
}

impl EdgeTts {
    pub fn new() -> Self {
        let mode = match std::env::var("QX_TTS_EDGE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "0" | "false" => Mode::Off,
            "on" | "1" | "true" => Mode::On,
            _ => Mode::Auto,
        };
        Self { mode, cooldown_until: AtomicI64::new(0) }
    }

    pub fn enabled(&self) -> bool {
        self.mode != Mode::Off
    }

    /// 「在线小伊此刻不可用」：显式关闭，或连通性探测失败 / 合成失败后处于冷却期。
    /// 离线音色守卫据此决定是否允许回退到另一个发音人。
    pub fn is_offline(&self) -> bool {
        !self.enabled() || self.cooldown_until.load(Ordering::Relaxed) > now_secs()
    }

    /// /api/stats 用的可读状态。
    pub fn status(&self) -> &'static str {
        if self.mode == Mode::Off {
            "off"
        } else if self.cooldown_until.load(Ordering::Relaxed) > now_secs() {
            "cooling"
        } else {
            "ready"
        }
    }

    /// 快速判断能否连到微软语音端点，避免断网时白等一次 WebSocket 超时。
    fn online(&self) -> bool {
        let Ok(addrs) = ("speech.platform.bing.com", 443).to_socket_addrs() else {
            return false;
        };
        addrs
            .take(3)
            .any(|a| TcpStream::connect_timeout(&a, PROBE_TIMEOUT).is_ok())
    }

    fn cool_down(&self) {
        self.cooldown_until
            .store(now_secs() + COOLDOWN_SECS, Ordering::Relaxed);
    }

    /// 现场把 `text` 合成成小伊语音，返回 mp3 字节；失败返回 None 由调用方回退离线 Piper。
    pub fn synth(&self, text: &str) -> Option<Vec<u8>> {
        if !self.enabled() || self.cooldown_until.load(Ordering::Relaxed) > now_secs() {
            return None;
        }
        if text.trim().is_empty() {
            return None;
        }
        if self.mode == Mode::Auto && !self.online() {
            self.cool_down();
            return None;
        }
        // 超长回答截断，避免在线合成拖过前端超时；文字回答仍完整显示。
        let owned: String = if text.chars().count() > 500 {
            text.chars().take(500).collect()
        } else {
            text.to_string()
        };
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(synth_once(&owned));
        });
        match rx.recv_timeout(SYNTH_TIMEOUT) {
            Ok(Some(bytes)) => Some(bytes),
            _ => {
                self.cool_down();
                None
            }
        }
    }
}

impl Default for EdgeTts {
    fn default() -> Self {
        Self::new()
    }
}

/// 赛前烘焙专用：不受 `QX_TTS_EDGE` / 冷却状态影响，直接合成一句小伊语音。
/// 同样用独立线程 + 超时隔离，避免某一条卡住拖垮整批烘焙。
pub(crate) fn synth_bake(text: &str) -> Option<Vec<u8>> {
    if text.trim().is_empty() {
        return None;
    }
    let owned: String = if text.chars().count() > 500 {
        text.chars().take(500).collect()
    } else {
        text.to_string()
    };
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(synth_once(&owned));
    });
    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(Some(bytes)) => Some(bytes),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_escape_neutralizes_ssml_metacharacters() {
        assert_eq!(xml_escape("A&B <C> \"D\" 'E'"), "A&amp;B &lt;C&gt; &quot;D&quot; &apos;E&apos;");
    }

    #[test]
    fn mp3_magic_is_recognized() {
        assert!(!is_mp3(b"RIFFxxxxWAVE"));
        assert!(!is_mp3(&[0xff, 0xfb]));
        let mut mp3 = vec![0xff, 0xfb, 0x90, 0x00];
        mp3.extend(vec![0u8; 4096]);
        assert!(is_mp3(&mp3));
        let mut id3 = b"ID3".to_vec();
        id3.extend(vec![0u8; 4096]);
        assert!(is_mp3(&id3));
    }

    /// 需要联网的手动探针：验证 GEC v2 签名、TLS 握手与小伊语音真的可用。
    /// 平时 `cargo test` 忽略，用 `cargo test -p qx-desktop live_edge_tts -- --ignored --nocapture` 跑。
    #[test]
    #[ignore = "需联网：手动验证微软 Edge TTS 通路"]
    fn live_edge_tts() {
        let e = EdgeTts::new();
        assert!(e.enabled(), "QX_TTS_EDGE 不应为 off");
        let bytes = e.synth("您好，欢迎来到侨乡江门展厅。").expect("在线小伊合成应成功");
        assert!(is_mp3(&bytes), "应返回合法 mp3");
        println!("[edge-probe] 小伊实时合成成功：{} bytes", bytes.len());
    }
}
