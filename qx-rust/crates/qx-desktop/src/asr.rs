//! 离线中文语音识别 —— ①「语音识别」的 Rust 化实现。
//!
//! 走 [whisper-rs](https://codeberg.org/tazz4843/whisper-rs)（whisper.cpp 的
//! Rust 绑定）：构建期用 cc 直接编译 whisper.cpp，静态链进可执行文件。
//! 因此：
//! - 运行时零 Python、零外部进程、零预编译 DLL（不再有 whisper.dll/ggml*.dll），
//!   Windows（MSVC）与 Linux 通吃，仍是单文件自包含；
//! - 纯 CPU 推理，不占显存（Ollama 常驻已吃掉约 3 GB，再加模型会爆 8 GB 卡）；
//! - 模型常驻内存一次加载，base 模型加载 0.3–0.6 s，短句转写约 0.7 s。
//!
//! 前端 `web/src/audio.js` 送来的二进制帧就是 **16 kHz 单声道 float32 小端**
//! （浏览器侧已按 sampleRate/16000 降采样），无需再重采样；文本帧 `END`
//! 表示录音结束。这里只负责「PCM 数组 -> 中文句子」，WebSocket 帧循环在 http.rs。
//!
//! 找不到模型 / 加载失败时静默降级为「不可用」，不影响键盘问答与其余功能。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// 简体中文 + 展厅专有名词提示。实测关键：不加这段，base 模型会输出繁体、
/// 并把「新会陈皮」听成「新会成皮」；加上后与原文一字不差。
const INITIAL_PROMPT: &str =
    "以下是普通话的句子，讲的是广东江门侨乡、开平碉楼、赤坎骑楼、新会陈皮、侨批银信和梁启超。";

/// whisper 输入固定 16 kHz；内部窗口 30 s，前端 28 s 自动收尾（DEF-19）。
const SAMPLE_RATE: usize = 16_000;
const MAX_SAMPLES: usize = SAMPLE_RATE * 30;
const MIN_SAMPLES: usize = SAMPLE_RATE / 4;
/// 全程 RMS 低于此值视为静音，直接返回空，避免 whisper 对纯静音产生幻听。
const SILENCE_RMS: f32 = 0.003;

static DEBUG_LOG: AtomicBool = AtomicBool::new(false);

// ─────────────────────── 推理引擎（whisper-rs，静态链接） ───────────────────────

/// 模型上下文常驻；whisper-rs 每次 `create_state()` 都会重置内部状态，
/// 这里为省开销复用一个常驻 state（Mutex 串行化，与旧版 FFI 方案一致）。
struct Engine {
    ctx: WhisperContext,
    threads: i32,
}

impl Engine {
    fn load(model: &Path) -> Result<Self, String> {
        let ctx = WhisperContext::new_with_params(
            model.to_string_lossy().as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("加载模型失败：{}：{e}", model.display()))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8) as i32)
            .unwrap_or(4);
        Ok(Self { ctx, threads })
    }

    fn transcribe(&self, samples: &[f32]) -> Option<String> {
        let mut state = self.ctx.create_state().ok()?;
        // Greedy best_of=1：贪心一次解码，速度优先，与旧版 DLL 参数一致。
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(self.threads);
        p.set_language(Some("zh"));
        p.set_detect_language(false);
        p.set_translate(false);
        p.set_no_timestamps(true);
        p.set_single_segment(true);
        p.set_no_context(true);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        p.set_initial_prompt(INITIAL_PROMPT);
        p.set_temperature(0.0);

        state.full(p, samples).ok()?;
        let n = state.full_n_segments().max(0);
        let mut out = String::new();
        for i in 0..n {
            if let Some(seg) = state.get_segment(i) {
                out.push_str(&seg.to_str_lossy().unwrap_or_default());
            }
        }
        Some(clean_text(&out))
    }
}

// ─────────────────────── 对外接口 ───────────────────────

pub struct Asr {
    engine: Mutex<Option<Engine>>,
    load_ms: AtomicU64,
    ready: AtomicBool,
    error: Mutex<String>,
}

impl Asr {
    /// 同步加载（base 模型约 0.3–0.6 s）。失败时只降级，不阻断启动。
    pub fn load(root: &Path) -> Arc<Self> {
        DEBUG_LOG.store(
            std::env::var("QX_ASR_DEBUG").is_ok_and(|v| v != "0" && !v.is_empty()),
            Ordering::Relaxed,
        );
        let me = Arc::new(Self {
            engine: Mutex::new(None),
            load_ms: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            error: Mutex::new(String::new()),
        });
        // 现场排障开关：跳过 whisper 加载，退回纯键盘问答。
        if std::env::var("QX_ASR_DISABLE").is_ok_and(|v| v != "0" && !v.is_empty()) {
            tracing::warn!("离线语音识别已由 QX_ASR_DISABLE 关闭");
            *me.error.lock().unwrap() = "已由 QX_ASR_DISABLE 关闭".into();
            return me;
        }
        let t0 = Instant::now();
        let dir = root.join("asr");
        let model = pick_model(&dir);
        let result = match model {
            Some(m) => Engine::load(&m),
            None => Err(format!("{} 下没有 Whisper 模型", dir.display())),
        };
        match result {
            Ok(e) => {
                let ms = t0.elapsed().as_millis() as u64;
                *me.engine.lock().unwrap() = Some(e);
                me.load_ms.store(ms, Ordering::Relaxed);
                me.ready.store(true, Ordering::Relaxed);
                tracing::info!(elapsed_ms = ms, "离线语音识别就绪");
            }
            Err(e) => {
                tracing::warn!(error = %e, "离线语音识别不可用");
                *me.error.lock().unwrap() = e;
            }
        }
        me
    }

    /// 后台线程加载（http.rs 启动路径用）：窗口 / HTTP 服务不等 ASR 就绪，
    /// 就绪前 available() 如实返回 false。
    pub fn spawn_load(root: PathBuf) -> Arc<Self> {
        DEBUG_LOG.store(
            std::env::var("QX_ASR_DEBUG").is_ok_and(|v| v != "0" && !v.is_empty()),
            Ordering::Relaxed,
        );
        let me = Arc::new(Self {
            engine: Mutex::new(None),
            load_ms: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            error: Mutex::new(String::new()),
        });
        if std::env::var("QX_ASR_DISABLE").is_ok_and(|v| v != "0" && !v.is_empty()) {
            tracing::warn!("离线语音识别已由 QX_ASR_DISABLE 关闭");
            *me.error.lock().unwrap() = "已由 QX_ASR_DISABLE 关闭".into();
            return me;
        }
        std::thread::Builder::new()
            .name("asr-load".into())
            .spawn({
                let me = Arc::clone(&me);
                move || {
                    let t0 = Instant::now();
                    let dir = root.join("asr");
                    let result = match pick_model(&dir) {
                        Some(m) => Engine::load(&m),
                        None => Err(format!("{} 下没有 Whisper 模型", dir.display())),
                    };
                    match result {
                        Ok(e) => {
                            let ms = t0.elapsed().as_millis() as u64;
                            *me.engine.lock().unwrap() = Some(e);
                            me.load_ms.store(ms, Ordering::Relaxed);
                            me.ready.store(true, Ordering::Relaxed);
                            tracing::info!(elapsed_ms = ms, "离线语音识别就绪（后台加载）");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "离线语音识别不可用");
                            *me.error.lock().unwrap() = e;
                        }
                    }
                }
            })
            .ok();
        me
    }

    pub fn available(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    pub fn load_ms(&self) -> u64 {
        self.load_ms.load(Ordering::Relaxed)
    }

    pub fn error(&self) -> String {
        self.error.lock().unwrap().clone()
    }

    /// PCM f32（16 kHz 单声道）-> 中文句子。静音 / 过短 / 引擎不可用时返回 None。
    pub fn transcribe(&self, pcm: &[f32]) -> Option<String> {
        if !self.available() {
            return None;
        }
        let samples = prepare(pcm)?;
        let guard = self.engine.lock().ok()?;
        let engine = guard.as_ref()?;
        engine.transcribe(&samples)
    }
}

/// 模型优先级：显式 `QX_ASR_MODEL` > 内嵌 base（快，0.7 s 级）> 外置 small（更准但约 2 s）。
/// base 随 exe 分发，保证赛场只拷一个文件也能用；想换模型放个文件 + 设环境变量即可。
fn pick_model(dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QX_ASR_MODEL") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        tracing::warn!(path = %p.display(), "QX_ASR_MODEL 指向的文件不存在");
    }
    ["ggml-base-q5_1.bin", "ggml-small-q5_1.bin", "ggml-base.bin", "ggml-small.bin"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// 清洗输入：去非有限值、截断 30 s、判静音。单独抽出来便于单测，不依赖引擎。
fn prepare(pcm: &[f32]) -> Option<Vec<f32>> {
    if pcm.len() < MIN_SAMPLES {
        return None;
    }
    let n = pcm.len().min(MAX_SAMPLES);
    let mut out = Vec::with_capacity(n);
    let mut sum = 0f64;
    for &v in &pcm[..n] {
        let v = if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        sum += (v as f64) * (v as f64);
        out.push(v);
    }
    let rms = (sum / n as f64).sqrt() as f32;
    if rms < SILENCE_RMS {
        return None;
    }
    Some(out)
}

/// 去掉 whisper 在静音/音乐段常见的括号标记与首尾空白。
/// 注意 `(音乐)` 这类汉字会被 `is_alphanumeric` 判成「有内容」，
/// 所以必须先按括号标记削掉，不能只靠最后的标点检查。
fn clean_text(s: &str) -> String {
    let mut t = s.trim().to_string();
    for noise in ["[BLANK_AUDIO]", "(BLANK_AUDIO)", "[MUSIC]", "字幕由", "请不吝点赞", "订阅", "转发"] {
        t = t.replace(noise, "");
    }
    let t = strip_bracket_markers(&t);
    let t = t.trim().trim_start_matches(&['，', '。', '、', ' '][..]).trim().to_string();
    if t.chars().all(|c| !c.is_alphanumeric()) {
        String::new()
    } else {
        t
    }
}

/// whisper 用 `(音乐)` `[MUSIC]` `(掌声)` 之类标注非语音事件。
/// 短且不含句读的括号片段一律丢弃；带句读的长括号内容按正常文本保留。
fn strip_bracket_markers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let punctuation = ['，', '。', '！', '？', ',', '.', '!', '?'];
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let open = chars[i];
        let close = match open {
            '(' => Some(')'),
            '[' => Some(']'),
            '（' => Some('）'),
            '【' => Some('】'),
            _ => None,
        };
        if let Some(close) = close {
            if let Some(rel) = chars[i + 1..].iter().position(|&c| c == close) {
                let end = i + 1 + rel;
                let inner: String = chars[i + 1..end].iter().collect();
                let short = inner.chars().count() <= 12;
                let has_sentence = inner.chars().any(|c| punctuation.contains(&c));
                if short && !has_sentence {
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(open);
        i += 1;
    }
    out
}

/// 把 WebSocket 二进制帧（float32 小端）追加到 PCM 缓冲。
/// 长度不是 4 的倍数时丢弃尾部残字节，避免后续帧整体错位。
pub fn append_pcm(buf: &mut Vec<f32>, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(4) {
        buf.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    // 前端 28 s 上限，这里再兜一道，防止异常连接把内存吃满。
    if buf.len() > MAX_SAMPLES {
        buf.truncate(MAX_SAMPLES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_pcm_decodes_little_endian() {
        let mut buf = Vec::new();
        append_pcm(&mut buf, &0.5f32.to_le_bytes());
        append_pcm(&mut buf, &(-0.25f32).to_le_bytes());
        assert_eq!(buf, vec![0.5, -0.25]);
    }

    #[test]
    fn append_pcm_drops_incomplete_tail() {
        let mut buf = Vec::new();
        append_pcm(&mut buf, &[0, 0, 0]); // 不足 4 字节
        assert!(buf.is_empty());
        append_pcm(&mut buf, &1.0f32.to_le_bytes());
        assert_eq!(buf, vec![1.0]);
    }

    #[test]
    fn append_pcm_caps_at_30_seconds() {
        let mut buf = vec![0.0; MAX_SAMPLES - 1];
        append_pcm(&mut buf, &1.0f32.to_le_bytes());
        append_pcm(&mut buf, &1.0f32.to_le_bytes());
        assert_eq!(buf.len(), MAX_SAMPLES);
    }

    #[test]
    fn prepare_rejects_short_silence_and_nan() {
        assert!(prepare(&[]).is_none());
        assert!(prepare(&vec![0.0; MIN_SAMPLES - 1]).is_none());
        assert!(prepare(&vec![0.0; MIN_SAMPLES + 100]).is_none(), "纯静音应拒绝");
        let loud = vec![0.2; MIN_SAMPLES + 100];
        assert_eq!(prepare(&loud).unwrap().len(), loud.len());
        let mut with_nan = loud.clone();
        with_nan[0] = f32::NAN;
        with_nan[1] = f32::INFINITY;
        let out = prepare(&with_nan).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn prepare_caps_long_audio() {
        let loud = vec![0.2; MAX_SAMPLES + 1000];
        assert_eq!(prepare(&loud).unwrap().len(), MAX_SAMPLES);
    }

    #[test]
    fn clean_text_strips_noise() {
        assert_eq!(clean_text("  您好，欢迎来到侨乡江门。  "), "您好，欢迎来到侨乡江门。");
        assert_eq!(clean_text("[BLANK_AUDIO]"), "");
        assert_eq!(clean_text("(音乐)"), "");
        assert_eq!(clean_text("(掌声) 您好"), "您好");
        assert_eq!(clean_text("，陈皮。"), "陈皮。");
        assert_eq!(clean_text("（这是一句带，逗号的完整括号内容）"), "（这是一句带，逗号的完整括号内容）");
    }

    /// 真机冒烟：需要 asr/ 下有 ggml 模型。
    /// `cargo test -p qx-desktop asr::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 asr/ 下的 ggml 模型"]
    fn live_transcribe_chinese() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
        let asr = Asr::load(&root);
        assert!(asr.available(), "引擎未就绪：{}", asr.error());
        // prebuilt-voices 里这段是 22050 Hz PCM16，此处手工降到 16 kHz 喂进去。
        let wav = root.join("prebuilt-voices/74cae77537c34f62.wav");
        let pcm = read_wav_16k(&wav);
        let text = asr.transcribe(&pcm).expect("转写失败");
        println!("live asr -> {text}");
        assert!(text.contains("陈皮"), "转写结果不含「陈皮」：{text}");
        assert!(text.contains("慢走"), "转写结果不含「慢走」：{text}");
    }

    /// 只给测试用：读 PCM16 单声道 WAV 并线性插值到 16 kHz。
    /// 生产链路拿到的本来就是 16 kHz float32，不需要这一步。
    fn read_wav_16k(path: &Path) -> Vec<f32> {
        let b = std::fs::read(path).expect("wav 不存在");
        assert_eq!(&b[0..4], b"RIFF");
        let rate = u32::from_le_bytes(b[24..28].try_into().unwrap()) as usize;
        let bits = u16::from_le_bytes(b[34..36].try_into().unwrap());
        assert_eq!(bits, 16);
        // data chunk 不一定紧跟 fmt，简单扫一遍
        let mut pos = 12;
        let mut data = &b[0..0];
        while pos + 8 <= b.len() {
            let sz = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if &b[pos..pos + 4] == b"data" {
                data = &b[pos + 8..(pos + 8 + sz).min(b.len())];
                break;
            }
            pos += 8 + sz + (sz % 2);
        }
        let src: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();
        if rate == SAMPLE_RATE {
            return src;
        }
        let n = src.len() * SAMPLE_RATE / rate;
        (0..n)
            .map(|i| {
                let x = i as f64 * rate as f64 / SAMPLE_RATE as f64;
                let j = x.floor() as usize;
                let f = (x - j as f64) as f32;
                src[j] * (1.0 - f) + src.get(j + 1).copied().unwrap_or(0.0) * f
            })
            .collect()
    }
}
