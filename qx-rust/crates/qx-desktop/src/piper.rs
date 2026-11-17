//! 离线神经语音合成：Piper VITS（zh_CN-huayan 华艳女声）。
//!
//! 这是「真实女孩子声音」的关键：离在线「小伊」都不可用时的本地后备。
//! 技术栈与原 Python 版同源（Piper VITS 模型），但驱动换成
//! [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs)（sherpa-onnx 的
//! Rust 绑定），彻底跨平台：
//!
//! - 推理由 sherpa-onnx 内置的 onnxruntime 完成，编译期静态链入；
//!   不再动态加载 Windows 专用的 onnxruntime.dll；
//! - 音素化（G2P）由 sherpa-onnx 内置的 **C++ espeak-ng** 完成
//!   （`tts/espeak-ng-data/`），取代依赖 CPython 运行时的 `espeakbridge.pyd`
//!   —— 旧方案在 Rust exe 里本来就无法加载（CPython 扩展），离线合成一直
//!   处于不可用状态；现在真正可用了，Windows / Linux 行为一致。
//!
//! 资产（`tts/`）：`zh_CN-huayan-medium.onnx`（22050 Hz）+ `tokens.txt` +
//! `espeak-ng-data/`，随 exe 打包释放。全部离线，无需联网；
//! 单句 30 字合成约 0.2~0.9 s（RTF≈0.14），语速≈5.5 字/秒。

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use sherpa_rs::tts::{VitsTts, VitsTtsConfig};

struct Inner {
    engine: VitsTts,
    sample_rate: u32,
}

pub struct PiperTts {
    dir: PathBuf,
    inner: Mutex<Option<Inner>>,
    load_ms: AtomicU64,
}

impl PiperTts {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, inner: Mutex::new(None), load_ms: AtomicU64::new(0) }
    }

    /// 自检面板用：引擎就绪耗时（毫秒），0 = 未就绪（将回退静音）。
    pub fn load_ms(&self) -> u64 {
        self.load_ms.load(Ordering::Relaxed)
    }

    fn has_assets(&self) -> bool {
        self.dir.join("zh_CN-huayan-medium.onnx").is_file()
            && self.dir.join("tokens.txt").is_file()
            && self.dir.join("espeak-ng-data").is_dir()
    }

    fn init_inner(&self) -> Option<Inner> {
        let dir = &self.dir;
        if !self.has_assets() {
            tracing::warn!(dir = %dir.display(), "缺少 VITS TTS 资产，离线神经语音不可用");
            return None;
        }
        // 采样率仍从 onnx.json 读取；缺失时按 Piper huayan-medium 默认 22050。
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("zh_CN-huayan-medium.onnx.json")).ok()?).ok()?;
        let inf = &json["inference"];
        // 允许用环境变量现场调音（比赛现场微调语速/稳定度，无需重编译）。
        let env_f32 = |k: &str, def: f32| -> f32 {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
        };
        let engine = VitsTts::new(VitsTtsConfig {
            model: dir.join("zh_CN-huayan-medium.onnx").to_string_lossy().into_owned(),
            tokens: dir.join("tokens.txt").to_string_lossy().into_owned(),
            data_dir: dir.join("espeak-ng-data").to_string_lossy().into_owned(),
            noise_scale: env_f32(
                "QX_TTS_NOISE_SCALE",
                inf["noise_scale"].as_f64().unwrap_or(0.667) as f32,
            ),
            // 默认 0.95：huayan 在 1.0 时整体偏慢（长答案约 3.6 字/秒），
            // 略微压缩节奏后接近真人讲解的 4.2 字/秒，现场可再用环境变量微调。
            length_scale: env_f32(
                "QX_TTS_LENGTH_SCALE",
                (inf["length_scale"].as_f64().unwrap_or(1.0) as f32) * 0.95,
            ),
            noise_scale_w: env_f32("QX_TTS_NOISE_W", inf["noise_w"].as_f64().unwrap_or(0.8) as f32),
            ..Default::default()
        });
        Some(Inner {
            engine,
            sample_rate: json["audio"]["sample_rate"].as_u64().unwrap_or(22050) as u32,
        })
    }

    /// 预热：应用启动线程调用，把模型加载藏在启动阶段。
    /// 空跑一句短文本验证推理链路，并顺便记录真实预热耗时。
    pub fn preload(&self) {
        let t0 = Instant::now();
        let mut guard = self.inner.lock().unwrap();
        if guard.is_none() {
            *guard = self.init_inner();
        }
        if let Some(inner) = guard.as_mut() {
            if inner.synth_once("好").is_some() {
                let ms = t0.elapsed().as_millis() as u64;
                self.load_ms.store(ms, Ordering::Relaxed);
                tracing::info!(elapsed_ms = ms, "离线神经语音（sherpa-onnx VITS）就绪");
            } else {
                // 引擎创建成功但首句失败：视为不可用，避免 synthesize 走死路。
                tracing::warn!("VITS 首句合成失败，离线神经语音不可用");
                *guard = None;
            }
        }
    }

    /// 合成中文文本为 16-bit PCM 单声道 wav（22.05 kHz）。失败返回 None（调用方回退）。
    ///
    /// 长句一次性喂给 VITS 容易出现韵母拖沓、句间粘连甚至杂音碎片，所以先做
    /// 文本规整（缩写/符号）再按标点分句，逐句合成后用 100 ms 静音拼接；
    /// 这同时让句间有自然停顿，听感更接近真人播报。
    pub fn synth(&self, text: &str) -> Option<Vec<u8>> {
        let pieces = split_for_tts(text);
        if pieces.is_empty() {
            return None;
        }
        let t0 = Instant::now();
        let mut guard = self.inner.lock().unwrap();
        if guard.is_none() {
            *guard = self.init_inner();
        }
        let inner = guard.as_mut()?;

        let rate = inner.sample_rate as f32;
        let gap = vec![0.0f32; (0.10 * rate) as usize];
        let mut pcm: Vec<f32> = Vec::new();
        for piece in pieces.iter() {
            let part = inner.synth_once(piece)?;
            let part = trim_silence(&part, rate);
            if part.is_empty() {
                continue;
            }
            if !pcm.is_empty() {
                pcm.extend_from_slice(&gap);
            }
            pcm.extend_from_slice(&part);
        }
        if pcm.is_empty() {
            return None;
        }
        if self.load_ms.load(Ordering::Relaxed) == 0 {
            self.load_ms.store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
        Some(wav_bytes(&pcm, rate as u32))
    }
}

impl Inner {
    /// 单句合成：返回 f32 单声道采样。sid=0 取包内唯一发音人。
    fn synth_once(&mut self, text: &str) -> Option<Vec<f32>> {
        let audio = self.engine.create(text, 0, 1.0).ok()?;
        (!audio.samples.is_empty()).then_some(audio.samples)
    }
}

/// TTS 专用文本规整：拉丁缩写念中文、去掉朗读无意义的符号。
fn normalize_for_tts(text: &str) -> String {
    let mut s = text.to_string();
    for (from, to) in [
        ("AI 智能讲解员", "人工智能讲解员"),
        ("AI智能讲解员", "人工智能讲解员"),
        ("AI", "人工智能"),
        ("VRM", "虚拟形象"),
        ("3D", "三维"),
        ("cm", "厘米"),
        ("km", "公里"),
        ("%", "百分之"),
    ] {
        s = s.replace(from, to);
    }
    // 「」《》等只做视觉强调，朗读时去掉即可；破折号与省略号转成停顿
    for ch in ['「', '」', '『', '』', '《', '》', '"', '"', '\'', '\'', '*', '#', '`'] {
        s = s.replace(ch, "");
    }
    s = s.replace("——", "，").replace("……", "，");
    s
}

/// 分句策略（实测结论）：VITS 每段都会带一段模型自身的前后静音，分得越碎整体
/// 节奏越慢（同样文本 137 字：整段 33 s，切 4 段 46 s）。因此**默认整段合成**，
/// 只有超过 120 字的长答案才按句号分组切开，兼顾流畅与长文本稳定性。
fn split_for_tts(text: &str) -> Vec<String> {
    let normalized = normalize_for_tts(text);
    let total = normalized.chars().count();
    if total <= 120 {
        let t = normalized.trim();
        return if t.is_empty() { Vec::new() } else { vec![t.to_string()] };
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    for raw in normalized.split_inclusive(['。', '！', '？', '；', '\n', '!', '?', ';']) {
        if buf.chars().count() + raw.chars().count() > 120 && !buf.is_empty() {
            out.push(buf.clone());
            buf.clear();
        }
        buf.push_str(raw);
    }
    if !buf.trim().is_empty() {
        out.push(buf);
    }
    out
}

/// 去掉一段语音首尾的静音（各留 40 ms 余量），减少拼接处的顿挫感。
fn trim_silence(samples: &[f32], rate: f32) -> Vec<f32> {
    const TH: f32 = 0.03;
    let margin = (0.02 * rate) as usize;
    let start = samples.iter().position(|s| s.abs() > TH).unwrap_or(0).saturating_sub(margin);
    let end = samples
        .iter()
        .rposition(|s| s.abs() > TH)
        .map(|i| (i + margin).min(samples.len()))
        .unwrap_or(0);
    if end <= start {
        return Vec::new();
    }
    samples[start..end].to_vec()
}

fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    // 峰值归一到 0.92：VITS 原始输出电平偏低，直接播放显得
    // 「不清晰、闷」，归一化后再写入，听感更饱满通透。
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let gain = if peak > 0.001 { (0.92 / peak).min(4.0) } else { 1.0 };
    let data: Vec<u8> = samples
        .iter()
        .flat_map(|s| ((((s * gain).clamp(-1.0, 1.0)) * 32767.0) as i16).to_le_bytes())
        .collect();
    let mut w = Vec::with_capacity(44 + data.len());
    w.extend(b"RIFF");
    w.extend(((36 + data.len()) as u32).to_le_bytes());
    w.extend(b"WAVEfmt ");
    w.extend(16u32.to_le_bytes());
    w.extend(1u16.to_le_bytes()); // PCM
    w.extend(1u16.to_le_bytes()); // mono
    w.extend(rate.to_le_bytes());
    w.extend((rate * 2).to_le_bytes());
    w.extend(2u16.to_le_bytes());
    w.extend(16u16.to_le_bytes());
    w.extend(b"data");
    w.extend((data.len() as u32).to_le_bytes());
    w.extend(data);
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 单元冒烟：不依赖模型，只验证文本规整 / 分句 / wav 头。
    #[test]
    fn helpers_work() {
        assert_eq!(normalize_for_tts("我是 AI 智能讲解员"), "我是 人工智能讲解员");
        assert_eq!(split_for_tts("").len(), 0);
        assert_eq!(split_for_tts("短句。").len(), 1);
        let long = "很。".repeat(80);
        assert!(split_for_tts(&long).len() > 1, "超长文本应分行");
        let wav = wav_bytes(&[0.5, -0.5], 22050);
        assert_eq!(&wav[0..4], b"RIFF");
    }

    /// 真机冒烟：需要 tts/ 下有 onnx / tokens.txt / espeak-ng-data。
    /// `cargo test -p qx-desktop piper::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 tts/ 下的 VITS 模型资产"]
    fn dump_tts_probe() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tts");
        let tts = PiperTts::new(dir);
        tts.preload();
        assert!(tts.load_ms() > 0, "VITS 引擎应就绪");
        let probe = concat!(
            "开平碉楼、赤坎骑楼街、新会陈皮、梁启超故居、小鸟天堂、侨批银信。",
            "我是 AI 智能讲解员小侨，慢慢走，侨乡的故事都在砖缝、信纸和柑皮里。",
            "开平的碉楼是华侨把在海外挣下的钱寄回家一层一层盖起来的，楼身是中式的，",
            "可柱式、栏杆和顶上的尖顶又带着意大利和拜占庭的样子，所以被叫做开平第一楼。",
            "小时候在潭江边长大，天天看这些楼，却一直不知道它们背后的故事。"
        );
        let wav = tts.synth(probe).expect("合成失败");
        let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.diag/probe.wav");
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        std::fs::write(&out, &wav).unwrap();
        println!("probe -> {}", out.display());
    }
}
