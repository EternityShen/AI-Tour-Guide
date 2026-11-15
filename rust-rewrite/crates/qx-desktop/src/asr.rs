//! 离线中文语音识别 —— ①「语音识别」的 Rust 化实现。
//!
//! 走 whisper.cpp (b5130) 的 C API：官方预编译 `whisper.dll` / `ggml*.dll`
//! 由 build.rs 打进 bundle.zip、启动时释放，`LoadLibrary` 动态加载后直接调 FFI。
//! 因此：
//! - 运行时零 Python、零外部进程、零 C++ 工具链，仍是单 exe 自包含；
//! - 纯 CPU 推理，不占显存（Ollama 常驻已吃掉约 3 GB，再加模型会爆 8 GB 卡）；
//! - 模型常驻内存一次加载，短句转写约 0.7 s，不像起子进程那样每次重载模型。
//!
//! 前端 `web/src/audio.js` 送来的二进制帧就是 **16 kHz 单声道 float32 小端**
//! （浏览器侧已按 sampleRate/16000 降采样），无需再重采样；文本帧 `END`
//! 表示录音结束。这里只负责「PCM 数组 -> 中文句子」，WebSocket 帧循环在 http.rs。
//!
//! 找不到 DLL / 模型时静默降级为「不可用」，不影响键盘问答与其余功能。

// 本模块所有导出入口都是 unsafe fn，内部逐条 FFI 调用再套一层 unsafe 块
// 只会制造噪音；edition 2024 起默认要求显式块，这里整模块豁免。
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

// ─────────────────────── Windows 动态加载 ───────────────────────

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(lp_lib_file_name: *const u16, h_file: *mut c_void, dw_flags: u32) -> *mut c_void;
    fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
    fn SetDllDirectoryW(lp_path_name: *const u16) -> i32;
    fn GetShortPathNameW(lpsz_long_path: *const u16, lpsz_short_path: *mut u16, cch_buffer: u32) -> u32;
    fn FreeLibrary(h_lib_module: *mut c_void) -> i32;
}

/// `LOAD_WITH_ALTERED_SEARCH_PATH`：让 whisper.dll 的导入依赖（ggml.dll 等）
/// 优先从 DLL 所在目录解析，而不是 exe 所在目录。
const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

fn wide(p: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    p.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

/// whisper.cpp 在 Windows 上用 `fopen(const char*)` 打开模型，MSVC 的 fopen 不吃
/// UTF-8 路径。用户目录若含中文，temp 路径就会加载失败 —— 用 8.3 短名兜底，
/// 短名全是 ASCII，且不改动任何实际文件。
fn model_cstring(path: &Path) -> Option<CString> {
    let s = path.to_string_lossy();
    if s.is_ascii() {
        return CString::new(s.as_bytes()).ok();
    }
    let w = wide(path);
    let mut buf = vec![0u16; 1024];
    let n = unsafe { GetShortPathNameW(w.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
    if n == 0 || n as usize >= buf.len() {
        return None;
    }
    buf.truncate(n as usize);
    let short = String::from_utf16_lossy(&buf);
    CString::new(short.as_bytes()).ok()
}

unsafe fn sym<T: Copy>(h: *mut c_void, name: &[u8]) -> Option<T> {
    let p = GetProcAddress(h, name.as_ptr() as *const c_char);
    if p.is_null() {
        eprintln!("[asr] whisper.dll 缺少导出符号 {}", String::from_utf8_lossy(name));
        return None;
    }
    Some(std::mem::transmute_copy::<*mut c_void, T>(&p))
}

// ─────────────────────── C ABI（whisper.h b5130） ───────────────────────
//
// 布局由 include/whisper.h 精确抄写，并用 ctypes 原型实测过：
//   sizeof(whisper_full_params) = 304, sizeof(whisper_context_params) = 48
// 下面 #[cfg(test)] 里有 offset_of! 断言，DLL 升级导致布局漂移会立刻编译失败。

#[repr(C)]
#[derive(Clone, Copy)]
struct Ahead {
    n_text_layer: i32,
    n_head: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Aheads {
    n_heads: usize,
    heads: *const Ahead,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContextParams {
    use_gpu: bool,
    flash_attn: bool,
    gpu_device: i32,
    dtw_token_timestamps: bool,
    dtw_aheads_preset: i32,
    dtw_n_top: i32,
    dtw_aheads: Aheads,
    dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VadParams {
    threshold: f32,
    min_speech_duration_ms: i32,
    min_silence_duration_ms: i32,
    max_speech_duration_s: f32,
    speech_pad_ms: i32,
    samples_overlap: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Greedy {
    best_of: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BeamSearch {
    beam_size: i32,
    patience: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FullParams {
    strategy: i32,
    n_threads: i32,
    n_max_text_ctx: i32,
    offset_ms: i32,
    duration_ms: i32,
    translate: bool,
    no_context: bool,
    no_timestamps: bool,
    single_segment: bool,
    print_special: bool,
    print_progress: bool,
    print_realtime: bool,
    print_timestamps: bool,
    token_timestamps: bool,
    thold_pt: f32,
    thold_ptsum: f32,
    max_len: i32,
    split_on_word: bool,
    max_tokens: i32,
    debug_mode: bool,
    audio_ctx: i32,
    tdrz_enable: bool,
    suppress_regex: *const c_char,
    initial_prompt: *const c_char,
    carry_initial_prompt: bool,
    prompt_tokens: *const i32,
    prompt_n_tokens: i32,
    language: *const c_char,
    detect_language: bool,
    suppress_blank: bool,
    suppress_nst: bool,
    temperature: f32,
    max_initial_ts: f32,
    length_penalty: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
    greedy: Greedy,
    beam_search: BeamSearch,
    new_segment_callback: *const c_void,
    new_segment_callback_user_data: *const c_void,
    progress_callback: *const c_void,
    progress_callback_user_data: *const c_void,
    encoder_begin_callback: *const c_void,
    encoder_begin_callback_user_data: *const c_void,
    abort_callback: *const c_void,
    abort_callback_user_data: *const c_void,
    logits_filter_callback: *const c_void,
    logits_filter_callback_user_data: *const c_void,
    grammar_rules: *const c_void,
    n_grammar_rules: usize,
    i_start_rule: usize,
    grammar_penalty: f32,
    vad: bool,
    vad_model_path: *const c_char,
    vad_params: VadParams,
}

type FnDefaultCtxParams = unsafe extern "C" fn() -> ContextParams;
type FnInitFromFile = unsafe extern "C" fn(*const c_char, ContextParams) -> *mut c_void;
type FnDefaultFullParams = unsafe extern "C" fn(i32) -> FullParams;
type FnFull = unsafe extern "C" fn(*mut c_void, FullParams, *const f32, i32) -> i32;
type FnFree = unsafe extern "C" fn(*mut c_void);
type FnNSegments = unsafe extern "C" fn(*mut c_void) -> i32;
type FnSegmentText = unsafe extern "C" fn(*mut c_void, i32) -> *const c_char;
type FnBackendLoadAll = unsafe extern "C" fn(*const c_char);
type FnLogSet = unsafe extern "C" fn(Option<unsafe extern "C" fn(i32, *const c_char, *mut c_void)>, *mut c_void);

struct Fns {
    default_ctx_params: FnDefaultCtxParams,
    init_from_file: FnInitFromFile,
    default_full_params: FnDefaultFullParams,
    full: FnFull,
    free: FnFree,
    n_segments: FnNSegments,
    segment_text: FnSegmentText,
}

/// 默认吞掉 whisper.cpp 的模型加载日志（GUI 下 stderr 无人看，无头测试下刷屏）。
/// 设 `QX_ASR_DEBUG=1` 可恢复打印，便于现场排查。
unsafe extern "C" fn quiet_log(_level: i32, text: *const c_char, _user: *mut c_void) {
    if DEBUG_LOG.load(Ordering::Relaxed) && !text.is_null() {
        eprint!("{}", CStr::from_ptr(text).to_string_lossy());
    }
}

struct Engine {
    _h_ggml: *mut c_void,
    _h_whisper: *mut c_void,
    ctx: *mut c_void,
    f: Fns,
    threads: i32,
}

// 句柄与 ctx 只在本类型的 &mut 路径下使用，且全部访问都经 Asr 的 Mutex 串行化。
unsafe impl Send for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.f.free)(self.ctx);
            let _ = FreeLibrary(self._h_whisper);
            let _ = FreeLibrary(self._h_ggml);
        }
    }
}

impl Engine {
    unsafe fn load(dir: &Path, model: &Path) -> Result<Self, String> {
        let dll = |name: &str| -> Result<*mut c_void, String> {
            let p = dir.join(name);
            if !p.is_file() {
                return Err(format!("缺少 {}", p.display()));
            }
            let w = wide(&p);
            let h = LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
            if h.is_null() {
                return Err(format!("加载 {name} 失败（依赖 DLL 可能缺失）"));
            }
            Ok(h)
        };

        // 把 ASR 目录加入 DLL 搜索路径：ggml 会在运行时按文件名动态挑一个
        // ggml-cpu-*.dll 后端，只有让它看到同目录才找得到。
        SetDllDirectoryW(wide(dir).as_ptr());

        // 必须先注册动态 CPU 后端，否则 whisper_init 里 ggml 找不到 device，
        // 直接 GGML_ASSERT(device) 把进程 abort 掉（CLI 启动时替我们做了这步）。
        let h_ggml = dll("ggml.dll")?;
        let load_all: FnBackendLoadAll = sym(h_ggml, b"ggml_backend_load_all_from_path\0")
            .ok_or("ggml.dll 没有 ggml_backend_load_all_from_path")?;
        let dir_c = CString::new(dir.to_string_lossy().as_bytes())
            .map_err(|_| "ASR 目录路径含 NUL")?;
        load_all(dir_c.as_ptr());

        let h_whisper = dll("whisper.dll")?;
        let f = Fns {
            default_ctx_params: sym(h_whisper, b"whisper_context_default_params\0")
                .ok_or("缺 whisper_context_default_params")?,
            init_from_file: sym(h_whisper, b"whisper_init_from_file_with_params\0")
                .ok_or("缺 whisper_init_from_file_with_params")?,
            default_full_params: sym(h_whisper, b"whisper_full_default_params\0")
                .ok_or("缺 whisper_full_default_params")?,
            full: sym(h_whisper, b"whisper_full\0").ok_or("缺 whisper_full")?,
            free: sym(h_whisper, b"whisper_free\0").ok_or("缺 whisper_free")?,
            n_segments: sym(h_whisper, b"whisper_full_n_segments\0")
                .ok_or("缺 whisper_full_n_segments")?,
            segment_text: sym(h_whisper, b"whisper_full_get_segment_text\0")
                .ok_or("缺 whisper_full_get_segment_text")?,
        };
        let log_set: Option<FnLogSet> = sym(h_whisper, b"whisper_log_set\0");
        if let Some(log_set) = log_set {
            log_set(Some(quiet_log), std::ptr::null_mut());
        }

        let mut cp = (f.default_ctx_params)();
        cp.use_gpu = false; // 纯 CPU：显存留给 Ollama / 数字人
        let model_c = model_cstring(model).ok_or("模型路径无法转为 C 字符串")?;
        let ctx = (f.init_from_file)(model_c.as_ptr(), cp);
        if ctx.is_null() {
            return Err(format!("加载模型失败：{}", model.display()));
        }

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8) as i32)
            .unwrap_or(4);
        Ok(Self { _h_ggml: h_ggml, _h_whisper: h_whisper, ctx, f, threads })
    }

    unsafe fn transcribe(&self, samples: &[f32]) -> Option<String> {
        let prompt = CString::new(INITIAL_PROMPT).ok()?;
        // 0 = WHISPER_SAMPLING_GREEDY；默认值由 DLL 填充，只覆盖下面几项，
        // 这样 DLL 升级新增的字段不会因为我们手写结构体而漏初始化。
        let mut p = (self.f.default_full_params)(0);
        p.n_threads = self.threads;
        p.language = b"zh\0".as_ptr() as *const c_char;
        p.detect_language = false;
        p.translate = false;
        p.no_timestamps = true;
        p.single_segment = true;
        p.no_context = true;
        p.print_special = false;
        p.print_progress = false;
        p.print_realtime = false;
        p.print_timestamps = false;
        p.initial_prompt = prompt.as_ptr();
        p.temperature = 0.0;

        let rc = (self.f.full)(self.ctx, p, samples.as_ptr(), samples.len() as i32);
        if rc != 0 {
            return None;
        }
        let n = (self.f.n_segments)(self.ctx);
        let mut out = String::new();
        for i in 0..n.max(0) {
            let s = (self.f.segment_text)(self.ctx, i);
            if !s.is_null() {
                out.push_str(&CStr::from_ptr(s).to_string_lossy());
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
        // 现场排障开关：跳过 whisper.dll 加载，退回纯键盘问答。
        if std::env::var("QX_ASR_DISABLE").is_ok_and(|v| v != "0" && !v.is_empty()) {
            eprintln!("[asr] 已由 QX_ASR_DISABLE 关闭");
            *me.error.lock().unwrap() = "已由 QX_ASR_DISABLE 关闭".into();
            return me;
        }
        let t0 = Instant::now();
        let dir = root.join("asr");
        let model = pick_model(&dir);
        let result = match model {
            Some(m) => unsafe { Engine::load(&dir, &m) },
            None => Err(format!("{} 下没有 Whisper 模型", dir.display())),
        };
        match result {
            Ok(e) => {
                let ms = t0.elapsed().as_millis() as u64;
                *me.engine.lock().unwrap() = Some(e);
                me.load_ms.store(ms, Ordering::Relaxed);
                me.ready.store(true, Ordering::Relaxed);
                eprintln!("[asr] 离线识别就绪：{} ms", ms);
            }
            Err(e) => {
                eprintln!("[asr] 离线识别不可用：{e}");
                *me.error.lock().unwrap() = e;
            }
        }
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
        unsafe { engine.transcribe(&samples) }
    }
}

/// 模型优先级：显式 `QX_ASR_MODEL` > 内嵌 base（快，0.7 s 级）> 外置 small（更准但约 2 s）。
/// base 随 exe 分发，保证赛场只拷一个 exe 也能用；想换模型放个文件 + 设环境变量即可。
fn pick_model(dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("QX_ASR_MODEL") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        eprintln!("[asr] QX_ASR_MODEL 指向的文件不存在：{}", p.display());
    }
    ["ggml-base-q5_1.bin", "ggml-small-q5_1.bin", "ggml-base.bin", "ggml-small.bin"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// 清洗输入：去非有限值、截断 30 s、判静音。单独抽出来便于单测，不依赖 DLL。
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
    fn abi_layout_matches_whisper_h() {
        // 与 ctypes 实测的 b5130 头文件布局逐项对齐；DLL 升级后这里会先炸。
        assert_eq!(std::mem::size_of::<FullParams>(), 304);
        assert_eq!(std::mem::size_of::<ContextParams>(), 48);
        assert_eq!(std::mem::offset_of!(FullParams, n_threads), 4);
        assert_eq!(std::mem::offset_of!(FullParams, no_timestamps), 22);
        assert_eq!(std::mem::offset_of!(FullParams, single_segment), 23);
        assert_eq!(std::mem::offset_of!(FullParams, initial_prompt), 72);
        assert_eq!(std::mem::offset_of!(FullParams, language), 104);
        assert_eq!(std::mem::offset_of!(FullParams, temperature), 116);
        assert_eq!(std::mem::offset_of!(FullParams, vad_params), 280);
    }

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

    /// 真机冒烟：需要 asr/ 下有 DLL 与模型。
    /// `cargo test -p qx-desktop asr::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore = "需要 asr/ 下的 whisper.dll 与模型"]
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
