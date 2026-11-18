//! 离线中文语音识别 —— cpal 采集 + valle-asr（Qwen3-ASR 0.6B）
//!
//! 前端不再通过 WebAudio 采集麦克风：`/ws/asr` 只作为控制/结果通道。连接建立后，
//! 本模块用 cpal 打开系统默认麦克风，在本地做能量 VAD 分段，再把每段音频交给
//! valle-asr 的 Qwen3-ASR 后端识别，并通过 `partial` / `final` 消息回传前端。
//!
//! 这样音频不经过 WebView，也不需要浏览器麦克风权限；仍保持原有 hold-to-talk
//! 交互和 WebSocket 协议不变。模型完全本地加载，不联网、不写音频到磁盘。
//!
//! 模型目录查找顺序（命中即用）：
//! 1. 环境变量 `QX_ASR_MODEL_DIR`（绝对路径，或相对 bundle 根目录）；
//! 2. exe 同级的 `asr/qwen3-asr-0.6b` / `asr/Qwen3-ASR-0.6B`（便于把权重放在 exe 旁分发）；
//! 3. bundle 根目录的 `asr/...`（build.rs 打包进 exe 时使用），也兼容 `asr/models/...`；
//! 4. debug 构建下 qx-rust/asr 源目录，方便 `cargo run` 直接调试。
//!
//! 识别全程在内存中完成，音频不落盘；模型不会在运行时下载。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use valle_asr::models::qwen3::Qwen3;
use valle_asr::{AsrEngine, Audio, CancellationToken, TimestampMode, TranscribeOptions};

/// valle-asr 中注册的模型 ID；与 Qwen3::load 的第一个参数保持一致。
const MODEL_ID: &str = "qwen3-asr";
/// 官方 checkpoint 目录名（valle-asr models.json）。
const MODEL_DIR_NAME: &str = "qwen3-asr-0.6b";
/// Qwen3-ASR 固定使用 16 kHz 单声道。
const SAMPLE_RATE: u32 = 16_000;
/// 少于 0.2 秒的碎片不值得送入模型。
const MIN_AUDIO_SAMPLES: usize = SAMPLE_RATE as usize / 5;
/// 前端按住说话上限 28 秒，服务端到点自动收尾，避免无限占用麦克风。
const MAX_SESSION_SECS: usize = 28;
/// 等待后台 tokenizer/配置加载完成的上限（权重在首次推理时才加载）。
const ENGINE_WAIT: Duration = Duration::from_secs(30);

// ───────────────────────── VAD 参数 ─────────────────────────

/// 20 ms 一帧，便于做轻量能量 VAD。
const VAD_FRAME: usize = SAMPLE_RATE as usize / 50;
/// 起音前保留 300 ms 音频，避免吞掉第一个字。
const VAD_PREROLL_FRAMES: usize = 15;
/// 连续 60 ms 超阈值才判定开始说话，滤掉键盘/敲击等短脉冲。
const VAD_START_FRAMES: usize = 3;
/// 连续 600 ms 静音判定一句话结束。
const VAD_END_SILENCE_FRAMES: usize = 30;
/// 句尾保留 200 ms，避免把尾音切掉。
const VAD_TRAILING_KEEP_FRAMES: usize = 10;
/// 太短的段落不送模型。
const VAD_MIN_SEGMENT: usize = SAMPLE_RATE as usize / 4;
/// 说话过程中每 6 秒出一次中间结果，兼顾字幕实时感与 CPU 推理开销
/// （Qwen3-ASR 0.6B 比 whisper base 重，过于频繁会积压音频）。
const VAD_PARTIAL_STEP: usize = SAMPLE_RATE as usize * 6;

/// 后台加载完成的引擎槽。加载期间 `transcribe` 会等待，而不是直接丢音频。
enum EngineSlot {
    Loading,
    Ready(AsrEngine),
    Failed,
}

pub struct Asr {
    model_dir: Option<PathBuf>,
    failed: AtomicBool,
    load_ms: AtomicU64,
    error: Mutex<String>,
    engine: Mutex<EngineSlot>,
    engine_cv: Condvar,
}

/// 采集线程回传给 WebSocket 线程的事件。
#[derive(Debug, Clone)]
pub enum AsrEvent {
    /// 中间结果：已确认的前文 + 当前段落的最新识别文本。
    Partial(String),
    /// 最终结果；`audio_ms` 用于日志与性能统计。
    Final { text: String, audio_ms: u64 },
    /// 麦克风或识别引擎错误；前端会显示在状态栏。
    Error(String),
}

/// 一次 hold-to-talk 的采集会话。
///
/// `finish()` 停止采集并允许完成最后一段转写；`cancel()` 用于客户端断开时
/// 立即中止推理（valle-asr 支持协作式取消）。会话 drop 时自动取消。
pub struct CaptureSession {
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    events: Receiver<AsrEvent>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl CaptureSession {
    pub fn finish(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn try_event(&self) -> Option<AsrEvent> {
        self.events.try_recv().ok()
    }

    pub fn is_finished(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.cancel();
        if let Some(handle) = self.join.take() {
            // 已结束时立刻回收；仍在转写时不能阻塞 WebSocket 线程，交给取消标志收尾。
            if self.done.load(Ordering::SeqCst) {
                let _ = handle.join();
            } else {
                drop(handle);
            }
        }
    }
}

impl Asr {
    /// 启动后台加载。tokenizer/配置读取很快；1.8 GB 权重在首次推理时懒加载，
    /// 随后后台预热一次，避免用户第一次说话卡十几秒。
    pub fn spawn_load(root: PathBuf) -> Arc<Self> {
        let model_dir = resolve_model_dir(&root);
        let (error, slot, failed) = match &model_dir {
            Some(dir) => (
                format!("离线语音识别模型加载中：{}", dir.display()),
                EngineSlot::Loading,
                false,
            ),
            None => (missing_model_error(), EngineSlot::Failed, true),
        };
        let me = Arc::new(Self {
            model_dir: model_dir.clone(),
            failed: AtomicBool::new(failed),
            load_ms: AtomicU64::new(0),
            error: Mutex::new(error),
            engine: Mutex::new(slot),
            engine_cv: Condvar::new(),
        });
        if let Some(dir) = model_dir {
            let bg = Arc::clone(&me);
            let spawned = std::thread::Builder::new()
                .name("qx-asr-load".into())
                .spawn(move || bg.load_engine(dir));
            if let Err(e) = spawned {
                me.fail(format!("启动语音识别加载线程失败：{e}"));
            }
        } else {
            tracing::warn!(
                root = %root.display(),
                "未找到 Qwen3-ASR 模型，离线语音识别不可用"
            );
        }
        me
    }

    /// 模型资产是否可用。tokenizer 仍在加载时也返回 true，前端可提前显示
    /// 「加载中」，真正转写会等待引擎就绪。
    pub fn available(&self) -> bool {
        self.model_dir.is_some() && !self.failed.load(Ordering::Acquire)
    }

    pub fn load_ms(&self) -> u64 {
        self.load_ms.load(Ordering::Relaxed)
    }

    pub fn error(&self) -> String {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// 兼容旧接口：默认不可取消的同步转写，供测试/简单调用使用。
    #[allow(dead_code)] // 二进制目标里仅测试调用；保留简单同步 API 便于复用
    pub fn transcribe(&self, pcm: &[f32]) -> Option<String> {
        self.transcribe_with_budget(pcm, &CancellationToken::default(), 448)
    }

    /// 启动一次麦克风采集。返回的会话由 WebSocket 线程持有并轮询事件。
    pub fn start_capture(self: &Arc<Self>) -> Result<CaptureSession, String> {
        if !self.available() {
            return Err(self.error());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let me = Arc::clone(self);
        let stop_bg = Arc::clone(&stop);
        let cancel_bg = Arc::clone(&cancel);
        let done_bg = Arc::clone(&done);
        let handle = std::thread::Builder::new()
            .name("qx-asr-capture".into())
            .spawn(move || {
                let _done = DoneGuard(done_bg);
                capture_worker(me, stop_bg, cancel_bg, tx);
            })
            .map_err(|e| format!("启动麦克风采集线程失败：{e}"))?;
        Ok(CaptureSession {
            stop,
            cancel,
            done,
            events: rx,
            join: Some(handle),
        })
    }

    fn load_engine(self: Arc<Self>, dir: PathBuf) {
        let started = Instant::now();
        let result = Qwen3::load(MODEL_ID, &dir, None)
            .and_then(|model| {
                let mut engine = AsrEngine::new();
                engine.register(model)?;
                Ok(engine)
            })
            .map_err(|e| e.to_string());
        match result {
            Ok(engine) => {
                {
                    let mut slot = self.engine.lock().unwrap_or_else(|e| e.into_inner());
                    *slot = EngineSlot::Ready(engine);
                }
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = String::new();
                self.failed.store(false, Ordering::Release);
                self.engine_cv.notify_all();
                tracing::info!(
                    model_dir = %dir.display(),
                    "离线语音识别引擎已注册，开始预热权重"
                );
                self.warm_up();
                // load_ms 表示「真正可转写」的耗时，包含权重载入与预热，
                // 自检面板据此显示就绪状态，而不是只显示 tokenizer 读取时间。
                let mut ms = started.elapsed().as_millis() as u64;
                if ms == 0 {
                    ms = 1;
                }
                self.load_ms.store(ms, Ordering::Relaxed);
                tracing::info!(load_ms = ms, "离线语音识别引擎就绪");
            }
            Err(e) => self.fail(format!("离线语音识别模型加载失败：{e}")),
        }
    }

    /// 用 0.5 秒静音跑一次小预算推理，把 1.8 GB 权重载入内存并缓存。
    /// `QX_ASR_NO_WARMUP=1` 可关闭，适合内存紧张的部署。
    fn warm_up(self: &Arc<Self>) {
        if std::env::var("QX_ASR_NO_WARMUP").as_deref() == Ok("1") {
            tracing::info!("QX_ASR_NO_WARMUP=1，跳过 ASR 权重预热");
            return;
        }
        let pcm = vec![0.0f32; SAMPLE_RATE as usize / 2];
        let started = Instant::now();
        // 预热只验证权重可加载，不追求文本：给 8 个 token 的上限，
        // 避免静音输入在极端情况下一直解码到 448 token。
        let text = self.transcribe_with_budget(&pcm, &CancellationToken::default(), 8);
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            text_chars = text.as_deref().map(|t| t.chars().count()).unwrap_or(0),
            "ASR 权重预热完成"
        );
    }

    fn fail(&self, message: String) {
        tracing::warn!(error = %message, "离线语音识别不可用");
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = message;
        self.failed.store(true, Ordering::Release);
        {
            let mut slot = self.engine.lock().unwrap_or_else(|e| e.into_inner());
            *slot = EngineSlot::Failed;
        }
        self.engine_cv.notify_all();
    }

    fn transcribe_cancellable(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
    ) -> Option<String> {
        self.transcribe_with_budget(pcm, cancellation, 448)
    }

    fn transcribe_with_budget(
        &self,
        pcm: &[f32],
        cancellation: &CancellationToken,
        max_new_tokens: usize,
    ) -> Option<String> {
        if pcm.len() < MIN_AUDIO_SAMPLES || cancellation.is_cancelled() {
            return None;
        }
        if !self.available() {
            return None;
        }

        // valle-asr 要求有限且位于 [-1, 1] 的样本；麦克风偶发 NaN/削波在这里统一清洗。
        let mut samples = Vec::with_capacity(pcm.len());
        for &s in pcm {
            samples.push(if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            });
        }
        let audio = match Audio::from_mono(samples, SAMPLE_RATE) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(error = %e, "ASR 输入音频无效");
                return None;
            }
        };

        let mut slot = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + ENGINE_WAIT;
        loop {
            match &*slot {
                EngineSlot::Ready(_) => break,
                EngineSlot::Failed => return None,
                EngineSlot::Loading => {
                    let now = Instant::now();
                    if now >= deadline {
                        tracing::warn!("等待 ASR 引擎加载超时");
                        return None;
                    }
                    let (guard, _) = self
                        .engine_cv
                        .wait_timeout(slot, deadline - now)
                        .unwrap_or_else(|e| e.into_inner());
                    slot = guard;
                }
            }
        }
        let engine = match &mut *slot {
            EngineSlot::Ready(engine) => engine,
            _ => return None,
        };
        if cancellation.is_cancelled() {
            return None;
        }

        let options = TranscribeOptions {
            // 固定普通话，避免短句语种检测抖动；Qwen3-ASR 支持中文/英文等。
            language: Some("Chinese".into()),
            // 只出文本，不加载额外的 1.8 GB 强制对齐模型。
            timestamps: TimestampMode::None,
            chunk_seconds: 30,
            max_new_tokens,
            context: String::new(),
            cancellation: cancellation.clone(),
        };
        let started = Instant::now();
        match engine.transcribe(MODEL_ID, &audio, &options) {
            Ok(transcript) => {
                let text = transcript.text.trim().to_string();
                if text.is_empty() {
                    tracing::debug!(
                        audio_ms = transcript.duration_ms,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "ASR 返回空文本"
                    );
                    None
                } else {
                    tracing::info!(
                        audio_ms = transcript.duration_ms,
                        text_chars = text.chars().count(),
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "valle-asr 转写完成"
                    );
                    Some(text)
                }
            }
            Err(e) => {
                if cancellation.is_cancelled() {
                    tracing::debug!("ASR 推理已取消");
                } else {
                    tracing::warn!(error = %e, "valle-asr 转写失败");
                }
                None
            }
        }
    }
}

struct DoneGuard(Arc<AtomicBool>);

impl Drop for DoneGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

// ───────────────────────── cpal 采集 ─────────────────────────

enum CaptureMsg {
    /// 单声道、设备原采样率的 f32 采样。
    Audio(Vec<f32>),
    Error(String),
}

fn capture_worker(
    asr: Arc<Asr>,
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    events: Sender<AsrEvent>,
) {
    let (audio_tx, audio_rx) = mpsc::channel::<CaptureMsg>();
    let (stream, input_rate) = match open_microphone(audio_tx) {
        Ok(v) => v,
        Err(e) => {
            let _ = events.send(AsrEvent::Error(format!("麦克风不可用：{e}")));
            let _ = events.send(AsrEvent::Final {
                text: String::new(),
                audio_ms: 0,
            });
            return;
        }
    };

    let mut resampler = LinearResampler::new(input_rate);
    let mut vad = Vad::new();
    let mut committed = String::new();
    let mut captured = 0usize;
    let mut mic_error: Option<String> = None;
    let cancellation = CancellationToken::from_shared_flag(Arc::clone(&cancel));
    let max_samples = MAX_SESSION_SECS * SAMPLE_RATE as usize;

    'capture: loop {
        // 音频回调通常每 10–40 ms 就有一块数据，不能只在 recv 超时时看停止标志，
        // 否则松开按钮后仍会一直录到 28 秒上限。
        if stop.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst) {
            break 'capture;
        }
        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(CaptureMsg::Audio(mono)) => {
                let samples = resampler.push(&mono);
                if samples.is_empty() {
                    continue;
                }
                captured += samples.len();
                let actions = vad.push(&samples);
                if !handle_vad_actions(&asr, &actions, &mut committed, &cancellation, &events) {
                    return;
                }
                if captured >= max_samples {
                    tracing::info!(
                        audio_ms = captured as u64 * 1000 / SAMPLE_RATE as u64,
                        "达到 ASR 会话上限，自动收尾"
                    );
                    break 'capture;
                }
            }
            Ok(CaptureMsg::Error(e)) => {
                mic_error = Some(e);
                break 'capture;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst) {
                    break 'capture;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break 'capture,
        }
    }

    // 先停采集，再把停止前已经进入队列的音频处理完，避免丢掉最后半句话。
    drop(stream);
    while let Ok(msg) = audio_rx.try_recv() {
        match msg {
            CaptureMsg::Audio(mono) => {
                let samples = resampler.push(&mono);
                if samples.is_empty() {
                    continue;
                }
                captured += samples.len();
                let actions = vad.push(&samples);
                if !handle_vad_actions(&asr, &actions, &mut committed, &cancellation, &events) {
                    return;
                }
                if captured >= max_samples {
                    break;
                }
            }
            CaptureMsg::Error(e) => {
                mic_error.get_or_insert(e);
            }
        }
    }

    if cancel.load(Ordering::SeqCst) {
        return;
    }

    let mut actions = Vec::new();
    vad.flush(&mut actions);
    if !handle_vad_actions(&asr, &actions, &mut committed, &cancellation, &events) {
        return;
    }

    if let Some(e) = mic_error {
        let _ = events.send(AsrEvent::Error(format!("麦克风采集中断：{e}")));
    }
    if committed.is_empty() && !asr.available() {
        let _ = events.send(AsrEvent::Error(asr.error()));
    }

    let text = committed.trim().to_string();
    tracing::info!(
        audio_ms = captured as u64 * 1000 / SAMPLE_RATE as u64,
        text_chars = text.chars().count(),
        "valle-asr 会话识别完成"
    );
    let _ = events.send(AsrEvent::Final {
        text,
        audio_ms: captured as u64 * 1000 / SAMPLE_RATE as u64,
    });
}

fn open_microphone(tx: Sender<CaptureMsg>) -> Result<(cpal::Stream, u32), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "未找到系统默认麦克风".to_string())?;
    let name = device.to_string();
    let supported = device
        .default_input_config()
        .map_err(|e| format!("读取麦克风默认配置失败：{e}"))?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let input_rate = config.sample_rate;
    let channels = config.channels as usize;
    if channels == 0 {
        return Err("麦克风声道数为 0".into());
    }

    let stream = match sample_format {
        SampleFormat::F32 => {
            build_stream::<f32>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::F64 => {
            build_stream::<f64>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::I8 => build_stream::<i8>(&device, config.clone(), channels, tx.clone(), &tx),
        SampleFormat::I16 => {
            build_stream::<i16>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::I32 => {
            build_stream::<i32>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::I64 => {
            build_stream::<i64>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::U8 => build_stream::<u8>(&device, config.clone(), channels, tx.clone(), &tx),
        SampleFormat::U16 => {
            build_stream::<u16>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::U32 => {
            build_stream::<u32>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        SampleFormat::U64 => {
            build_stream::<u64>(&device, config.clone(), channels, tx.clone(), &tx)
        }
        other => return Err(format!("不支持的麦克风采样格式：{other:?}")),
    }
    .map_err(|e| format!("创建麦克风采集流失败：{e}"))?;

    stream
        .play()
        .map_err(|e| format!("启动麦克风采集失败：{e}"))?;
    tracing::info!(
        device = %name,
        sample_rate = input_rate,
        channels = channels,
        sample_format = ?sample_format,
        "cpal 麦克风已打开"
    );
    Ok((stream, input_rate))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    channels: usize,
    tx: Sender<CaptureMsg>,
    err_tx: &Sender<CaptureMsg>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_fn = make_err_fn(err_tx.clone());
    device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if data.is_empty() || channels == 0 {
                return;
            }
            let mut mono = Vec::with_capacity(data.len() / channels);
            for frame in data.chunks_exact(channels) {
                let mut sum = 0.0f32;
                for &sample in frame {
                    sum += f32::from_sample_(sample);
                }
                mono.push(sum / channels as f32);
            }
            if !mono.is_empty() {
                // 无界 channel 不会阻塞音频回调；转写慢时音频先排队，
                // 会话最多 28 秒（约 1.8 MB f32），不会无限增长。
                let _ = tx.send(CaptureMsg::Audio(mono));
            }
        },
        err_fn,
        None,
    )
}

fn make_err_fn(tx: Sender<CaptureMsg>) -> impl FnMut(cpal::Error) + Send + 'static {
    let mut warned = false;
    move |e| {
        match e.kind() {
            // 这三类不会终止流：偶发欠载只造成一次音频毛刺，设备切换已由 cpal
            // 自动改路，实时调度被拒只是延迟变差。若把它们当成致命错误，用户
            // 刚按下按钮就会看到「麦克风采集中断」。
            cpal::ErrorKind::Xrun
            | cpal::ErrorKind::DeviceChanged
            | cpal::ErrorKind::RealtimeDenied => {
                if !warned {
                    warned = true;
                    tracing::warn!(error = %e, kind = ?e.kind(), "cpal 采集告警（可恢复，继续录音）");
                }
            }
            _ => {
                let _ = tx.send(CaptureMsg::Error(e.to_string()));
            }
        }
    }
}

// ───────────────────────── 16 kHz 重采样 ─────────────────────────

/// 流式线性重采样：设备常用 48 kHz / 44.1 kHz，模型固定要 16 kHz。
struct LinearResampler {
    /// 每个输出采样对应的输入采样数。
    step: f64,
    /// 当前缓冲区内的下一个输出位置。
    pos: f64,
    buf: VecDeque<f32>,
}

impl LinearResampler {
    fn new(input_rate: u32) -> Self {
        let step = if input_rate == 0 {
            1.0
        } else {
            input_rate as f64 / SAMPLE_RATE as f64
        };
        Self {
            step,
            pos: 0.0,
            buf: VecDeque::new(),
        }
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        self.buf.extend(input.iter().copied());
        let mut out = Vec::with_capacity((input.len() as f64 / self.step).ceil() as usize + 2);
        while self.pos + 1.0 < self.buf.len() as f64 {
            let i = self.pos.floor() as usize;
            let frac = (self.pos - i as f64) as f32;
            let s0 = self.buf[i];
            let s1 = self.buf[i + 1];
            out.push(s0 + (s1 - s0) * frac);
            self.pos += self.step;
        }
        let consume = self.pos.floor() as usize;
        if consume > 0 {
            self.buf.drain(..consume);
            self.pos -= consume as f64;
        }
        out
    }
}

// ───────────────────────── 能量 VAD ─────────────────────────

enum VadAction {
    /// 句中断点：发中间结果，但不把文本写入 committed。
    Progress(Vec<f32>),
    /// 一句话结束：转写后追加到 committed。
    SegmentDone(Vec<f32>),
}

struct Vad {
    frame: Vec<f32>,
    preroll: VecDeque<Vec<f32>>,
    segment: Vec<f32>,
    active: bool,
    noise: f32,
    speech_run: usize,
    silence_run: usize,
    last_partial_len: usize,
}

impl Vad {
    fn new() -> Self {
        Self {
            frame: Vec::with_capacity(VAD_FRAME),
            preroll: VecDeque::new(),
            segment: Vec::new(),
            active: false,
            // 初始底噪略高于绝对安静，避免空调/风扇被当成语音。
            noise: 0.004,
            speech_run: 0,
            silence_run: 0,
            last_partial_len: 0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<VadAction> {
        let mut actions = Vec::new();
        let mut offset = 0;
        while offset < samples.len() {
            let take = (VAD_FRAME - self.frame.len()).min(samples.len() - offset);
            self.frame
                .extend_from_slice(&samples[offset..offset + take]);
            offset += take;
            if self.frame.len() == VAD_FRAME {
                let frame = std::mem::take(&mut self.frame);
                self.frame = Vec::with_capacity(VAD_FRAME);
                self.process_frame(&frame, &mut actions);
            }
        }
        actions
    }

    fn flush(&mut self, actions: &mut Vec<VadAction>) {
        if !self.frame.is_empty() {
            let frame = std::mem::take(&mut self.frame);
            self.frame = Vec::with_capacity(VAD_FRAME);
            self.process_frame(&frame, actions);
        }
        if self.active {
            self.active = false;
            self.silence_run = 0;
            self.speech_run = 0;
            self.last_partial_len = 0;
            if self.segment.len() >= VAD_MIN_SEGMENT {
                actions.push(VadAction::SegmentDone(std::mem::take(&mut self.segment)));
            } else {
                self.segment.clear();
            }
        }
    }

    fn process_frame(&mut self, frame: &[f32], actions: &mut Vec<VadAction>) {
        if frame.is_empty() {
            return;
        }
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        // 自适应底噪：安静时缓慢跟随环境，说话时保持不动（避免把语音当噪声）。
        let threshold = (self.noise * 3.0).clamp(0.006, 0.08);

        if !self.active {
            self.preroll.push_back(frame.to_vec());
            while self.preroll.len() > VAD_PREROLL_FRAMES {
                self.preroll.pop_front();
            }
            if rms > threshold {
                self.speech_run += 1;
                if self.speech_run >= VAD_START_FRAMES {
                    self.active = true;
                    self.silence_run = 0;
                    self.segment.clear();
                    for f in self.preroll.drain(..) {
                        self.segment.extend_from_slice(&f);
                    }
                    self.last_partial_len = 0;
                }
            } else {
                self.speech_run = 0;
                if rms < threshold * 0.8 {
                    self.noise = (self.noise * 0.95 + rms * 0.05).clamp(0.0005, 0.05);
                }
            }
            return;
        }

        self.segment.extend_from_slice(frame);
        if rms < threshold * 0.7 {
            self.silence_run += 1;
        } else {
            self.silence_run = 0;
        }

        if self.silence_run >= VAD_END_SILENCE_FRAMES {
            // 去掉大部分句尾静音，只留 200 ms，减少模型输入长度。
            let drop_frames = self.silence_run.saturating_sub(VAD_TRAILING_KEEP_FRAMES);
            let drop_samples = (drop_frames * frame.len()).min(self.segment.len());
            self.segment.truncate(self.segment.len() - drop_samples);
            self.active = false;
            self.silence_run = 0;
            self.speech_run = 0;
            self.last_partial_len = 0;
            self.preroll.clear();
            if self.segment.len() >= VAD_MIN_SEGMENT {
                actions.push(VadAction::SegmentDone(std::mem::take(&mut self.segment)));
            } else {
                self.segment.clear();
            }
        } else if self.segment.len() >= self.last_partial_len + VAD_PARTIAL_STEP
            && self.segment.len() >= SAMPLE_RATE as usize * 2
        {
            actions.push(VadAction::Progress(self.segment.clone()));
            self.last_partial_len = self.segment.len();
        }
    }
}

fn handle_vad_actions(
    asr: &Asr,
    actions: &[VadAction],
    committed: &mut String,
    cancellation: &CancellationToken,
    events: &Sender<AsrEvent>,
) -> bool {
    for action in actions {
        if cancellation.is_cancelled() {
            return false;
        }
        match action {
            VadAction::Progress(segment) => {
                if let Some(text) = asr.transcribe_cancellable(segment, cancellation) {
                    let preview = join_text(committed, &text);
                    if !preview.is_empty() {
                        let _ = events.send(AsrEvent::Partial(preview));
                    }
                }
            }
            VadAction::SegmentDone(segment) => {
                if let Some(text) = asr.transcribe_cancellable(segment, cancellation) {
                    *committed = join_text(committed, &text);
                    let _ = events.send(AsrEvent::Partial(committed.clone()));
                }
            }
        }
    }
    true
}

/// 中文直接拼接；英文/数字之间补一个空格，避免 "howareyou"。
fn join_text(left: &str, right: &str) -> String {
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    let need_space = left
        .chars()
        .last()
        .is_some_and(|c| c.is_ascii_alphanumeric())
        && right
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric());
    if need_space {
        format!("{left} {right}")
    } else {
        format!("{left}{right}")
    }
}

// ───────────────────────── 模型目录 ─────────────────────────

fn resolve_model_dir(root: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(value) = std::env::var_os("QX_ASR_MODEL_DIR") {
        let path = PathBuf::from(value);
        candidates.push(if path.is_absolute() {
            path
        } else {
            root.join(path)
        });
    }

    if let Some(dir) = candidates.into_iter().find(|dir| is_model_dir(dir)) {
        return Some(dir);
    }

    let mut bases = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            bases.push(dir.to_path_buf());
        }
    }
    bases.push(root.to_path_buf());
    // `cargo run` 时 bundle 根是临时解压目录，补一个源码目录方便开发调试。
    #[cfg(debug_assertions)]
    bases.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../"));

    for base in bases {
        for rel in [
            Path::new("asr").join(MODEL_DIR_NAME),
            Path::new("asr").join("Qwen3-ASR-0.6B"),
            Path::new("asr").join("models").join(MODEL_DIR_NAME),
            Path::new("asr").join("models").join("Qwen3-ASR-0.6B"),
            PathBuf::from("asr"),
        ] {
            let dir = base.join(rel);
            if is_model_dir(&dir) {
                return Some(dir);
            }
        }
    }
    None
}

fn is_model_dir(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    // 官方 0.6B 是单文件 model.safetensors；也兼容分片 checkpoint 的索引。
    if !dir.join("model.safetensors").is_file()
        && !dir.join("model.safetensors.index.json").is_file()
    {
        return false;
    }
    // valle-asr 优先读 tokenizer.json；旧转换脚本只产出 vocab/merges 也能用。
    dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file()
            && dir.join("merges.txt").is_file()
            && dir.join("tokenizer_config.json").is_file())
}

fn missing_model_error() -> String {
    format!("未找到离线语音识别模型（请放置 asr/{MODEL_DIR_NAME}，或设置 QX_ASR_MODEL_DIR）")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dir_requires_all_checkpoint_files() {
        let dir = std::env::temp_dir().join(format!("qx-asr-model-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!is_model_dir(&dir));
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"{}").unwrap();
        std::fs::write(dir.join("vocab.json"), b"{}").unwrap();
        std::fs::write(dir.join("merges.txt"), b"").unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), b"{}").unwrap();
        assert!(is_model_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);

        // 新版 checkpoint 只带 tokenizer.json 也应识别
        let dir = std::env::temp_dir().join(format!("qx-asr-model-hf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert!(is_model_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);

        // 分片 checkpoint 只带 model.safetensors.index.json 也应识别
        let dir = std::env::temp_dir().join(format!("qx-asr-model-shard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        assert!(is_model_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resampler_48k_to_16k_keeps_amplitude() {
        let mut resampler = LinearResampler::new(48_000);
        let out = resampler.push(&[0.5f32; 48_000]);
        assert!(
            (out.len() as i64 - 16_000).abs() <= 3,
            "48k→16k 输出长度异常：{}",
            out.len()
        );
        assert!(out.iter().all(|v| (*v - 0.5).abs() < 1e-5));
    }

    #[test]
    fn vad_detects_a_speech_burst() {
        let mut vad = Vad::new();
        let mut actions = Vec::new();
        let silence = vec![0.0f32; VAD_FRAME];
        for _ in 0..25 {
            actions.extend(vad.push(&silence));
        }
        let tone: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (i as f32 * 0.12).sin() * 0.25)
            .collect();
        actions.extend(vad.push(&tone));
        for _ in 0..50 {
            actions.extend(vad.push(&silence));
        }
        vad.flush(&mut actions);

        let segments = actions
            .iter()
            .filter(|a| matches!(a, VadAction::SegmentDone(_)))
            .count();
        assert!(segments >= 1, "应至少切出一段语音");
        let total: usize = actions
            .iter()
            .map(|a| match a {
                VadAction::Progress(s) | VadAction::SegmentDone(s) => s.len(),
            })
            .sum();
        assert!(
            total >= SAMPLE_RATE as usize / 2,
            "送检音频过短：{total} samples"
        );
    }

    #[test]
    fn join_text_handles_chinese_and_english() {
        assert_eq!(join_text("你好", "世界"), "你好世界");
        assert_eq!(join_text("hello", "world"), "hello world");
        assert_eq!(join_text("", "你好"), "你好");
    }

    #[test]
    fn asr_without_model_reports_unavailable() {
        let asr = Asr::spawn_load(PathBuf::from("."));
        // 仓库/部署目录若已放模型则跳过该断言，避免环境相关误报。
        if asr.model_dir.is_none() {
            assert!(!asr.available());
            assert!(asr.transcribe(&[0.0; 100]).is_none());
            assert!(asr.start_capture().is_err(), "没有模型时不应打开麦克风");
        }
    }
}
