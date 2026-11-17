//! 离线中文语音识别 — 已禁用
//!
//! whisper-rs 已移除：构建需从 GitHub 下载 kaldi-native-fbank，
//! 现场离线无法下载。语音识别功能已降级为不可用，不影响 TTS 语音合成链路。
//!
//! 保持与原接口兼容，始终返回“不可用”。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};

pub struct Asr {
    engine: Mutex<Option<()>>,
    load_ms: AtomicU64,
    ready: AtomicBool,
    error: Mutex<String>,
}

impl Asr {
    /// 同步加载（始终不可用）。
    pub fn load(root: &Path) -> Arc<Self> {
        let _ = root;
        let me = Arc::new(Self {
            engine: Mutex::new(None),
            load_ms: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            error: Mutex::new("whisper-rs 已移除，语音识别不可用".into()),
        });
        me
    }

    /// 后台线程加载（始终不可用）。
    pub fn spawn_load(root: PathBuf) -> Arc<Self> {
        let _ = root;
        let me = Arc::new(Self {
            engine: Mutex::new(None),
            load_ms: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            error: Mutex::new("whisper-rs 已移除，语音识别不可用".into()),
        });
        me
    }

    pub fn available(&self) -> bool {
        false
    }

    pub fn load_ms(&self) -> u64 {
        0
    }

    pub fn error(&self) -> String {
        self.error.lock().unwrap().clone()
    }

    /// 始终返回 None。
    pub fn transcribe(&self, _pcm: &[f32]) -> Option<String> {
        None
    }
}

/// 始终不做任何事。
pub fn append_pcm(_buf: &mut Vec<f32>, _bytes: &[u8]) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_always_unavailable() {
        let asr = Asr::load(Path::new("."));
        assert!(!asr.available());
        assert_eq!(asr.load_ms(), 0);
        assert!(asr.transcribe(&[0.0; 100]).is_none());
    }
}
