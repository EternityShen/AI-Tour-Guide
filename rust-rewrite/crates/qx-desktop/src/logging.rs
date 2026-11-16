//! 运行日志：按天滚动的文件 + 可选控制台 + panic 记录。
//!
//! GUI 子系统没有控制台，现场排障只能靠文件，所以默认把 `info` 及以上写到
//! `%LOCALAPPDATA%\QiaoxiangGuide\logs\qx-guide.log.<日期>`。可用环境变量调整：
//!
//!   * `QX_LOG_DIR`  日志目录（便携版可指到 U 盘或项目目录）
//!   * `QX_LOG`      过滤级别，语法同 `RUST_LOG`，例如 `debug`、
//!                   `qx_desktop=debug,warn`；未设置时默认 `info`
//!   * `RUST_LOG`    兜底，优先级低于 `QX_LOG`
//!
//! 文件保留 14 天，启动时清理过期文件；进程正常退出时 `WorkerGuard` 会冲刷
//! 缓冲区。panic 会连同线程名、位置与 backtrace 一起写进日志。

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// 默认只给本项目开 info，依赖库保持 warn，避免刷屏。
const DEFAULT_FILTER: &str = "warn,qx_desktop=info,qx_core=info";
const LOG_PREFIX: &str = "qx-guide.log";
const RETENTION: Duration = Duration::from_secs(14 * 24 * 60 * 60);

static LOG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// 日志 guard：必须活到进程结束，否则文件写入线程可能来不及冲刷。
pub struct LogGuard {
    _file_guard: WorkerGuard,
}

/// 初始化全局日志与 panic hook。应尽早调用，且只调用一次。
pub fn init() -> LogGuard {
    let dir = prepare_dir();
    let _ = LOG_DIR.set(dir.clone());

    let file_appender = tracing_appender::rolling::daily(&dir, LOG_PREFIX);
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true);

    let console_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(false);

    let spec = log_spec();
    let filter = build_filter(spec.as_deref());

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(console_layer)
        .init();

    install_panic_hook();

    tracing::info!(
        dir = %dir.display(),
        filter = %spec.as_deref().unwrap_or(DEFAULT_FILTER),
        "日志系统已就绪（按天滚动，保留 14 天）"
    );

    LogGuard { _file_guard: file_guard }
}

/// 当前日志目录，供 `/api/stats` 与自检面板展示。
pub fn dir() -> Option<&'static Path> {
    LOG_DIR.get().map(PathBuf::as_path)
}

/// 当前最新的日志文件（按修改时间取最新），供自检面板展示。
pub fn latest_file() -> Option<PathBuf> {
    let dir = dir()?;
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(LOG_PREFIX))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// 目录优先级：`QX_LOG_DIR` > `%LOCALAPPDATA%` > 临时目录。
/// 首选目录不可写时自动退到临时目录，保证日志系统不会挡住启动。
fn prepare_dir() -> PathBuf {
    let preferred = std::env::var_os("QX_LOG_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("LOCALAPPDATA")
                .filter(|v| !v.is_empty())
                .map(|v| PathBuf::from(v).join("QiaoxiangGuide").join("logs"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("QiaoxiangGuide").join("logs"));

    if writable_dir(&preferred) {
        cleanup_old_logs(&preferred);
        return preferred;
    }

    let fallback = std::env::temp_dir().join("QiaoxiangGuide").join("logs");
    let _ = std::fs::create_dir_all(&fallback);
    cleanup_old_logs(&fallback);
    fallback
}

/// 不能只看 `create_dir_all`：目录已存在但只读时它同样返回 Ok，
/// 真正开写才会失败。这里用探针文件确认可写，再决定是否回退到临时目录。
fn writable_dir(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(format!(".qx-log-probe-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// 删除 14 天前的旧日志；只碰本程序前缀，避免清理共享目录里的其他文件。
fn cleanup_old_logs(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with(LOG_PREFIX) {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age > RETENTION {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn log_spec() -> Option<String> {
    for key in ["QX_LOG", "RUST_LOG"] {
        if let Ok(v) = std::env::var(key) {
            let v = v.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn build_filter(spec: Option<&str>) -> EnvFilter {
    match spec {
        Some(s) => EnvFilter::try_new(s).unwrap_or_else(|e| {
            eprintln!("[log] 日志过滤规则无效（{e}），回退到 {DEFAULT_FILTER}");
            EnvFilter::new(DEFAULT_FILTER)
        }),
        None => EnvFilter::new(DEFAULT_FILTER),
    }
}

/// panic 在 GUI 下没有控制台可见，必须落盘；保留默认 hook 以便终端运行时仍有输出。
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<非字符串 panic>");
        // 只在 panic 时抓一次，代价可以接受；strip 后的 exe 至少还有地址与调用栈。
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!(
            thread = %thread_name,
            location = %location,
            backtrace = %backtrace,
            "进程 panic：{payload}"
        );
        default_hook(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_is_two_weeks() {
        assert_eq!(RETENTION.as_secs(), 14 * 24 * 60 * 60);
    }

    #[test]
    fn default_filter_keeps_project_visible() {
        assert!(DEFAULT_FILTER.contains("qx_desktop=info"));
        assert!(DEFAULT_FILTER.contains("qx_core=info"));
    }
}
