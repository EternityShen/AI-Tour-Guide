//! 从 exe 尾部的 PE overlay 读出「内嵌」的 Qwen2.5-VL 模型并释放到临时目录。
//!
//! 3 GiB 级模型不能用 `include_bytes!` 塞进 bundle.zip：LLVM 生成不了这么大的
//! 静态数组，链接器也会爆内存。做法是构建期由 `tools/embed_model.py` 把模型
//! 追加到 exe 文件尾（PE overlay，Windows 加载器不映射这段数据），运行时从
//! `current_exe()` 读回来，首次释放到临时目录、之后按大小校验跳过。
//! 这样交付物仍是**一个 exe**，运行时零 Python、零外部模型文件。
//!
//! trailer 固定 128 字节（小端），布局：
//!   0   magic "QXEMBED2" (8)
//!   8   version u32 = 2
//!   12  file count u32 = 2
//!   16  model_size u64
//!   24  mmproj_size u64
//!   32  reserved (64)
//!   96  payload_size u64 = model_size + mmproj_size
//!   104 trailer_size u64 = 128
//!   112 reserved (16)
//!
//! exe 尾部结构：[主模型字节][mmproj 字节][trailer]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"QXEMBED2";
const VERSION: u32 = 2;
const TRAILER_SIZE: u64 = 128;
const CACHE_DIR: &str = "qx-embedded-model-v2";

pub struct EmbeddedModels {
    pub model: PathBuf,
    pub mmproj: PathBuf,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetShortPathNameW(lpsz_long_path: *const u16, lpsz_short_path: *mut u16, cch_buffer: u32) -> u32;
}

fn wide(p: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    p.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

/// llama.cpp 用 `fopen(const char*)` 打开模型，MSVC 的 fopen 不吃 UTF-8 路径。
/// 中文用户名下 `%TEMP%` 会带非 ASCII，转成 8.3 短名（全 ASCII）兜底。
pub fn ascii_path(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    if s.is_ascii() {
        return s;
    }
    let w = wide(p);
    let mut buf = vec![0u16; 1024];
    let n = unsafe { GetShortPathNameW(w.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
    if n == 0 || n as usize >= buf.len() {
        return s;
    }
    buf.truncate(n as usize);
    String::from_utf16_lossy(&buf)
}

/// 读出并释放内嵌模型。exe 没有 overlay（普通构建 / 未打包）时返回 None，
/// 由调用方回退到 `--model` 或 Ollama，行为与旧版一致。
pub fn release() -> Option<EmbeddedModels> {
    let exe = std::env::current_exe().ok()?;
    let mut f = File::open(&exe).ok()?;
    let total = f.metadata().ok()?.len();
    if total < TRAILER_SIZE {
        return None;
    }
    f.seek(SeekFrom::End(-(TRAILER_SIZE as i64))).ok()?;
    let mut t = [0u8; TRAILER_SIZE as usize];
    f.read_exact(&mut t).ok()?;
    if &t[0..8] != MAGIC {
        return None;
    }
    let u64at = |o: usize| -> Option<u64> { Some(u64::from_le_bytes(t[o..o + 8].try_into().ok()?)) };
    if u32::from_le_bytes(t[8..12].try_into().ok()?) != VERSION {
        return None;
    }
    let model_size = u64at(16)?;
    let mmproj_size = u64at(24)?;
    let payload = u64at(96)?;
    if u64at(104)? != TRAILER_SIZE || payload != model_size + mmproj_size || model_size == 0 {
        return None;
    }
    let start = total
        .checked_sub(TRAILER_SIZE)?
        .checked_sub(payload)?;

    let dir = std::env::temp_dir().join(CACHE_DIR);
    fs::create_dir_all(&dir).ok()?;
    let model = dir.join("model.gguf");
    let mmproj = dir.join("mmproj.gguf");

    f.seek(SeekFrom::Start(start)).ok()?;
    copy_section(&mut f, &model, model_size)?;
    copy_section(&mut f, &mmproj, mmproj_size)?;
    eprintln!(
        "[embed] 内嵌模型就绪：{:.2} GiB + {:.2} GiB → {}",
        model_size as f64 / (1 << 30) as f64,
        mmproj_size as f64 / (1 << 30) as f64,
        dir.display()
    );
    Some(EmbeddedModels { model, mmproj })
}

/// 目标文件已存在且大小一致就跳过写入（但要同步移动源指针），
/// 否则先写 .part 再原子改名，避免中途退出留下半截模型。
fn copy_section(src: &mut File, dst: &Path, size: u64) -> Option<()> {
    if fs::metadata(dst).map(|m| m.len() == size).unwrap_or(false) {
        src.seek(SeekFrom::Current(size as i64)).ok()?;
        return Some(());
    }
    let part = dst.with_extension("part");
    let mut out = File::create(&part).ok()?;
    let mut left = size;
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    while left > 0 {
        let want = left.min(buf.len() as u64) as usize;
        let n = src.read(&mut buf[..want]).ok()?;
        if n == 0 {
            return None;
        }
        out.write_all(&buf[..n]).ok()?;
        left -= n as u64;
    }
    out.flush().ok()?;
    drop(out);
    let _ = fs::remove_file(dst);
    fs::rename(&part, dst).ok()?;
    Some(())
}
