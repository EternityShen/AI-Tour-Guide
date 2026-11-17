//! 把 Qwen2.5-VL 的 GGUF + mmproj 追加到 exe 尾部（PE overlay），做成单文件交付。
//!
//! 纯 Rust 原生工具，零 Python、零第三方依赖。用法：
//!     cargo build --release -p embed-model
//!     embed-model.exe <原始exe> <主模型.gguf> <mmproj.gguf> <输出exe>
//!
//! trailer 布局与 `crates/qx-desktop/src/embedded_model.rs`（运行时读取端）严格同步，
//! 改一处必须改两处：
//!   0   magic "QXEMBED2" (8)
//!   8   version u32 = 2
//!   12  file count u32 = 2
//!   16  model_size u64
//!   24  mmproj_size u64
//!   96  payload_size u64 = model_size + mmproj_size
//!   104 trailer_size u64 = 128

use std::fs::File;
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"QXEMBED2";
const VERSION: u32 = 2;
const TRAILER_SIZE: usize = 128;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("用法：{} <原始exe> <主模型.gguf> <mmproj.gguf> <输出exe>", args[0]);
        std::process::exit(2);
    }
    let (src, model, mmproj, dst) = (&args[1], &args[2], &args[3], &args[4]);

    if same_file(src, dst) {
        eprintln!("输出路径不能与原始 exe 相同（会破坏构建产物）");
        std::process::exit(1);
    }
    for p in [src, model, mmproj] {
        if !std::path::Path::new(p).is_file() {
            eprintln!("找不到输入文件：{p}");
            std::process::exit(1);
        }
    }

    std::fs::copy(src, dst).unwrap_or_else(|e| die(&format!("复制 exe 失败：{e}")));
    let model_size = append(dst, model).unwrap_or_else(|e| die(&format!("写入模型失败：{e}")));
    let mmproj_size = append(dst, mmproj).unwrap_or_else(|e| die(&format!("写入 mmproj 失败：{e}")));

    let mut t = [0u8; TRAILER_SIZE];
    t[0..8].copy_from_slice(MAGIC);
    t[8..12].copy_from_slice(&VERSION.to_le_bytes());
    t[12..16].copy_from_slice(&2u32.to_le_bytes());
    t[16..24].copy_from_slice(&model_size.to_le_bytes());
    t[24..32].copy_from_slice(&mmproj_size.to_le_bytes());
    t[96..104].copy_from_slice(&(model_size + mmproj_size).to_le_bytes());
    t[104..112].copy_from_slice(&(TRAILER_SIZE as u64).to_le_bytes());

    let mut out = File::options()
        .append(true)
        .open(dst)
        .unwrap_or_else(|e| die(&format!("打开输出 exe 失败：{e}")));
    out.write_all(&t).unwrap_or_else(|e| die(&format!("写 trailer 失败：{e}")));

    let gi = 1u64 << 30;
    let total = std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0);
    println!("原始 exe      : {} B", std::fs::metadata(src).map(|m| m.len()).unwrap_or(0));
    println!("主模型        : {} B ({:.2} GiB)", model_size, model_size as f64 / gi as f64);
    println!("mmproj        : {} B ({:.2} GiB)", mmproj_size, mmproj_size as f64 / gi as f64);
    println!("单文件交付 exe: {}  {} B ({:.2} GiB)", dst, total, total as f64 / gi as f64);
}

fn same_file(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

fn append(dst: &str, src_path: &str) -> std::io::Result<u64> {
    let mut src = File::open(src_path)?;
    let mut out = File::options().append(true).open(dst)?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut n = 0u64;
    loop {
        let r = src.read(&mut buf)?;
        if r == 0 {
            break;
        }
        out.write_all(&buf[..r])?;
        n += r as u64;
    }
    Ok(n)
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
