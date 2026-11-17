//! 构建期把原页面与主题包打包进 exe：把 `web/` 与 `themes/` 压成
//! `OUT_DIR/bundle.zip`，运行时由 main.rs 解压到临时目录后由 Rust HTTP 服务托管。
//! 这样产出的 qx-desktop.exe 是自包含的单个文件，不依赖任何外部目录 / Python。

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use zip::write::FileOptions;
use zip::CompressionMethod;

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // crates/qx-desktop → crates → qx-rust
    let root = Path::new(manifest).join("../../");

    embed_windows_icon();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR 未设置");
    let zip_path = Path::new(&out_dir).join("bundle.zip");
    let file = File::create(&zip_path).expect("无法创建 bundle.zip");
    let mut zw = zip::ZipWriter::new(file);
    let opts = FileOptions::default().compression_method(CompressionMethod::Deflated);

    add_dir(&mut zw, &root.join("web"), Path::new("web"), &opts);
    add_dir(&mut zw, &root.join("themes"), Path::new("themes"), &opts);
    // 离线神经语音资产（VITS 模型 + tokens.txt + espeak-ng-data），随 exe 分发、
    // 启动释放到临时目录。推理引擎（onnxruntime + espeak-ng G2P）已由
    // sherpa-rs 静态链入，不再依赖平台专用动态库。
    add_dir(&mut zw, &root.join("tts"), Path::new("tts"), &opts);
    // 赛前预生成的标准普通话讲解语音（可选增强）：按 tts-cache 键命名，
    // 解压到 bundle 根目录后运行时直接命中磁盘缓存，演示全程离线。
    // 目录为空时 add_dir 静默跳过，等价于纯离线 Piper 方案。
    add_dir(&mut zw, &root.join("prebuilt-voices"), Path::new("tts-cache"), &opts);
    // 离线语音识别（①）：whisper.cpp 已由 whisper-rs 静态编译进主程序，
    // 随包只需 base 量化模型（不再有 whisper.dll / ggml*.dll）。
    add_asr_assets(&mut zw, &root.join("asr"), Path::new("asr"), &opts);
    zw.finish().expect("打包失败");

    // 页面/主题/语音资产有变动时重新打包
    println!("cargo:rerun-if-changed={}", root.join("web").display());
    println!("cargo:rerun-if-changed={}", root.join("themes").display());
    println!("cargo:rerun-if-changed={}", root.join("tts").display());
    println!("cargo:rerun-if-changed={}", root.join("prebuilt-voices").display());
    println!("cargo:rerun-if-changed={}", root.join("asr").display());
    // voice_bake 的固定话术常量用 include_str! 校验 nlu.rs，文案变动需重新编译
    println!("cargo:rerun-if-changed={}", root.join("crates/qx-core/src/nlu.rs").display());
}

fn add_dir(zw: &mut zip::ZipWriter<File>, base: &Path, prefix: &Path, opts: &FileOptions) {
    let Ok(rd) = fs::read_dir(base) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let rel = prefix.join(entry.file_name());
        if path.is_dir() {
            add_dir(zw, &path, &rel, opts);
        } else {
            let Ok(data) = fs::read(&path) else { continue };
            zw.start_file(rel.to_string_lossy().replace('\\', "/"), *opts)
                .expect("写入条目失败");
            zw.write_all(&data).expect("写入数据失败");
        }
    }
}

/// 打包 ASR 运行时：只带 ggml 模型（引擎静态链入主程序，无需 DLL）。
/// 量化模型已高度压缩，Deflate 几乎压不动还拖慢每次增量构建，改为 Stored 直存。
/// Windows 专用的 dll（whisper.dll/ggml*.dll/SDL2.dll）在跨平台版里已不需要，
/// 目录里如有残留也不再打包。
fn add_asr_assets(zw: &mut zip::ZipWriter<File>, dir: &Path, prefix: &Path, opts: &FileOptions) {
    let Ok(rd) = fs::read_dir(dir) else {
        println!("cargo:warning=asr/ 不存在，离线语音识别将不可用");
        return;
    };
    let stored = FileOptions::default().compression_method(CompressionMethod::Stored);
    let mut count = 0;
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let is_model = name.eq_ignore_ascii_case("ggml-base-q5_1.bin") && path.is_file()
            || name.eq_ignore_ascii_case("ggml-base.bin");
        if !is_model {
            continue;
        }
        let Ok(data) = fs::read(&path) else { continue };
        let o = if is_model { &stored } else { opts };
        let rel = prefix.join(name);
        zw.start_file(rel.to_string_lossy().replace('\\', "/"), *o)
            .expect("写入 ASR 条目失败");
        zw.write_all(&data).expect("写入 ASR 数据失败");
        count += 1;
    }
    println!("cargo:warning=ASR 随包资产：{count} 个文件");
}

/// 把 resources/app.rc 编成 .res 并交给链接器，让 exe 带上图标与版本信息。
///
/// 走「自己调 rc.exe + rustc-link-arg」而不是引第三方构建依赖：本机已装
/// Windows Kits，rc.exe 就在 `Windows Kits\10\bin\<版本>\x64\` 下，少一个
/// 需要联网拉的 crate，构建链路更短。
/// 找不到 rc.exe 时只告警、不中断——图标是加分项，不该让整个构建失败。
fn embed_windows_icon() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rc_src = manifest.join("resources").join("app.rc");
    let ico = manifest.join("resources").join("app.ico");
    if !rc_src.exists() || !ico.exists() {
        println!("cargo:warning=缺少 resources/app.rc 或 app.ico，跳过 exe 图标");
        return;
    }
    println!("cargo:rerun-if-changed={}", rc_src.display());
    println!("cargo:rerun-if-changed={}", ico.display());

    let Some(rc) = find_rc() else {
        println!("cargo:warning=未找到 rc.exe（Windows Kits），跳过 exe 图标");
        return;
    };
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR 未设置");
    let res = Path::new(&out_dir).join("app.res");
    let _ = fs::remove_file(&res);

    let status = std::process::Command::new(&rc)
        .arg("/nologo")
        .arg("/fo")
        .arg(&res)
        .arg(&rc_src)
        .status();

    match status {
        Ok(s) if s.success() && res.exists() => {
            println!("cargo:rustc-link-arg={}", res.display());
        }
        other => println!("cargo:warning=rc.exe 编译资源失败：{other:?}，exe 将沿用默认图标"),
    }
}

/// 顺序：PATH 里的 rc → Windows Kits 下按版本号从高到低取第一个存在的。
fn find_rc() -> Option<PathBuf> {
    if std::process::Command::new("rc").arg("/?").output().is_ok() {
        return Some(PathBuf::from("rc"));
    }
    let kits = std::env::var("ProgramFiles(x86)")
        .ok()
        .map(|b| PathBuf::from(b).join("Windows Kits").join("10").join("bin"))
        .or_else(|| {
            std::env::var("ProgramFiles")
                .ok()
                .map(|b| PathBuf::from(b).join("Windows Kits").join("10").join("bin"))
        })?;

    let mut versions: Vec<PathBuf> = fs::read_dir(&kits)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    for v in versions.iter().rev() {
        for arch in ["x64", "x86", "arm64"] {
            let p = v.join(arch).join("rc.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}
