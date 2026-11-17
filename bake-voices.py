#!/usr/bin/env python3
"""
预合成脚本：读取 voice_manifest.json，用 Piper (zh_CN-huayan-medium) 批量合成 WAV，
保存到 tts-cache/，随 exe 分发。
比赛现场断网时，HTTP 层直接命中缓存，声音与本地 huayan 完全一致。

用法：
  python bake-voices.py                           # 合成全部
  python bake-voices.py --limit 20                # 只合成前20条（测试用）
"""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "qx-rust" / "voice_manifest.json"
TTS_CACHE = PROJECT_DIR / "qx-rust" / "tts-cache"
PIPER_EXE = PROJECT_DIR / "qx-rust" / "tts-test" / "piper" / "piper.exe"
PIPER_MODEL = PROJECT_DIR / "qx-rust" / "tts" / "zh_CN-huayan-medium.onnx"


def tts_key(text: str) -> str:
    """与 Rust 侧 tts_key 一致：FNV-1a 64-bit 哈希，前16位十六进制"""
    h = 0xCBF2_9CE4_8422_2325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{h:016x}"


def synth_piper(text: str, out_wav: Path) -> bool:
    """用 Piper 命令行离线合成（从 stdin 传文本）"""
    if not PIPER_EXE.exists():
        print(f"[ERR] Piper 可执行文件不存在: {PIPER_EXE}")
        return False
    if not PIPER_MODEL.exists():
        print(f"[ERR] Piper 模型不存在: {PIPER_MODEL}")
        return False

    try:
        result = subprocess.run(
            [
                str(PIPER_EXE),
                "--model", str(PIPER_MODEL),
                "--output_file", str(out_wav),
            ],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[ERR] Piper 合成失败: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"[ERR] Piper 异常: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="用 Piper zh_CN-huayan-medium 预合成离线语音")
    p.add_argument("--limit", type=int, default=None, help="只合成前 N 条（测试用）")
    args = p.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    TTS_CACHE.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    failed = 0
    total = len(manifest)

    for item in manifest:
        if args.limit and count + skipped >= args.limit:
            break

        text = item.get("text", "").strip()
        name = item.get("name", "")
        if not text:
            continue

        hex_key = tts_key(text)
        out_wav = TTS_CACHE / f"{hex_key}.wav"

        if out_wav.exists():
            skipped += 1
            continue

        try:
            ok = synth_piper(text, out_wav)
            if ok:
                count += 1
                print(f"[OK] {name[:40]}... -> {out_wav.name}")
            else:
                failed += 1
                print(f"[ERR] {name[:40]}...: Piper 合成失败")
        except Exception as e:
            failed += 1
            print(f"[ERR] {name[:40]}...: {e}")

    print(f"\n=== 完成 ===")
    print(f"  新合成: {count} 条")
    print(f"  已存在: {skipped} 条")
    print(f"  失败:   {failed} 条")
    print(f"  总计:   {count + skipped + failed} 条")
    print(f"输出目录: {TTS_CACHE}")


if __name__ == "__main__":
    main()
