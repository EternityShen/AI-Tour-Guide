"""拼装多尺寸 ICO：小尺寸用简化版源图，大尺寸用完整版源图。

PIL 的 save(..., sizes=[...]) 只能从单一源图缩放，做不出「16px 用简化版」的效果，
因此直接按 ICO 文件结构拼装（每条目内嵌 PNG，Vista 以后 Windows 全支持）。
"""

import struct
from pathlib import Path

BRAND = Path(r"C:\Users\ghhjhjh\Desktop\AI-Tour-Guide\ppt\qiaoxiang-deck\brand")
PNG = BRAND / "png"

# 尺寸 → 源图：16/24/32/48 用简化版（去外侧两根条、放大信封），64 以上用完整版
PLAN = [
    (16, "icon-small-16.png"),
    (24, "icon-small-24.png"),
    (32, "icon-small-32.png"),
    (48, "icon-small-48.png"),
    (64, "icon-64.png"),
    (128, "icon-128.png"),
    (256, "icon-256.png"),
]


def build_ico(entries: list[tuple[int, bytes]]) -> bytes:
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    directory, blob = b"", b""
    for size, png in entries:
        dim = 0 if size >= 256 else size          # 256 在 ICO 目录里记作 0
        directory += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
        blob += png
    return header + directory + blob


def main() -> int:
    entries = []
    for size, name in PLAN:
        path = PNG / name
        if not path.exists():
            raise SystemExit(f"missing {name}")
        entries.append((size, path.read_bytes()))
    ico = build_ico(entries)
    target = BRAND / "logo-icon.ico"
    target.write_bytes(ico)
    print(f"logo-icon.ico {len(ico)} B | entries {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
