import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BRAND = Path(r"C:\Users\ghhjhjh\Desktop\AI-Tour-Guide\ppt\qiaoxiang-deck\brand")
OUT = BRAND / "png"
OUT.mkdir(exist_ok=True)

JOBS = [
    ("logo-icon.svg", [(512, 512), (256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)], False),
    ("logo-icon-small.svg", [(48, 48), (32, 32), (24, 24), (16, 16)], False),
    ("logo-mark.svg", [(1024, 1024), (512, 512), (256, 256)], True),
    ("logo-mark-dark.svg", [(1024, 1024), (512, 512), (256, 256)], True),
    ("logo-lockup.svg", [(1640, 536), (820, 268)], True),
    ("logo-seal.svg", [(512, 512), (256, 256), (128, 128)], False),
]


def fit_svg(text: str) -> str:
    m = re.search(r"<svg[^>]*>", text)
    tag = m.group(0)
    new_tag = re.sub(r'\s(?:width|height)="[^"]*"', "", tag)
    new_tag = new_tag[:-1] + ' style="width:100%;height:100%;display:block">'
    return text.replace(tag, new_tag, 1)


def main() -> int:
    log = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, sizes, transparent in JOBS:
            svg_path = BRAND / name
            if not svg_path.exists():
                log.append(f"MISSING {name}")
                continue
            body = fit_svg(svg_path.read_text(encoding="utf-8"))
            bg = "transparent" if transparent else "#ffffff"
            for w, h in sizes:
                page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
                page.set_content(
                    f'<html><body style="margin:0;padding:0;background:{bg}">'
                    f'<div style="width:{w}px;height:{h}px">{body}</div></body></html>',
                    wait_until="load",
                )
                page.wait_for_timeout(200)
                stem = Path(name).stem.replace("logo-", "")
                target = OUT / f"{stem}-{w}.png"
                page.screenshot(
                    path=str(target),
                    omit_background=transparent,
                    clip={"x": 0, "y": 0, "width": w, "height": h},
                )
                page.close()
                log.append(f"{target.name} {w}x{h} {target.stat().st_size}B")
        browser.close()

    (BRAND / "render2.log").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
