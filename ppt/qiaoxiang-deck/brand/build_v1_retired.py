"""侨乡 · 江门 AI 智能讲解员 —— 标志生成器

单一几何源，输出四版 SVG，保证图形完全一致，避免手写四份坐标漂移。
调参只改本文件顶部的常量，然后重跑渲染脚本。
"""

from pathlib import Path

BRAND = Path(r"C:\Users\ghhjhjh\Desktop\AI-Tour-Guide\ppt\qiaoxiang-deck\brand")

# ── 品牌色 ────────────────────────────────────────────────
INK = "#1F2A2E"     # 碉楼青
PAPER = "#F5F1E8"   # 宣纸米
SEAL = "#A8402F"    # 侨批朱砂
SEAL_LIGHT = "#D9634A"   # 深底上的朱砂（提亮）
SEAL_MARK = "#C0503A"    # 浅底上的朱砂（压深）

# ── 统一几何（512 画布）────────────────────────────────────
ENV_X, ENV_Y, ENV_W, ENV_H, ENV_R = 106, 220, 300, 200, 22   # 信封
FLAP_INSET, FLAP_DROP = 26, 92                                # 封口折线内缩 / 下垂
BAR_W, BAR_GAP = 22, 32                                       # 声波条宽 / 间隙
BAR_BOTTOM = 228                                              # 条底（伸入信封 8px，被信封遮住形成无缝）
BAR_TOPS = (168, 130, 92, 130, 168)                           # 五根条的顶端 y
RADIUS_BG = 116                                               # 底板圆角

BAR_CX = [ENV_X + ENV_W // 2 + i * (BAR_W + BAR_GAP) - 2 * (BAR_W + BAR_GAP) for i in range(5)]


def waves(stroke: str, width: int = BAR_W) -> str:
    bars = "\n".join(
        f'    <line x1="{cx}" y1="{BAR_BOTTOM}" x2="{cx}" y2="{top}"/>'
        for cx, top in zip(BAR_CX, BAR_TOPS)
    )
    return (
        f'  <g stroke="{stroke}" stroke-width="{width}" stroke-linecap="round">\n'
        f"{bars}\n  </g>"
    )


def envelope(fill: str, flap: str, flap_width: int = 18) -> str:
    left = ENV_X + FLAP_INSET
    right = ENV_X + ENV_W - FLAP_INSET
    apex_y = ENV_Y + FLAP_DROP
    return (
        f'  <rect x="{ENV_X}" y="{ENV_Y}" width="{ENV_W}" height="{ENV_H}" rx="{ENV_R}" fill="{fill}"/>\n'
        f'  <path d="M{left} {ENV_Y + 30} L{ENV_X + ENV_W // 2} {apex_y} L{right} {ENV_Y + 30}" '
        f'fill="none" stroke="{flap}" stroke-width="{flap_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def svg(body: str, w: int, h: int, label: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{label}">\n'
        f"  <title>{label}</title>\n  <desc>{desc}</desc>\n"
        f"{body}\n</svg>\n"
    )


# ── 1. 方形图标（深底）──────────────────────────────────────
icon = (
    f'  <rect width="512" height="512" rx="{RADIUS_BG}" fill="{INK}"/>\n'
    + waves(SEAL_LIGHT)
    + "\n"
    + envelope(PAPER, INK)
)
(BRAND / "logo-icon.svg").write_text(
    svg(icon, 512, 512, "侨乡江门 AI 智能讲解员 标志",
        "深青底方形图标：米色侨批信封，封口为深青折线，顶部五根朱砂竖条为讲解声波。"),
    encoding="utf-8",
)

# ── 2. 图形标（无底板）──────────────────────────────────────
# 2a 浅底：深青信封 + 米白折线 + 压深朱砂声波
mark = waves(SEAL_MARK) + "\n" + envelope(INK, PAPER)
(BRAND / "logo-mark.svg").write_text(
    svg(mark, 512, 512, "侨乡江门 AI 智能讲解员 图形标",
        "无底板版本，用于宣纸米等浅色背景：深青信封 + 朱砂声波。"),
    encoding="utf-8",
)

# 2b 深底反白：米白信封 + 深青折线 + 提亮朱砂声波
mark_dark = waves(SEAL_LIGHT) + "\n" + envelope(PAPER, INK)
(BRAND / "logo-mark-dark.svg").write_text(
    svg(mark_dark, 512, 512, "侨乡江门 AI 智能讲解员 图形标（反白）",
        "无底板反白版本，用于碉楼青等深色背景：米白信封 + 朱砂声波。"),
    encoding="utf-8",
)

# ── 3. 印章标（朱底白文）────────────────────────────────────
scale = 0.66
cx, cy = 256, 256
shift = f'translate({cx},{cy}) scale({scale}) translate(-256,-256)'
seal_body = (
    f'  <circle cx="256" cy="256" r="240" fill="{SEAL}"/>\n'
    f'  <circle cx="256" cy="256" r="210" fill="none" stroke="{PAPER}" stroke-width="5" opacity="0.55"/>\n'
    f'  <g transform="{shift}">\n'
    + "\n".join("  " + ln for ln in (waves(PAPER, 24) + "\n" + envelope(PAPER, SEAL, 22)).splitlines())
    + "\n  </g>"
)
(BRAND / "logo-seal.svg").write_text(
    svg(seal_body, 512, 512, "侨乡江门 AI 智能讲解员 印章标",
        "朱砂圆章，白文（阴刻）效果，用于文档压印与物料角标。"),
    encoding="utf-8",
)

# ── 4. 横版主标 ─────────────────────────────────────────────
ICON_BOX, PAD_X, PAD_Y = 180, 28, 46
k = ICON_BOX / 512
lockup_body = (
    f'  <g transform="translate({PAD_X},{PAD_Y}) scale({k:.5f})">\n'
    + "\n".join("  " + ln for ln in (waves(SEAL_MARK) + "\n" + envelope(INK, PAPER)).splitlines())
    + "\n  </g>\n"
    f'  <text x="244" y="108" fill="{SEAL}" font-family="\'Source Han Sans SC\',\'Microsoft YaHei\',sans-serif" '
    f'font-size="27" font-weight="500" letter-spacing="5">侨乡 · 江门</text>\n'
    f'  <text x="240" y="188" fill="{INK}">'
    f'<tspan font-family="\'Segoe UI\',\'Helvetica Neue\',Arial,sans-serif" font-size="62" font-weight="700" '
    f'letter-spacing="1">AI</tspan>'
    f'<tspan dx="20" font-family="\'Source Han Serif SC\',\'Noto Serif CJK SC\',\'SimSun\',serif" '
    f'font-size="76" font-weight="700" letter-spacing="4">智能讲解员</tspan>'
    f"</text>"
)
(BRAND / "logo-lockup.svg").write_text(
    svg(lockup_body, 820, 250, "侨乡 · 江门 AI 智能讲解员",
        "图形标 + 中文字标，用于封面、页眉、文档首屏。"),
    encoding="utf-8",
)

print("BAR_CX =", BAR_CX)
print("wrote 4 svg")

# ── 5. 小尺寸简化版（16–32px 专用）──────────────────────────
# 原理：像素越少，可保留的信息越少。去掉外侧两根条、放大信封、加粗折线，
# 让小尺寸下仍读作「方块 + 浅色信封 + 橙色信号」。
S_ENV_X, S_ENV_Y, S_ENV_W, S_ENV_H, S_ENV_R = 70, 250, 372, 230, 30
S_BAR_W, S_BAR_GAP, S_BAR_BOTTOM = 34, 34, 258
S_BAR_TOPS = (176, 120, 176)
S_BAR_CX = [256 - S_BAR_W - S_BAR_GAP, 256, 256 + S_BAR_W + S_BAR_GAP]
s_bars = "\n".join(
    f'    <line x1="{cx}" y1="{S_BAR_BOTTOM}" x2="{cx}" y2="{top}"/>'
    for cx, top in zip(S_BAR_CX, S_BAR_TOPS)
)
s_flap_left = S_ENV_X + 30
s_flap_right = S_ENV_X + S_ENV_W - 30
small = (
    f'  <rect width="512" height="512" rx="100" fill="{INK}"/>\n'
    f'  <g stroke="{SEAL_LIGHT}" stroke-width="{S_BAR_W}" stroke-linecap="round">\n{s_bars}\n  </g>\n'
    f'  <rect x="{S_ENV_X}" y="{S_ENV_Y}" width="{S_ENV_W}" height="{S_ENV_H}" rx="{S_ENV_R}" fill="{PAPER}"/>\n'
    f'  <path d="M{s_flap_left} {S_ENV_Y + 30} L256 {S_ENV_Y + 104} L{s_flap_right} {S_ENV_Y + 30}" '
    f'fill="none" stroke="{INK}" stroke-width="26" stroke-linecap="round" stroke-linejoin="round"/>'
)
(BRAND / "logo-icon-small.svg").write_text(
    svg(small, 512, 512, "侨乡江门 AI 智能讲解员 小尺寸图标",
        "16–32 像素专用简化版：三根声波条、放大信封、加粗折线，保证小尺寸可辨。"),
    encoding="utf-8",
)
print("wrote 5 svg (incl. small)")
