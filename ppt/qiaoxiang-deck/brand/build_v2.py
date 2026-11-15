"""侨乡 · 江门 AI 智能讲解员 —— 标志生成器 v2

核心概念：碉楼开口说话
  1. 塔身     = 开平碉楼（江门唯一世界文化遗产，六展项之首，答辩封面）
  2. 塔顶竖条 = 碉楼顶层柱廊 / 燕子窝（建筑特征）
                同时读作声波条（讲解员开口）
  3. 拱门     = 展厅入口（观众由此进入）

一形双读，不靠两个图标拼接，所以「建筑」与「说话」是同一个形。

单一几何源：调参只改本文件顶部常量。
"""

from dataclasses import dataclass
from pathlib import Path

BRAND = Path(r"C:\Users\ghhjhjh\Desktop\AI-Tour-Guide\ppt\qiaoxiang-deck\brand")

# ── 品牌色 ────────────────────────────────────────────────
INK = "#1F2A2E"        # 碉楼青
PAPER = "#F5F1E8"      # 宣纸米
SEAL = "#A8402F"       # 侨批朱砂
SEAL_LIGHT = "#D9634A"  # 深底朱砂（提亮）
SEAL_MARK = "#C0503A"   # 浅底朱砂（压深）

S = 512


# ── 几何 ──────────────────────────────────────────────────
@dataclass
class Tower:
    """碉楼几何。所有坐标基于 512 画布。"""

    body_l: int = 178      # 塔身直筒左
    body_r: int = 334      # 塔身直筒右 → 宽 156，瘦高 1:1.92 才有碉楼的挺立感
    top: int = 150         # 塔身顶（挑檐顶）
    bot: int = 450         # 塔基底
    eave: int = 18         # 顶部挑檐外挑
    eave_h: int = 20       # 挑檐高度（碉楼线脚是厚的一圈，细了像帽沿）
    radius: int = 0        # 碉楼是砌体，不倒圆角

    bar_w: int = 16        # 声波条宽
    bar_gap: int = 18
    bar_tops: tuple = (88, 74, 64, 74, 88)  # 五条，平头阶梯包络
    bar_bot: int = 156     # 条底伸入塔身 6px，吃掉接缝

    win_w: int = 12        # 碉楼窗窄高，竖长条
    win_h: int = 24
    win_rows: tuple = (198, 248, 298)

    door_w: int = 44       # 拱门窄（碉楼门为防御性小门）
    door_arc: int = 386    # 拱顶 y


T = Tower()


def bars(t: Tower, color: str) -> str:
    """声波条用平头矩形，不用圆头线：圆头 + 高低差会被读成火焰或皇冠。"""
    step = t.bar_w + t.bar_gap
    total = len(t.bar_tops) * t.bar_w + (len(t.bar_tops) - 1) * t.bar_gap
    left = (S - total) / 2
    rects = "\n".join(
        f'    <rect x="{left + i * step:.0f}" y="{top}" '
        f'width="{t.bar_w}" height="{t.bar_bot - top}" rx="2"/>'
        for i, top in enumerate(t.bar_tops)
    )
    return f'  <g fill="{color}">\n{rects}\n  </g>'


def body_path(t: Tower) -> str:
    """塔身 + 顶部挑檐，一笔画成。挑出 14px 是碉楼最认得出的轮廓特征。"""
    return (
        f"M {t.body_l} {t.bot} "
        f"L {t.body_l} {t.top + t.eave_h} "
        f"L {t.body_l - t.eave} {t.top + t.eave_h} "
        f"L {t.body_l - t.eave} {t.top} "
        f"L {t.body_r + t.eave} {t.top} "
        f"L {t.body_r + t.eave} {t.top + t.eave_h} "
        f"L {t.body_r} {t.top + t.eave_h} "
        f"L {t.body_r} {t.bot} Z"
    )


def windows(t: Tower, cols: int = 2, gap: int = 56) -> str:
    """两列窄高窗，整体居中，两侧必须留白，否则窗会贴边被读成缺口。"""
    total = cols * t.win_w + (cols - 1) * gap
    left = (S - total) / 2
    out = []
    for r in t.win_rows:
        for c in range(cols):
            x = left + c * (t.win_w + gap)
            out.append(
                f'    <rect x="{x:.0f}" y="{r}" width="{t.win_w}" height="{t.win_h}"/>'
            )
    return "\n".join(out)


def door(t: Tower) -> str:
    cx = S / 2
    w = t.door_w
    r = w / 2
    return (
        f'    <path d="M {cx - r} {t.bot} V {t.door_arc} '
        f'A {r} {r} 0 0 1 {cx + r} {t.door_arc} V {t.bot} Z"/>'
    )


def tower_group(t: Tower, fill: str, openings: bool = True) -> str:
    """实心塔身；窗与门用 mask 真正镂空，任意底色上都成立。"""
    if not openings:
        return f'  <path d="{body_path(t)}" fill="{fill}"/>'
    cut = "\n".join([windows(t), door(t)])
    return (
        f'  <mask id="holes" maskUnits="userSpaceOnUse" x="0" y="0" width="{S}" height="{S}">\n'
        f'    <path d="{body_path(t)}" fill="#fff"/>\n'
        f'    <g fill="#000">\n{cut}\n    </g>\n'
        f"  </mask>\n"
        f'  <rect width="{S}" height="{S}" fill="{fill}" mask="url(#holes)"/>'
    )


def svg(body: str, w: int, h: int, label: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{label}">\n'
        f"  <title>{label}</title>\n  <desc>{desc}</desc>\n"
        f"{body}\n</svg>\n"
    )


# ── 1. 方形应用图标（深底）──────────────────────────────────
icon = (
    f'  <rect width="{S}" height="{S}" rx="116" fill="{INK}"/>\n'
    + bars(T, SEAL_LIGHT)
    + "\n"
    + tower_group(T, PAPER)
)
(BRAND / "logo-icon.svg").write_text(
    svg(icon, S, S, "侨乡江门 AI 智能讲解员 标志",
        "深青底方形图标：米色碉楼塔身，窗与拱门镂空透出底色，塔顶五根朱砂竖条既是碉楼柱廊也是讲解声波。"),
    encoding="utf-8",
)

# ── 2. 图形标（无底板，浅底用）──────────────────────────────
mark = bars(T, SEAL_MARK) + "\n" + tower_group(T, INK)
(BRAND / "logo-mark.svg").write_text(
    svg(mark, S, S, "侨乡江门 AI 智能讲解员 图形标",
        "无底板版本，用于宣纸米等浅色背景：深青碉楼 + 朱砂声波。"),
    encoding="utf-8",
)

# ── 3. 图形标（深底反白）───────────────────────────────────
mark_dark = bars(T, SEAL_LIGHT) + "\n" + tower_group(T, PAPER)
(BRAND / "logo-mark-dark.svg").write_text(
    svg(mark_dark, S, S, "侨乡江门 AI 智能讲解员 图形标（反白）",
        "无底板反白版本，用于碉楼青等深色背景：米白碉楼 + 朱砂声波。"),
    encoding="utf-8",
)

# ── 4. 小尺寸简化（16–32px）─────────────────────────────────
# 像素越少能保留的信息越少：去窗、去挑檐、三根条、塔身加宽、门洞放大。
SMALL = Tower(
    body_l=178, body_r=334, top=150, bot=450, eave=18, eave_h=20,
    bar_w=48, bar_gap=22, bar_tops=(104, 98, 104), bar_bot=156,
    win_rows=(), door_w=72, door_arc=376,
)
small = (
    f'  <rect width="{S}" height="{S}" rx="100" fill="{INK}"/>\n'
    + bars(SMALL, SEAL_LIGHT)
    + "\n"
    + tower_group(SMALL, PAPER, openings=False)
    + "\n"
    f'  <path d="M {S/2 - SMALL.door_w/2} {SMALL.bot} V {SMALL.door_arc} '
    f'A {SMALL.door_w/2} {SMALL.door_w/2} 0 0 1 {S/2 + SMALL.door_w/2} {SMALL.door_arc} '
    f'V {SMALL.bot} Z" fill="{INK}"/>'
)
(BRAND / "logo-icon-small.svg").write_text(
    svg(small, S, S, "侨乡江门 AI 智能讲解员 小尺寸图标",
        "16–32 像素专用简化版：三根声波条、无窗、放大拱门，保证小尺寸仍读作塔楼与开口。"),
    encoding="utf-8",
)

# ── 5. 印章标（朱底白文，物料/证书用）────────────────────────
# 窗与门必须保留镂空：白文印章靠「挖掉」成形，填实就退化成一根白条。
# 缩到 0.86 让四周留出足够朱边，实心白块才不会读成贴纸。
seal_k = 0.86
seal_shift = f"translate({S/2},{S/2}) scale({seal_k}) translate({-S/2},{-S/2})"
seal_tower = "\n".join(
    "  " + ln for ln in (bars(T, PAPER) + "\n" + tower_group(T, PAPER)).splitlines()
)
seal = (
    f'  <rect x="16" y="16" width="480" height="480" rx="28" fill="{SEAL}"/>\n'
    f'  <rect x="40" y="40" width="432" height="432" rx="12" fill="none" '
    f'stroke="{PAPER}" stroke-width="4" opacity="0.45"/>\n'
    f'  <g transform="{seal_shift}">\n{seal_tower}\n  </g>'
)
(BRAND / "logo-seal.svg").write_text(
    svg(seal, S, S, "侨乡江门 AI 智能讲解员 印章标",
        "朱砂方章白文效果，碉楼塔身与顶檐为阳文，用于文档压印、证书与物料角标。"),
    encoding="utf-8",
)

# ── 6. 横版主标 ─────────────────────────────────────────────
# 用「深底方块图标 + 字标」而不是裸图形：裸图形太窄，与 78px 宋体字站不住。
BOX, PAD = 200, 24
kk = BOX / S
lockup = (
    f'  <g transform="translate({PAD},{PAD}) scale({kk:.5f})">\n'
    f'    <rect width="{S}" height="{S}" rx="116" fill="{INK}"/>\n'
    + "\n".join("  " + ln for ln in (bars(T, SEAL_LIGHT) + "\n" + tower_group(T, PAPER)).splitlines())
    + "\n  </g>\n"
    f'  <text x="272" y="98" fill="{SEAL}" '
    f"font-family=\"'Source Han Sans SC','Microsoft YaHei',sans-serif\" "
    f'font-size="28" font-weight="500" letter-spacing="6">侨乡 · 江门</text>\n'
    f'  <text x="268" y="184" fill="{INK}">'
    f"<tspan font-family=\"'Segoe UI','Helvetica Neue',Arial,sans-serif\" "
    f'font-size="64" font-weight="700" letter-spacing="1">AI</tspan>'
    f"<tspan dx=\"28\" font-family=\"'Source Han Serif SC','Noto Serif CJK SC','SimSun',serif\" "
    f'font-size="78" font-weight="700" letter-spacing="4">智能讲解员</tspan>'
    f"</text>"
)
(BRAND / "logo-lockup.svg").write_text(
    svg(lockup, 820, 268, "侨乡 · 江门 AI 智能讲解员",
        "图形标 + 中文字标，用于封面、页眉、文档首屏。"),
    encoding="utf-8",
)

print("wrote 6 svg")
print("tower width", T.body_r - T.body_l, "height", T.bot - T.top)
