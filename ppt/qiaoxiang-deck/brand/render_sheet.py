from pathlib import Path

from playwright.sync_api import sync_playwright

BRAND = Path(r"C:\Users\ghhjhjh\Desktop\AI-Tour-Guide\ppt\qiaoxiang-deck\brand")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=2)
    page.goto((BRAND / "sheet-v2.html").as_uri())
    page.wait_for_timeout(600)
    h = page.evaluate("document.body.scrollHeight")
    page.set_viewport_size({"width": 1600, "height": h})
    page.wait_for_timeout(300)
    page.screenshot(path=str(BRAND / "brand-sheet-v2.png"), clip={"x": 0, "y": 0, "width": 1600, "height": h})
    print("height", h)
    b.close()
