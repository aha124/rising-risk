"""Render docs/og.png (1200x630) from the live hero.

Serves docs/ on a local port, opens the page in headless Chromium with the member
field already in the model ordering, and saves a screenshot. Needs Playwright:
    python3 -m pip install playwright && python3 -m playwright install chromium
Run from anywhere:
    python3 docs/tests/make_og.py
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1]
CSS = """
.docket, .masthead nav, .lede-2, .field-foot, #field-table-wrap, .synthetic-note { display: none !important; }
.hero { padding-top: 14px !important; }
.field { margin-top: 14px !important; }
.page { max-width: 1120px !important; }
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main() -> None:
    handler = functools.partial(QuietHandler, directory=str(DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/?order=model"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1200, "height": 630},
                color_scheme="light",
                reduced_motion="reduce",
            )
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("html[data-ready]")
            page.add_style_tag(content=CSS)
            page.wait_for_timeout(400)
            page.screenshot(path=str(DOCS / "og.png"))
            fixture = page.evaluate("document.documentElement.dataset.fixture === 'true'")
            browser.close()
    finally:
        server.shutdown()
    print(
        f"wrote {DOCS / 'og.png'}"
        + (" (placeholder data, regenerate after the real export)" if fixture else "")
    )


if __name__ == "__main__":
    main()
