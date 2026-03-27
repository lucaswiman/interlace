"""Take a screenshot of a deadlock diagram from the dining philosophers HTML report.

Generates the DPOR HTML report for the 3-philosopher dining problem, then uses
Playwright to open it, navigate to a deadlocked execution, and capture a
screenshot of the timeline showing the deadlock.

Usage::

    python scripts/screenshot_deadlock.py [output.png]

The script expects to be run from the repo root with the frontrun package
importable (e.g. via ``make screenshot``).
"""

from __future__ import annotations

import os
import sys
import tempfile

from examples.dpor_dining_philosophers import run_exploration


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "docs/_static/deadlock-diagram.png"

    # Generate report into a temp file
    report_fd, report_path = tempfile.mkstemp(suffix=".html", prefix="frontrun_report_")
    os.close(report_fd)
    try:
        run_exploration(report_path)
        _take_screenshot(report_path, output_path)
        print(f"Screenshot saved to {output_path}")
    finally:
        os.unlink(report_path)


def _take_screenshot(html_path: str, output_path: str) -> None:
    """Open the HTML report in a headless browser and screenshot the deadlock view."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        file_url = f"file://{os.path.abspath(html_path)}"
        page.goto(file_url)

        # Wait for the web components to render.
        # dpor-report creates dpor-nav/dpor-timeline inside its shadow DOM.
        page.wait_for_selector("dpor-report", state="attached")
        page.wait_for_timeout(2000)

        # Navigate to a deadlocked execution by clicking a fail button in dpor-nav
        # (which lives inside dpor-report's shadow DOM).
        page.evaluate("""() => {
            const report = document.querySelector('dpor-report');
            if (!report || !report.shadowRoot) return;
            const nav = report.shadowRoot.querySelector('dpor-nav');
            if (!nav || !nav.shadowRoot) return;
            const buttons = nav.shadowRoot.querySelectorAll('.exec-btn');
            for (const btn of buttons) {
                if (btn.classList.contains('fail')) {
                    btn.click();
                    return;
                }
            }
        }""")
        page.wait_for_timeout(1000)

        # Click the DEADLOCK badge on the timeline to open the detail modal.
        # The marker is an SVG <g> element so we use dispatchEvent instead of click().
        page.evaluate("""() => {
            const report = document.querySelector('dpor-report');
            if (!report || !report.shadowRoot) return;
            const timeline = report.shadowRoot.querySelector('dpor-timeline');
            if (!timeline || !timeline.shadowRoot) return;
            const marker = timeline.shadowRoot.querySelector('.deadlock-marker');
            if (marker) {
                marker.dispatchEvent(new MouseEvent('click', {bubbles: true}));
            }
        }""")
        page.wait_for_timeout(1000)

        # Take screenshot of the viewport
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        page.screenshot(path=output_path, full_page=False)

        browser.close()


if __name__ == "__main__":
    main()
