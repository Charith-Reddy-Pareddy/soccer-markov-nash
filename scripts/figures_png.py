"""Rasterize every SVG under docs/figures/ to a PNG in docs/figures/png/.

    python scripts/figures_png.py            # all of them, 3x scale
    python scripts/figures_png.py story      # just the ones whose name contains "story"

Slide decks and Google Docs cannot import SVG; this renders each figure through
headless Chrome (the same engine that builds docs/report.pdf) so the CSS
variables and web fonts resolve exactly as in the report.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "figures"
OUT = SRC / "png"
SCALE = 3
_VB = re.compile(r'viewBox="\s*([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)[ ,]+([-\d.]+)')


def _chrome() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(name):
            return name
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if pathlib.Path(mac).exists():
        return mac
    raise SystemExit("no chromium / Google Chrome found on PATH")


def render(svg_path: pathlib.Path, chrome: str, wrapdir: pathlib.Path) -> tuple[int, int]:
    svg = svg_path.read_text()
    m = _VB.search(svg)
    if not m:
        raise ValueError(f"{svg_path.name}: no viewBox")
    w = round(float(m.group(3)) * SCALE)
    h = round(float(m.group(4)) * SCALE)
    svg = re.sub(r"<svg\s", f'<svg width="{w}" height="{h}" ', svg, count=1)
    html = (
        "<!doctype html><meta charset=utf-8>"
        "<style>*{margin:0;padding:0}html,body{background:#fff}svg{display:block}</style>"
        + svg
    )
    wrap = wrapdir / (svg_path.stem + ".html")
    wrap.write_text(html)
    png = OUT / (svg_path.stem + ".png")
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
         "--default-background-color=ffffffff", "--force-device-scale-factor=1",
         f"--screenshot={png}", f"--window-size={w},{h}", f"file://{wrap}"],
        check=True, capture_output=True,
    )
    return w, h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("filter", nargs="?", default="", help="substring of the file name")
    args = ap.parse_args()

    chrome = _chrome()
    OUT.mkdir(parents=True, exist_ok=True)
    wrapdir = ROOT / ".png-wrap"
    wrapdir.mkdir(exist_ok=True)

    svgs = sorted(p for p in SRC.rglob("*.svg") if args.filter in p.name)
    if not svgs:
        raise SystemExit(f"no SVGs matching {args.filter!r} under {SRC}")

    for svg in svgs:
        try:
            w, h = render(svg, chrome, wrapdir)
            print(f"{svg.stem:24} {w:>5} x {h:<5} -> docs/figures/png/{svg.stem}.png")
        except Exception as exc:  # noqa: BLE001 -- report and continue
            print(f"{svg.stem:24} FAILED: {exc}", file=sys.stderr)

    shutil.rmtree(wrapdir, ignore_errors=True)


if __name__ == "__main__":
    main()
