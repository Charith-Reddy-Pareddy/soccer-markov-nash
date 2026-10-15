#!/usr/bin/env python3
"""Render the PG assumptions HTML source as a compact, styled PDF."""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/pg_experiment_assumptions.html"
OUTPUT = ROOT / "docs/pg_experiment_assumptions.pdf"
GREEN = colors.HexColor("#23834b")
INK = colors.HexColor("#18251d")
MUTED = colors.HexColor("#64736a")
LINE = colors.HexColor("#d9e3dc")
WASH = colors.HexColor("#f3f7f4")
ORANGE = colors.HexColor("#a64c1a")


class TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "root", "attrs": {}, "children": []}
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = {"tag": tag, "attrs": dict(attrs), "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in {"br", "hr", "img", "meta", "link", "input"}:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx]["tag"] == tag:
                del self.stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1]["children"].append(data)


def inline(node: dict | str) -> str:
    if isinstance(node, str):
        return escape(node, quote=False)
    tag = node["tag"]
    body = "".join(inline(child) for child in node["children"])
    if tag in {"strong", "b"}:
        return f"<b>{body}</b>"
    if tag in {"em", "i"}:
        return f"<i>{body}</i>"
    if tag == "code":
        return f'<font name="Courier">{body}</font>'
    if tag == "a":
        href = escape(node["attrs"].get("href", ""), quote=True)
        return f'<link href="{href}" color="#23834b">{body}</link>'
    if tag == "br":
        return "<br/>"
    return body


def descendants(node: dict, tag: str) -> list[dict]:
    found = []
    for child in node["children"]:
        if isinstance(child, dict):
            if child["tag"] == tag:
                found.append(child)
            found.extend(descendants(child, tag))
    return found


def render() -> None:
    parser = TreeParser()
    parser.feed(SOURCE.read_text())
    body = descendants(parser.root, "body")[0]
    sample = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            textColor=INK,
            spaceAfter=3,
        ),
        "kicker": ParagraphStyle(
            "kicker",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=GREEN,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=25,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=16,
            textColor=GREEN,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "sub": ParagraphStyle(
            "sub",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=15,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "table",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.2,
            textColor=INK,
        ),
        "head": ParagraphStyle(
            "head",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.2,
            textColor=INK,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
    }
    story = []

    def add_table(node: dict) -> None:
        rows = []
        for row in descendants(node, "tr"):
            cells = []
            for cell in row["children"]:
                if isinstance(cell, dict) and cell["tag"] in {"th", "td"}:
                    style = styles["head"] if cell["tag"] == "th" else styles["table"]
                    cells.append(
                        Paragraph("".join(inline(item) for item in cell["children"]), style)
                    )
            if cells:
                rows.append(cells)
        if not rows:
            return
        widths = [1.15, 2.2, 3.95] if len(rows[0]) == 3 else [1.42, 0.68, 1.0, 1.42, 2.78]
        table = Table(
            rows, colWidths=[width * inch for width in widths], repeatRows=1, hAlign="LEFT"
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), WASH),
                    ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.extend([table, Spacer(1, 6)])

    def walk(node: dict) -> None:
        tag = node["tag"]
        classes = node["attrs"].get("class", "").split()
        if tag == "div" and "page" in classes:
            story.append(PageBreak())
        elif tag == "div" and "topline" in classes:
            bar = Table([[""]], colWidths=[7.34 * inch], rowHeights=[7])
            bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREEN)]))
            story.extend([bar, Spacer(1, 12)])
        elif tag == "div" and "kicker" in classes:
            story.append(
                Paragraph("".join(inline(item) for item in node["children"]), styles["kicker"])
            )
        elif tag == "div" and "footer" in classes:
            story.append(
                Paragraph("".join(inline(item) for item in node["children"]), styles["footer"])
            )
        elif tag in {"h1", "h2", "p"}:
            style_name = tag
            if tag == "p":
                style_name = (
                    "sub"
                    if "sub" in classes
                    else "small"
                    if "small" in classes
                    else "footer"
                    if "footer" in classes
                    else "body"
                )
            story.append(
                Paragraph("".join(inline(item) for item in node["children"]), styles[style_name])
            )
        elif tag == "div" and ("callout" in classes or "warn" in classes):
            content = "".join(inline(item) for item in node["children"])
            box = Table([[Paragraph(content, styles["body"])]], colWidths=[7.34 * inch])
            edge = ORANGE if "warn" in classes else GREEN
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), WASH),
                        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.extend([box, Spacer(1, 5)])
        elif tag == "table":
            add_table(node)
        elif tag == "ul":
            for item in descendants(node, "li"):
                story.append(
                    Paragraph(
                        "".join(inline(child) for child in item["children"]),
                        styles["body"],
                        bulletText="-",
                    )
                )
        elif tag == "div":
            for child in node["children"]:
                if isinstance(child, dict):
                    walk(child)

    for child in body["children"]:
        if isinstance(child, dict):
            walk(child)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.34 * inch,
        bottomMargin=0.34 * inch,
        title="Policy-gradient experiment assumptions",
        author="Charith Reddy Pareddy",
    )
    doc.build(story)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
