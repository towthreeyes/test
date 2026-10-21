"""Rasterize the report SVGs to PNG for visual inspection.

The sandbox blocks pip installs and headless-browser IPC, so matplotlib and
headless Edge are both out.  A PNG is just zlib-compressed scanlines; this
module re-draws the subset of SVG we emit (rect, line, text, polyline) with a
small built-in bitmap font.  It exists to eyeball the figures, not to be a
general converter.
"""

from __future__ import annotations

import re
import struct
import zlib
import xml.etree.ElementTree as ET

# 5x7 uppercase bitmap font for the ASCII subset we need.
_GLYPHS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    " ": ["00000"] * 7,
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
}


def _hex(color: str, default=(0, 0, 0)) -> tuple[int, int, int]:
    if not color or not color.startswith("#"):
        return default
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return default


class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.buf = bytearray(b"\xff" * (w * h * 3))

    def px(self, x: int, y: int, rgb: tuple[int, int, int]) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.buf[i : i + 3] = bytes(rgb)

    def rect(self, x: float, y: float, w: float, h: float, rgb) -> None:
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        for yy in range(max(0, y0), min(self.h, y1)):
            row = (yy * self.w) * 3
            for xx in range(max(0, x0), min(self.w, x1)):
                i = row + xx * 3
                self.buf[i : i + 3] = bytes(rgb)

    def line(self, x0: float, y0: float, x1: float, y1: float, rgb, width: int = 1) -> None:
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for s in range(steps + 1):
            t = s / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            for dx in range(width):
                for dy in range(width):
                    self.px(int(round(x)) + dx, int(round(y)) + dy, rgb)

    def text(self, x: float, y: float, s: str, rgb, scale: int = 2) -> None:
        """Draw ASCII with the 5x7 font; y is the baseline-ish anchor."""
        cx = int(x)
        cy = int(y) - 7 * scale
        for ch in s.upper():
            glyph = _GLYPHS.get(ch)
            if glyph is None:
                cx += 6 * scale
                continue
            for ry, row in enumerate(glyph):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + rx * scale, cy + ry * scale, scale, scale, rgb)
            cx += 6 * scale

    def save_png(self, path: str) -> None:
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)  # filter type 0
            raw += self.buf[y * self.w * 3 : (y + 1) * self.w * 3]

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        png += chunk(b"IEND", b"")
        with open(path, "wb") as fh:
            fh.write(png)


_NUM = r"-?\d+(?:\.\d+)?"


def _attrs(el: ET.Element) -> dict[str, str]:
    return {k: v for k, v in el.attrib.items()}


def render(svg_path: str, png_path: str) -> str:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    w = int(float(root.attrib.get("width", 800)))
    h = int(float(root.attrib.get("height", 450)))
    canvas = Canvas(w, h)
    white = (255, 255, 255)

    def walk(el: ET.Element) -> None:
        tag = el.tag.split("}")[-1]
        a = _attrs(el)
        if tag == "rect":
            fill = a.get("fill", "#ffffff")
            if fill != "none":
                canvas.rect(
                    float(a.get("x", 0)), float(a.get("y", 0)),
                    float(a.get("width", 0)), float(a.get("height", 0)),
                    _hex(fill, white),
                )
        elif tag == "line":
            canvas.line(
                float(a.get("x1", 0)), float(a.get("y1", 0)),
                float(a.get("x2", 0)), float(a.get("y2", 0)),
                _hex(a.get("stroke"), (0, 0, 0)),
                2 if float(a.get("stroke-width", 1)) >= 2 else 1,
            )
        elif tag == "circle":
            cx, cy, r = (float(a.get(k, 0)) for k in ("cx", "cy", "r"))
            col = _hex(a.get("fill"))
            for yy in range(int(cy - r), int(cy + r) + 1):
                for xx in range(int(cx - r), int(cx + r) + 1):
                    if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                        canvas.px(xx, yy, col)
        elif tag == "polyline":
            pts = [float(v) for v in re.findall(_NUM, a.get("points", ""))]
            col = _hex(a.get("stroke"))
            for i in range(0, len(pts) - 2, 2):
                canvas.line(pts[i], pts[i + 1], pts[i + 2], pts[i + 3], col, 2)
        elif tag == "text":
            content = "".join(el.itertext()).strip()
            if content:
                canvas.text(
                    float(a.get("x", 0)), float(a.get("y", 0)),
                    content, _hex(a.get("fill"), (30, 30, 30)),
                    2 if float(a.get("font-size", 12)) >= 13 else 1,
                )
        for child in el:
            walk(child)

    walk(root)
    canvas.save_png(png_path)
    return png_path


if __name__ == "__main__":
    import sys

    render(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
