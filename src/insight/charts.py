"""Tiny dependency-free SVG charting.

The sandbox refused to install matplotlib, and a report that only renders on
one machine is not reproducible.  These helpers emit plain SVG strings, so the
figures are vector, open in a browser, render on GitHub, and need nothing but
the Python standard library.

Only what the three figures need is implemented: a linear y-scale, grouped
bars with error bars, a line with markers, and text labels.
"""

from __future__ import annotations

import html

FONT = (
    "font-family='Segoe UI, Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif'"
)
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#0f766e"
ACCENT_LIGHT = "#99d5d0"
NEUTRAL = "#b8c0cc"
DANGER = "#b45309"


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


def bar_chart(
    labels: list[str],
    values: list[float],
    ci: list[tuple[float, float]] | None = None,
    *,
    width: int = 860,
    height: int = 420,
    y_max: float | None = None,
    y_label: str = "",
    title: str = "",
    subtitle: str = "",
    bar_colors: list[str] | None = None,
    value_fmt: str = "{:.1%}",
) -> str:
    """Vertical bars; optional 95% CI whiskers."""
    pad_l, pad_r, pad_t, pad_b = 70, 24, 78, 96
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    y_max = y_max or max(values) * 1.18 or 1.0
    n = len(values)
    slot = plot_w / n
    bar_w = slot * 0.58

    def y(v: float) -> float:
        return pad_t + plot_h - (v / y_max) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title)}'>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
        f"<text x='{pad_l}' y='28' {FONT} font-size='17' font-weight='600' fill='{INK}'>{_esc(title)}</text>",
    ]
    if subtitle:
        parts.append(
            f"<text x='{pad_l}' y='50' {FONT} font-size='12.5' fill='{MUTED}'>{_esc(subtitle)}</text>"
        )
    # gridlines
    for i in range(6):
        gy = pad_t + plot_h * i / 5
        gv = y_max * (1 - i / 5)
        parts.append(
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - pad_r}' y2='{gy:.1f}' stroke='{GRID}' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_l - 10}' y='{gy + 4:.1f}' text-anchor='end' {FONT} font-size='11.5' fill='{MUTED}'>"
            f"{_fmt(gv, 0) if y_max > 1.5 else _fmt(gv, 2)}</text>"
        )
    if y_label:
        parts.append(
            f"<text x='{(pad_l - 10) / 2:.0f}' y='{pad_t + plot_h / 2:.0f}' {FONT} font-size='12' fill='{MUTED}' "
            f"text-anchor='middle' transform='rotate(-90 {(pad_l - 10) / 2:.0f} {pad_t + plot_h / 2:.0f})'>"
            f"{_esc(y_label)}</text>"
        )
    # baseline
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - pad_r}' y2='{pad_t + plot_h}' "
        f"stroke='{INK}' stroke-width='1.2'/>"
    )
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = pad_l + slot * (i + 0.5)
        x = cx - bar_w / 2
        h = (v / y_max) * plot_h
        color = (bar_colors[i] if bar_colors else ACCENT)
        parts.append(
            f"<rect x='{x:.1f}' y='{y(v):.1f}' width='{bar_w:.1f}' height='{max(h, 0):.1f}' fill='{color}' rx='2'/>"
        )
        parts.append(
            f"<text x='{cx:.1f}' y='{y(v) - 8:.1f}' text-anchor='middle' {FONT} font-size='11.5' "
            f"font-weight='600' fill='{INK}'>{_esc(value_fmt.format(v))}</text>"
        )
        if ci:
            lo, hi = ci[i]
            if hi > lo:
                cw = bar_w * 0.32
                parts.append(
                    f"<line x1='{cx:.1f}' y1='{y(lo):.1f}' x2='{cx:.1f}' y2='{y(hi):.1f}' stroke='{INK}' stroke-width='1'/>"
                )
                parts.append(
                    f"<line x1='{cx - cw / 2:.1f}' y1='{y(lo):.1f}' x2='{cx + cw / 2:.1f}' y2='{y(lo):.1f}' stroke='{INK}' stroke-width='1'/>"
                )
                parts.append(
                    f"<line x1='{cx - cw / 2:.1f}' y1='{y(hi):.1f}' x2='{cx + cw / 2:.1f}' y2='{y(hi):.1f}' stroke='{INK}' stroke-width='1'/>"
                )
        label = lab if len(lab) <= 12 else lab[:11] + "…"
        parts.append(
            f"<text x='{cx:.1f}' y='{pad_t + plot_h + 20:.0f}' text-anchor='middle' {FONT} font-size='11' fill='{MUTED}'>{_esc(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart(
    xs: list[float],
    ys: list[float],
    *,
    width: int = 860,
    height: int = 430,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    subtitle: str = "",
    highlight_x: float | None = None,
    highlight_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    zero_line: bool = True,
    x_tick_step: float | None = None,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 78, 30, 78, 72
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    x_min = x_min if x_min is not None else min(xs)
    x_max = x_max if x_max is not None else max(xs)
    y_min = y_min if y_min is not None else min(min(ys), 0.0)
    y_max = y_max if y_max is not None else max(ys)
    if y_max == y_min:
        y_max = y_min + 1

    def px(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def py(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title)}'>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
        f"<text x='{pad_l}' y='28' {FONT} font-size='17' font-weight='600' fill='{INK}'>{_esc(title)}</text>",
    ]
    if subtitle:
        parts.append(
            f"<text x='{pad_l}' y='50' {FONT} font-size='12.5' fill='{MUTED}'>{_esc(subtitle)}</text>"
        )
    for i in range(6):
        gy = pad_t + plot_h * i / 5
        gv = y_min + (y_max - y_min) * (1 - i / 5)
        parts.append(
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - pad_r}' y2='{gy:.1f}' stroke='{GRID}' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_l - 10}' y='{gy + 4:.1f}' text-anchor='end' {FONT} font-size='11.5' fill='{MUTED}'>"
            f"{gv:,.0f}</text>"
        )
    if zero_line and y_min < 0:
        parts.append(
            f"<line x1='{pad_l}' y1='{py(0):.1f}' x2='{width - pad_r}' y2='{py(0):.1f}' stroke='{INK}' stroke-width='1'/>"
        )
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f"<polyline points='{pts}' fill='none' stroke='{ACCENT}' stroke-width='2.2'/>")
    if highlight_x is not None and x_min <= highlight_x <= x_max:
        hx = px(highlight_x)
        parts.append(
            f"<line x1='{hx:.1f}' y1='{pad_t}' x2='{hx:.1f}' y2='{pad_t + plot_h}' stroke='{DANGER}' "
            f"stroke-width='1.4' stroke-dasharray='5 4'/>"
        )
        hy = py(ys[min(range(len(xs)), key=lambda i: abs(xs[i] - highlight_x))])
        parts.append(f"<circle cx='{hx:.1f}' cy='{hy:.1f}' r='4.5' fill='{DANGER}'/>")
        if highlight_label:
            anchor = "start" if hx < width * 0.72 else "end"
            dx = 8 if anchor == "start" else -8
            parts.append(
                f"<text x='{hx + dx:.1f}' y='{pad_t + 16:.1f}' text-anchor='{anchor}' {FONT} "
                f"font-size='12' font-weight='600' fill='{DANGER}'>{_esc(highlight_label)}</text>"
            )
    # x ticks
    tick_step = x_tick_step or (x_max - x_min) / 6
    tick = x_min
    while tick <= x_max + 1e-9:
        tx = px(tick)
        parts.append(
            f"<line x1='{tx:.1f}' y1='{pad_t + plot_h:.1f}' x2='{tx:.1f}' y2='{pad_t + plot_h + 5:.1f}' stroke='{INK}'/>"
        )
        parts.append(
            f"<text x='{tx:.1f}' y='{pad_t + plot_h + 20:.0f}' text-anchor='middle' {FONT} font-size='11.5' fill='{MUTED}'>{tick:g}</text>"
        )
        tick += tick_step
    if x_label:
        parts.append(
            f"<text x='{pad_l + plot_w / 2:.0f}' y='{height - 12}' text-anchor='middle' {FONT} font-size='12' fill='{MUTED}'>{_esc(x_label)}</text>"
        )
    if y_label:
        parts.append(
            f"<text x='{20:.0f}' y='{pad_t + plot_h / 2:.0f}' {FONT} font-size='12' fill='{MUTED}' "
            f"text-anchor='middle' transform='rotate(-90 20 {pad_t + plot_h / 2:.0f})'>{_esc(y_label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def save(path: str, svg: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path
