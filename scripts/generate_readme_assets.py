from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

INK = "#151614"
MUTED = "#687063"
MINT = "#419400"
PURPLE = "#8acb68"
AMBER = "#c4a900"
CORAL = "#70835f"
PANEL = "#fbfef6"
BACKGROUND = "#f8fdef"
WALL = "#9ca994"


def _write_svg(name: str, content: str) -> None:
    destination = ASSET_DIR / name
    destination.write_text(content.strip() + "\n", encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def generate_hero() -> None:
    _write_svg(
        "pocketworld-hero.svg",
        f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360" role="img" aria-labelledby="title desc">
  <title id="title">PocketWorld — an observable world model laboratory</title>
  <desc id="desc">A real simulator and model imagination panel connected by a learned dynamics bridge.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f8fdef"/><stop offset="1" stop-color="#edf3e7"/></linearGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0"><stop stop-color="{MINT}"/><stop offset="1" stop-color="{PURPLE}"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#d9e4d2" stroke-width="1"/></pattern>
  </defs>
  <rect width="1200" height="360" rx="26" fill="url(#bg)"/>
  <circle cx="1050" cy="-30" r="230" fill="#86c96a" opacity=".13"/><circle cx="100" cy="390" r="240" fill="#a7d890" opacity=".16"/>
  <g transform="translate(72 66)">
    <text fill="{MINT}" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700" letter-spacing="4">TINY WORLD · VISIBLE IMAGINATION</text>
    <text y="70" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="62" font-weight="700">PocketWorld</text>
    <text y="112" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="21">Learn 2D dynamics. Imagine future trajectories. Plan inside the model.</text>
    <g transform="translate(0 154)"><rect width="152" height="36" rx="18" fill="#eff6e9" stroke="#a9c79a"/><text x="76" y="23" text-anchor="middle" fill="{MINT}" font-family="Segoe UI,Arial,sans-serif" font-size="13" font-weight="700">64×64 RGB WORLD</text></g>
    <g transform="translate(166 154)"><rect width="146" height="36" rx="18" fill="#edf5e7" stroke="#b6d5a7"/><text x="73" y="23" text-anchor="middle" fill="#4e8b35" font-family="Segoe UI,Arial,sans-serif" font-size="13" font-weight="700">MODEL PLANNING</text></g>
  </g>
  <g transform="translate(760 48)">
    <rect width="162" height="250" rx="18" fill="{PANEL}" stroke="#a9c79a"/><rect x="14" y="42" width="134" height="160" rx="8" fill="url(#grid)"/>
    <text x="18" y="27" fill="{MINT}" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="700" letter-spacing="1.5">REAL</text>
    <rect x="78" y="68" width="12" height="92" rx="2" fill="{WALL}"/><circle cx="42" cy="152" r="7" fill="{MINT}" filter="url(#glow)"/><circle cx="126" cy="86" r="9" fill="none" stroke="{AMBER}" stroke-width="4"/>
    <path d="M42 152 C45 100 58 84 75 84 M94 84 C108 84 118 85 126 86" fill="none" stroke="{MINT}" stroke-width="3" stroke-linecap="round" stroke-dasharray="5 5"/>
    <text x="18" y="230" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">GROUND TRUTH</text>
  </g>
  <path d="M936 173h48" stroke="url(#line)" stroke-width="4" stroke-linecap="round"/><path d="m978 165 10 8-10 8" fill="none" stroke="{PURPLE}" stroke-width="3"/><text x="960" y="151" text-anchor="middle" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="10">LEARN</text>
  <g transform="translate(998 48)">
    <rect width="162" height="250" rx="18" fill="{PANEL}" stroke="#b6d5a7"/><rect x="14" y="42" width="134" height="160" rx="8" fill="url(#grid)"/>
    <text x="18" y="27" fill="{PURPLE}" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="700" letter-spacing="1.5">IMAGINE</text>
    <rect x="78" y="68" width="12" height="92" rx="2" fill="{WALL}"/><circle cx="44" cy="150" r="7" fill="{PURPLE}" filter="url(#glow)"/><circle cx="126" cy="86" r="9" fill="none" stroke="{AMBER}" stroke-width="4"/>
    <path d="M44 150 C49 105 61 88 76 86 M94 86 C108 88 118 88 126 86" fill="none" stroke="{PURPLE}" stroke-width="3" stroke-linecap="round" stroke-dasharray="5 5"/>
    <text x="18" y="230" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">LEARNED FUTURE</text>
  </g>
</svg>
""",
    )


def generate_architecture() -> None:
    boxes = (
        (42, 82, 158, "RGB frame", "3 × 64 × 64", MINT),
        (235, 82, 158, "CNN encoder", "latent zₜ", "#78b95b"),
        (428, 82, 158, "GRU dynamics", "zₜ + action", PURPLE),
        (621, 82, 158, "State model", "position + velocity", AMBER),
        (814, 82, 158, "RGB decoder", "future frame", CORAL),
        (1007, 82, 158, "Planner", "best actions", MINT),
    )
    box_markup = []
    arrows = []
    for index, (x, y, width, title, subtitle, color) in enumerate(boxes):
        box_markup.append(
            f'<g transform="translate({x} {y})"><rect width="{width}" height="92" rx="14" fill="#ffffff" stroke="{color}" stroke-opacity=".65"/>'
            f'<circle cx="22" cy="23" r="5" fill="{color}"/><text x="36" y="29" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="16" font-weight="700">{title}</text>'
            f'<text x="22" y="62" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{subtitle}</text></g>'
        )
        if index < len(boxes) - 1:
            start = x + width + 8
            end = boxes[index + 1][0] - 9
            arrows.append(f'<path d="M{start} 128H{end}" stroke="#aab8a5" stroke-width="2" marker-end="url(#arrow)"/>')
    _write_svg(
        "pocketworld-architecture.svg",
        f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="236" viewBox="0 0 1200 236" role="img" aria-labelledby="title desc">
  <title id="title">PocketWorld model and planning architecture</title><desc id="desc">RGB observations flow through an encoder, dynamics, structured state, decoder, and planner.</desc>
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#f8fdef"/><stop offset="1" stop-color="#eef4e8"/></linearGradient><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0L6 3 0 6" fill="#91a18b"/></marker></defs>
  <rect width="1200" height="236" rx="22" fill="url(#bg)"/><text x="42" y="43" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="20" font-weight="700">One compact loop: observe → learn → imagine → plan</text><text x="1158" y="43" text-anchor="end" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="12">DETERMINISTIC WORLD MODEL</text>
  {''.join(arrows)}{''.join(box_markup)}
  <path d="M1086 188 C1086 222 506 222 506 185" fill="none" stroke="{PURPLE}" stroke-opacity=".5" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)"/><text x="794" y="217" text-anchor="middle" fill="{PURPLE}" font-family="Segoe UI,Arial,sans-serif" font-size="11">roll out candidate action sequences inside the learned model</text>
</svg>
""",
    )


def generate_results() -> None:
    horizons = (1, 5, 10, 20)
    errors = (1.69, 2.32, 3.15, 3.94)
    points = []
    labels = []
    for index, (horizon, error) in enumerate(zip(horizons, errors)):
        x = 710 + index * 110
        y = 248 - error * 36
        points.append(f"{x},{y:.1f}")
        labels.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{MINT}"/><text x="{x}" y="276" text-anchor="middle" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">{horizon} step</text><text x="{x}" y="{y - 12:.1f}" text-anchor="middle" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="11">{error:.2f}px</text>')
    _write_svg(
        "pocketworld-results.svg",
        f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="320" viewBox="0 0 1200 320" role="img" aria-labelledby="title desc">
  <title id="title">PocketWorld headline evaluation results</title><desc id="desc">Planning success and composited agent position error across horizons.</desc>
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f8fdef"/><stop offset="1" stop-color="#eef4e8"/></linearGradient></defs><rect width="1200" height="320" rx="22" fill="url(#bg)"/>
  <text x="42" y="45" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="20" font-weight="700">What the tiny model can do — and where imagination breaks</text>
  <g transform="translate(42 82)"><rect width="176" height="150" rx="16" fill="#ffffff" stroke="#a9c79a"/><text x="20" y="32" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11" letter-spacing="1.2">16-STEP PLANNING</text><text x="20" y="86" fill="{MINT}" font-family="Segoe UI,Arial,sans-serif" font-size="42" font-weight="700">96%</text><text x="20" y="113" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="14">real success</text><text x="20" y="136" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">98% imagined · 2pp gap</text></g>
  <g transform="translate(236 82)"><rect width="176" height="150" rx="16" fill="#ffffff" stroke="#b6d5a7"/><text x="20" y="32" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11" letter-spacing="1.2">32-STEP PLANNING</text><text x="20" y="86" fill="#69a94e" font-family="Segoe UI,Arial,sans-serif" font-size="42" font-weight="700">7pp</text><text x="20" y="113" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="14">imagination gap</text><text x="20" y="136" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">100% imagined · 93% real</text></g>
  <g transform="translate(430 82)"><rect width="176" height="150" rx="16" fill="#ffffff" stroke="#c4cfba"/><text x="20" y="32" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11" letter-spacing="1.2">BARRIER TEST</text><text x="20" y="86" fill="{CORAL}" font-family="Segoe UI,Arial,sans-serif" font-size="42" font-weight="700">0%</text><text x="20" y="113" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="14">pure learned real</text><text x="20" y="136" fill="{MUTED}" font-family="Segoe UI,Arial,sans-serif" font-size="11">honest negative result</text></g>
  <g><text x="690" y="92" fill="{INK}" font-family="Segoe UI,Arial,sans-serif" font-size="15" font-weight="700">Composited agent position error</text><path d="M680 112V250H1058" fill="none" stroke="#aebba9" stroke-width="1"/><path d="M680 214H1058M680 178H1058M680 142H1058" stroke="#d8e3d2" stroke-width="1" stroke-dasharray="3 5"/><polyline points="{' '.join(points)}" fill="none" stroke="{MINT}" stroke-width="3" stroke-linejoin="round"/>{''.join(labels)}<text x="1078" y="155" fill="{MINT}" font-family="Segoe UI,Arial,sans-serif" font-size="12">100% coverage</text></g>
</svg>
""",
    )


def _interpolate(points: list[tuple[float, float]], progress: float) -> tuple[float, float]:
    distances = [math.dist(points[index], points[index + 1]) for index in range(len(points) - 1)]
    total = sum(distances)
    target = progress * total
    traversed = 0.0
    for index, distance in enumerate(distances):
        if traversed + distance >= target:
            ratio = (target - traversed) / max(distance, 1e-6)
            start, end = points[index], points[index + 1]
            return start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio
        traversed += distance
    return points[-1]


def _draw_world(draw: ImageDraw.ImageDraw, origin: tuple[int, int], accent: str, position: tuple[float, float], trail: list[tuple[float, float]], model: bool) -> None:
    ox, oy = origin
    scale = 4
    draw.rounded_rectangle((ox, oy, ox + 256, oy + 256), radius=10, fill="#fbfef6", outline="#b9c9b2", width=2)
    for coordinate in range(0, 65, 8):
        draw.line((ox + coordinate * scale, oy, ox + coordinate * scale, oy + 256), fill="#e1eadc", width=1)
        draw.line((ox, oy + coordinate * scale, ox + 256, oy + coordinate * scale), fill="#e1eadc", width=1)
    draw.rounded_rectangle((ox + 29 * scale, oy + 10 * scale, ox + 34 * scale, oy + 54 * scale), radius=2, fill=WALL)
    gx, gy = ox + 54 * scale, oy + 32 * scale
    draw.ellipse((gx - 15, gy - 15, gx + 15, gy + 15), outline=AMBER, width=5)
    if len(trail) > 1:
        trail_pixels = [(ox + x * scale, oy + y * scale) for x, y in trail]
        draw.line(trail_pixels, fill=accent, width=3, joint="curve")
    px, py = ox + position[0] * scale, oy + position[1] * scale
    draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=accent)
    draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#f8fdef")


def generate_demo_gif() -> None:
    width, height = 960, 420
    frames: list[Image.Image] = []
    real_path = [(10.0, 32.0), (18.0, 32.0), (22.0, 7.0), (38.0, 7.0), (43.0, 18.0), (54.0, 32.0)]
    model_path = [(10.0, 32.0), (18.0, 31.5), (22.0, 8.4), (38.0, 8.1), (43.8, 19.2), (53.4, 31.2)]
    steps = 30
    for frame_index in range(steps + 6):
        progress = min(frame_index, steps - 1) / (steps - 1)
        real_position = _interpolate(real_path, progress)
        model_position = _interpolate(model_path, progress)
        drift = math.dist(real_position, model_position)
        real_trail = [_interpolate(real_path, sample / 50) for sample in range(int(progress * 50) + 1)]
        model_trail = [_interpolate(model_path, sample / 50) for sample in range(int(progress * 50) + 1)]
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((20, 18, width - 20, height - 18), radius=24, fill="#ffffff", outline="#cbd7c6", width=2)
        draw.text((48, 37), "POCKETWORLD · LIVE ROLLOUT", fill=INK, font=_font(22, bold=True))
        draw.text((48, 70), "Same action sequence, two futures", fill=MUTED, font=_font(14))
        draw.text((56, 107), "REAL SIMULATOR", fill=MINT, font=_font(13, bold=True))
        draw.text((530, 107), "MODEL IMAGINATION", fill=PURPLE, font=_font(13, bold=True))
        _draw_world(draw, (56, 133), MINT, real_position, real_trail, model=False)
        _draw_world(draw, (530, 133), PURPLE, model_position, model_trail, model=True)
        draw.rounded_rectangle((834, 133, 916, 389), radius=12, fill="#f1f6ec", outline="#c3d1bd")
        draw.text((850, 151), "STEP", fill=MUTED, font=_font(10, bold=True))
        draw.text((850, 170), f"{min(frame_index, steps - 1):02d}", fill=INK, font=_font(25, bold=True))
        draw.text((850, 221), "DRIFT", fill=MUTED, font=_font(10, bold=True))
        draw.text((850, 240), f"{drift:.1f}", fill=PURPLE, font=_font(25, bold=True))
        draw.text((850, 270), "px", fill=MUTED, font=_font(11))
        draw.text((850, 323), "GOAL", fill=MUTED, font=_font(10, bold=True))
        goal_distance = math.dist(model_position, (54.0, 32.0))
        draw.text((850, 342), f"{goal_distance:.1f}", fill=AMBER, font=_font(22, bold=True))
        frames.append(image.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    frames[0].save(
        ASSET_DIR / "pocketworld-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=105,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_hero()
    generate_architecture()
    generate_results()
    generate_demo_gif()
    for path in sorted(ASSET_DIR.iterdir()):
        print(f"generated {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
