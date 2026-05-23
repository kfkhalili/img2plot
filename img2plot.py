#!/usr/bin/env python3
"""img2plot - turn images into pen-plotter line drawings.

The pipeline is a sequence of pure transformations:

    image -> grayscale -> normalize -> preprocess -> extract lines -> SVG

Each transformation takes its input by value and returns a new array. The
line-extraction module owns the edge-probability and gradient derivation
internally, so its callers only have to hand over a preprocessed image.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndimage
import skimage.draw
import skimage.exposure
import svgwrite


# --------------------------------------------------------------------------- #
# Configuration / value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreprocessConfig:
    """Knobs that shape the image before edge extraction.

    ``None`` disables the step. Defaults reproduce the legacy script.
    """

    clahe_kernel_size: Optional[int] = 32
    blur_kernel_size: Optional[int] = 1


@dataclass(frozen=True)
class ExtractionConfig:
    """Knobs that shape how lines are pulled from the preprocessed image."""

    termination_ratio: float = 1.0 / 3.5
    line_continue_thresh: float = 0.01
    min_line_length: int = 21
    max_curve_angle_deg: float = 20.0
    lpf_atk: float = 0.05


@dataclass(frozen=True)
class Config:
    """Bundle of preprocessing + extraction knobs for the full pipeline."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    extract: ExtractionConfig = field(default_factory=ExtractionConfig)


@dataclass(frozen=True)
class Line:
    """An immutable line segment with its grown length in pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    length: int


# --------------------------------------------------------------------------- #
# Pure transformations
# --------------------------------------------------------------------------- #


def load_image(path: Path) -> np.ndarray:
    """Read an image from ``path`` and return it as a numpy array."""
    return np.asarray(iio.imread(str(path)))


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or already-grayscale image to a float gray array.

    Uses the ITU-R BT.601 luma weights (0.299, 0.587, 0.114). Alpha is
    discarded. ``(H, W, 1)`` (single-channel) and ``(H, W, 2)`` (gray + alpha)
    are treated as already-grayscale.
    """
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim != 3:
        raise ValueError(f"unsupported image shape {img.shape!r}")
    channels = img.shape[-1]
    if channels >= 3:
        return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    if channels >= 1:
        return img[..., 0].astype(np.float64)
    raise ValueError(f"unsupported channel count {channels}")


def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """Linearly scale ``img`` so that [min, max] maps to [0, 1]."""
    arr = np.asarray(img, dtype=np.float64)
    lo = arr.min()
    hi = arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def preprocess(img: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply CLAHE and/or Gaussian blur. A ``None`` kernel size disables the
    corresponding step. Returns a new array."""
    out = img
    if config.clahe_kernel_size is not None:
        out = skimage.exposure.equalize_adapthist(
            out, kernel_size=config.clahe_kernel_size
        )
    if config.blur_kernel_size is not None:
        out = ndimage.gaussian_filter(out, config.blur_kernel_size)
    return out


def edge_probability(img: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude weighted toward darker regions, as a PDF."""
    dx = ndimage.sobel(img, axis=0)
    dy = ndimage.sobel(img, axis=1)
    mag = np.hypot(dx, dy)
    blur = ndimage.gaussian_filter(img, 2)
    weighted = mag * (blur.max() - blur)
    total = weighted.sum()
    if total == 0:
        return weighted
    return weighted / total


def bilinear_interpolate(img: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample of ``img`` at fractional ``(x, y)``.

    Coordinates outside the image are clamped to the nearest valid pixel.
    """
    h, w = img.shape[:2]
    x_floor = max(0, min(int(math.floor(x)), w - 1))
    y_floor = max(0, min(int(math.floor(y)), h - 1))
    x_ceil = max(0, min(int(math.ceil(x)), w - 1))
    y_ceil = max(0, min(int(math.ceil(y)), h - 1))

    x_frac = x - math.floor(x)
    y_frac = y - math.floor(y)

    top_left = img[y_floor, x_floor]
    top_right = img[y_floor, x_ceil]
    bot_left = img[y_ceil, x_floor]
    bot_right = img[y_ceil, x_ceil]

    top = x_frac * top_right + (1 - x_frac) * top_left
    bot = x_frac * bot_right + (1 - x_frac) * bot_left
    return float(y_frac * bot + (1 - y_frac) * top)


# --------------------------------------------------------------------------- #
# Line extraction
# --------------------------------------------------------------------------- #


def grow_line(
    mag: np.ndarray,
    gradx: np.ndarray,
    grady: np.ndarray,
    px: int,
    py: int,
    config: ExtractionConfig,
) -> Line:
    """Grow a line outward from ``(px, py)`` along the local edge tangent.

    Endpoints are clamped to ``[0, width-1] x [0, height-1]`` — the inner
    walker can step one or two pixels past the image bound before the
    while-condition kicks in, and we don't want that leaking into the SVG.
    """
    seed_angle = math.atan2(grady[py, px], gradx[py, px])
    max_delta = config.max_curve_angle_deg * math.pi / 180.0
    threshold = config.line_continue_thresh * mag[py, px]
    height, width = mag.shape

    def grow(side_sign: int) -> Tuple[float, float, int]:
        """Walk outward; ``+1`` extends the "start" side, ``-1`` the "end" side."""
        x, y = float(px), float(py)
        mangle = seed_angle
        steps = 0
        while (
            0 < y < height - 1
            and 0 < x < width - 1
            and bilinear_interpolate(mag, x, y) > threshold
        ):
            steps += 1
            ix, iy = int(round(x)), int(round(y))
            cangle = math.atan2(grady[iy, ix], gradx[iy, ix])
            mangle = mangle * (1 - config.lpf_atk) + cangle * config.lpf_atk
            if abs(seed_angle - mangle) > max_delta:
                break
            x = px + side_sign * steps * math.sin(mangle)
            y = py - side_sign * steps * math.cos(mangle)
        return x, y, steps

    sx, sy, n_start = grow(+1)
    ex, ey, n_end = grow(-1)

    def clamp(v: float, hi: int) -> int:
        return max(0, min(int(round(v)), hi - 1))

    return Line(
        x1=clamp(sx, width),
        y1=clamp(sy, height),
        x2=clamp(ex, width),
        y2=clamp(ey, height),
        length=n_start + n_end + 1,
    )


def _smoothed_neighbour_value(mag: np.ndarray, py: int, px: int) -> float:
    """Mean of the 4-connected neighbours of ``(py, px)`` inside ``mag``."""
    h, w = mag.shape
    neighbours: List[float] = []
    if py + 1 < h:
        neighbours.append(float(mag[py + 1, px]))
    if px + 1 < w:
        neighbours.append(float(mag[py, px + 1]))
    if py - 1 >= 0:
        neighbours.append(float(mag[py - 1, px]))
    if px - 1 >= 0:
        neighbours.append(float(mag[py, px - 1]))
    if not neighbours:
        # 1x1 image: no neighbours to average. Drop the peak so the calling
        # loop's argmax stops returning this cell and termination triggers.
        return 0.0
    return sum(neighbours) / len(neighbours)


def extract_lines(
    preprocessed: np.ndarray,
    config: ExtractionConfig,
) -> List[Line]:
    """Pull lines from ``preprocessed`` until the magnitude peak drops below
    ``termination_ratio`` of its initial value.

    The edge-probability map and the directional gradients are derived from
    ``preprocessed`` inside this function — the caller does not need to know
    that mag comes from Sobel and the gradients from ``np.gradient`` of the
    *same* image. ``preprocessed`` is not mutated.
    """
    work = edge_probability(preprocessed)  # fresh array — safe to mutate
    grady, gradx = np.gradient(preprocessed)
    height, width = work.shape
    init_max = float(work.max())
    if init_max <= 0:
        return []
    termination = init_max * config.termination_ratio

    lines: List[Line] = []
    while float(work.max()) > termination:
        flat_idx = int(work.argmax())
        py, px = divmod(flat_idx, width)

        line = grow_line(work, gradx, grady, px, py, config)

        if line.length < config.min_line_length:
            # Replace this peak with the (smaller) mean of its 4-neighbours so
            # it stops winning argmax, but the surrounding field is preserved.
            work[py, px] = _smoothed_neighbour_value(work, py, px)
            continue

        lines.append(line)

        # Zero out the magnitude along the drawn line so the same edge isn't
        # picked again, then knock down the seed pixel itself.
        rr, cc, _ = skimage.draw.line_aa(line.y1, line.x1, line.y2, line.x2)
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        work[rr, cc] = 0
        work[py, px] = 0
    return lines


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def write_svg(
    lines: Sequence[Line],
    path: Path,
    size: Tuple[int, int],
) -> None:
    """Serialize ``lines`` to an SVG file at ``path``.

    ``size`` is ``(width, height)`` in pixels of the source image; it sets
    both the document's ``viewBox`` and concrete ``width``/``height`` so the
    file renders correctly in browsers and other SVG viewers (without these,
    coordinates land outside the default 300x150 viewport and the file looks
    empty).

    Stroke colour and ``stroke-linecap`` live on a parent ``<g>`` so they
    aren't repeated on every ``<line>`` — meaningful file-size win on dense
    images.
    """
    width, height = size
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
        profile="tiny",
    )
    group = dwg.add(dwg.g(stroke="black", stroke_linecap="round"))
    for line in lines:
        group.add(dwg.line((line.x1, line.y1), (line.x2, line.y2)))
    dwg.save()


def img_to_lines(image: np.ndarray, config: Config) -> List[Line]:
    """End-to-end pipeline from a raw image array to a list of lines."""
    gray = to_grayscale(image)
    normalized = normalize_to_unit(gray)
    preprocessed = preprocess(normalized, config.preprocess)
    return extract_lines(preprocessed, config.extract)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="img2plot",
        description="Convert an image to an SVG of pen-plotter line strokes.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True, type=Path,
        help="Path to the input image (PNG, JPG, ...).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True, type=Path,
        help="Path to the output SVG file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config()
    image = load_image(args.input)
    lines = img_to_lines(image, config)
    height, width = image.shape[:2]
    write_svg(lines, args.output, size=(width, height))
    print(f"img2plot: wrote {len(lines)} line(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
