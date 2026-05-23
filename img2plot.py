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
from dataclasses import dataclass, field, replace
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

    ``None`` disables a step where applicable. Defaults reproduce the
    legacy script.
    """

    clahe_kernel_size: Optional[int] = 32
    # CLAHE clip-limit. ``None`` means auto-derive from the input's dynamic
    # range (high-DR images get a gentler clip; low-DR get a stronger one).
    # Set to a float to pin the value.
    clahe_clip_limit: Optional[float] = None
    blur_kernel_size: Optional[int] = 1


@dataclass(frozen=True)
class ExtractionConfig:
    """Knobs that shape how lines are pulled from the preprocessed image."""

    termination_ratio: float = 1.0 / 3.5
    line_continue_thresh: float = 0.01
    min_line_length: int = 21
    max_curve_angle_deg: float = 20.0
    lpf_atk: float = 0.05
    # Defensive cap on the argmax/grow/reject loop. When ``min_line_length`` is
    # high relative to what the magnitude field actually supports, most
    # candidates are rejected; each rejection only lowers a single peak, so
    # neighbouring peaks in the same cluster keep getting picked and rejected.
    # The loop terminates eventually (the field does drain) but can take
    # tens of minutes on large images. The cap bounds wall time at the cost
    # of potentially-incomplete output, which the caller is warned about.
    max_iterations: int = 200_000


@dataclass(frozen=True)
class Config:
    """Bundle of preprocessing + extraction knobs for the full pipeline."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    extract: ExtractionConfig = field(default_factory=ExtractionConfig)
    # Auto-detect a uniform background (uniform corners) and exclude it from
    # line extraction. Big quality win on portraits and product shots; no-op
    # on busy scenes where corners disagree.
    auto_mask_background: bool = True


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
    corresponding step. Returns a new array.

    When ``clahe_clip_limit`` is ``None``, the clip is auto-derived from the
    input's dynamic range (5th-to-95th percentile band): wide-DR images get
    a gentler clip (~0.005), narrow-DR get a stronger one (~0.05). Set the
    field explicitly to pin the value.
    """
    out = img
    if config.clahe_kernel_size is not None:
        clip = config.clahe_clip_limit
        if clip is None:
            p5, p95 = np.percentile(img, [5, 95])
            dynamic_range = float(p95 - p5)
            # Linear map: DR=1 → clip=0.005, DR=0 → clip=0.05
            clip = 0.05 - 0.045 * max(0.0, min(dynamic_range, 1.0))
        out = skimage.exposure.equalize_adapthist(
            out, kernel_size=config.clahe_kernel_size, clip_limit=clip
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


def detect_uniform_background(
    image: np.ndarray,
    similarity_threshold: float = 0.08,
    min_coverage: float = 0.15,
    max_coverage: float = 0.95,
    corner_fraction: float = 0.02,
) -> Optional[np.ndarray]:
    """Detect a uniform background by sampling image corners.

    Returns a boolean mask where ``True`` marks background pixels — caller
    excludes those from edge extraction. Returns ``None`` when no
    consistent background is detected:

    - corners disagree on luminance (``similarity_threshold``);
    - the candidate background covers less than ``min_coverage`` of the
      image (probably not actually a background);
    - or covers more than ``max_coverage`` (subject is the same brightness
      as the background, so masking would erase the subject too).

    Works on luminance, so coloured subjects on coloured backgrounds work
    fine as long as the four corners agree on brightness.
    """
    h, w = image.shape[:2]
    if image.ndim == 3:
        gray = image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    else:
        gray = image.astype(np.float64)
    gray = gray / 255.0

    corner_size = max(5, int(min(h, w) * corner_fraction))
    corner_means = [
        float(gray[:corner_size, :corner_size].mean()),
        float(gray[:corner_size, -corner_size:].mean()),
        float(gray[-corner_size:, :corner_size].mean()),
        float(gray[-corner_size:, -corner_size:].mean()),
    ]
    # Require 3-of-4 corner agreement (not all 4): portraits often have the
    # subject bleed into one corner (sweater, hair). Look for a cluster of 3
    # corners with spread <= threshold; reject if no such cluster exists.
    sorted_means = sorted(corner_means)
    if sorted_means[2] - sorted_means[0] <= similarity_threshold:
        bg_corners = sorted_means[:3]
    elif sorted_means[3] - sorted_means[1] <= similarity_threshold:
        bg_corners = sorted_means[1:]
    else:
        return None

    bg_luminance = sum(bg_corners) / 3
    mask = np.abs(gray - bg_luminance) < similarity_threshold

    coverage = float(mask.mean())
    if coverage < min_coverage or coverage > max_coverage:
        return None
    return mask


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
    *,
    skip_mask: Optional[np.ndarray] = None,
) -> List[Line]:
    """Pull lines from ``preprocessed`` until the magnitude peak drops below
    ``termination_ratio`` of its initial value.

    The edge-probability map and the directional gradients are derived from
    ``preprocessed`` inside this function — the caller does not need to know
    that mag comes from Sobel and the gradients from ``np.gradient`` of the
    *same* image. ``preprocessed`` is not mutated.

    ``skip_mask`` (boolean, same shape as ``preprocessed``) marks pixels to
    exclude from line extraction. Used to drop a detected background so
    lines concentrate on the subject.
    """
    work = edge_probability(preprocessed)  # fresh array — safe to mutate
    if skip_mask is not None:
        work[skip_mask] = 0
    grady, gradx = np.gradient(preprocessed)
    height, width = work.shape
    init_max = float(work.max())
    if init_max <= 0:
        return []
    termination = init_max * config.termination_ratio

    lines: List[Line] = []
    iterations = 0
    while float(work.max()) > termination:
        if iterations >= config.max_iterations:
            print(
                f"img2plot: hit max_iterations={config.max_iterations} after "
                f"{len(lines)} lines; returning partial result. Consider "
                f"raising termination_ratio or lowering min_line_length.",
                file=sys.stderr,
            )
            break
        iterations += 1

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


def auto_scale_config(
    image: np.ndarray,
    base: Optional[Config] = None,
    reference_dim: int = 700,
    scale_min: float = 0.5,
    scale_max: float = 8.0,
) -> Config:
    """Return a Config with preprocessing kernels rescaled to image size.

    The legacy preprocessing defaults (``clahe_kernel_size=32``,
    ``blur_kernel_size=1``) were tuned by the upstream author for ~700px
    images. On larger inputs the CLAHE kernel covers proportionally less of
    the frame and the Gaussian blur becomes near-useless, leaving fine
    textures un-smoothed and the magnitude field full of scribble-inducing
    peaks. Scaling those two fields by ``min(width, height) / reference_dim``
    restores the intended relative behavior on any input size.

    ``min_line_length`` is *deliberately not scaled*. Raising it triggers
    the algorithm's rejection branch, where each rejected peak is lowered
    by averaging its 4 neighbours. The argmax loop is O(width * height) per
    iteration, so on large images a high rejection rate produces wall times
    measured in tens of minutes. Smarter pixel-length scaling can be added
    later (and gated by an iteration cap), but for now this is left to
    user tuning.

    Dimensionless fields (``termination_ratio``, ``line_continue_thresh``,
    angles, the LPF attack, ``max_iterations``) are never scaled. ``None``
    kernel sizes are passed through unchanged. ``scale_min`` / ``scale_max``
    clamp pathological inputs (32px thumbnail, 16k panorama).
    """
    if base is None:
        base = Config()
    h, w = image.shape[:2]
    raw = min(w, h) / reference_dim
    scale = max(scale_min, min(raw, scale_max))

    def scale_int(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return max(1, int(round(value * scale)))

    return Config(
        preprocess=PreprocessConfig(
            clahe_kernel_size=scale_int(base.preprocess.clahe_kernel_size),
            blur_kernel_size=scale_int(base.preprocess.blur_kernel_size),
        ),
        extract=base.extract,
    )


def img_to_lines(image: np.ndarray, config: Config) -> List[Line]:
    """End-to-end pipeline from a raw image array to a list of lines.

    If ``config.auto_mask_background`` is set (the default), a uniform
    background is auto-detected from ``image`` and excluded from line
    extraction. Detection runs on the *original* image, not the
    CLAHE/blur-preprocessed version, so the background-vs-subject
    distinction isn't smeared by local contrast equalization.
    """
    gray = to_grayscale(image)
    normalized = normalize_to_unit(gray)
    preprocessed = preprocess(normalized, config.preprocess)
    skip_mask = detect_uniform_background(image) if config.auto_mask_background else None
    return extract_lines(preprocessed, config.extract, skip_mask=skip_mask)


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
    image = load_image(args.input)
    config = auto_scale_config(image)
    lines = img_to_lines(image, config)
    height, width = image.shape[:2]
    write_svg(lines, args.output, size=(width, height))
    print(f"img2plot: wrote {len(lines)} line(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
