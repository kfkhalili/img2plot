"""Tests for img2plot.

Covers each pure transformation, the contained-mutation line extractor, and an
end-to-end CLI run on synthetic images whose edges are known by construction.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from img2plot import (
    Config,
    ExtractionConfig,
    Line,
    PreprocessConfig,
    _smoothed_neighbour_value,
    bilinear_interpolate,
    edge_probability,
    extract_lines,
    grow_line,
    img_to_lines,
    load_image,
    main,
    normalize_to_unit,
    parse_args,
    preprocess,
    to_grayscale,
    write_svg,
)

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


# --------------------------------------------------------------------------- #
# Config / Line are immutable value types
# --------------------------------------------------------------------------- #


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.preprocess = PreprocessConfig()  # type: ignore[misc]


def test_preprocess_config_is_frozen():
    cfg = PreprocessConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.clahe_kernel_size = 1  # type: ignore[misc]


def test_extraction_config_is_frozen():
    cfg = ExtractionConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.min_line_length = 1  # type: ignore[misc]


def test_line_is_frozen():
    line = Line(0, 0, 1, 1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        line.x1 = 5  # type: ignore[misc]


def test_config_defaults_match_legacy_script():
    cfg = Config()
    # Extraction defaults
    assert cfg.extract.termination_ratio == pytest.approx(1.0 / 3.5)
    assert cfg.extract.line_continue_thresh == pytest.approx(0.01)
    assert cfg.extract.min_line_length == 21
    assert cfg.extract.max_curve_angle_deg == pytest.approx(20.0)
    assert cfg.extract.lpf_atk == pytest.approx(0.05)
    # Preprocess defaults (None disables the step)
    assert cfg.preprocess.clahe_kernel_size == 32
    assert cfg.preprocess.blur_kernel_size == 1


# --------------------------------------------------------------------------- #
# to_grayscale
# --------------------------------------------------------------------------- #


def test_to_grayscale_uses_bt601_luma_weights():
    pixel = np.array([[[100, 50, 25]]], dtype=np.uint8)
    gray = to_grayscale(pixel)
    expected = 0.299 * 100 + 0.587 * 50 + 0.114 * 25
    assert gray.shape == (1, 1)
    assert gray.dtype == np.float64
    assert gray[0, 0] == pytest.approx(expected)


def test_to_grayscale_passes_through_2d_input():
    gray_in = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    gray_out = to_grayscale(gray_in)
    np.testing.assert_allclose(gray_out, gray_in.astype(np.float64))


def test_to_grayscale_drops_alpha_channel():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255  # alpha must not contribute
    gray = to_grayscale(rgba)
    np.testing.assert_allclose(gray, np.full((2, 2), 0.299 * 200))


def test_to_grayscale_rejects_unsupported_shape():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 3, 1)))


def test_to_grayscale_does_not_mutate_input():
    rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
    before = rgb.copy()
    _ = to_grayscale(rgb)
    np.testing.assert_array_equal(rgb, before)


def test_to_grayscale_handles_single_channel_3d():
    """A (H, W, 1) image should be treated as already grayscale."""
    img = np.array([[10], [20], [30]], dtype=np.uint8).reshape(3, 1, 1)
    gray = to_grayscale(img)
    assert gray.shape == (3, 1)
    np.testing.assert_allclose(gray, [[10.0], [20.0], [30.0]])


def test_to_grayscale_handles_gray_plus_alpha():
    """A (H, W, 2) image should drop the alpha and keep the gray channel."""
    img = np.zeros((2, 2, 2), dtype=np.uint8)
    img[..., 0] = 100  # gray
    img[..., 1] = 255  # alpha (must not contribute)
    gray = to_grayscale(img)
    np.testing.assert_allclose(gray, np.full((2, 2), 100.0))


# --------------------------------------------------------------------------- #
# normalize_to_unit
# --------------------------------------------------------------------------- #


def test_normalize_maps_min_to_zero_and_max_to_one():
    arr = np.array([[-2.0, 0.0], [1.0, 3.0]])
    out = normalize_to_unit(arr)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert out[0, 0] == 0.0
    assert out[1, 1] == 1.0


def test_normalize_is_linear():
    arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * 10.0
    out = normalize_to_unit(arr)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_normalize_constant_image_returns_zeros():
    arr = np.full((3, 3), 7.0)
    out = normalize_to_unit(arr)
    np.testing.assert_array_equal(out, np.zeros((3, 3)))


def test_normalize_does_not_mutate_input():
    arr = np.array([[1.0, 2.0]])
    before = arr.copy()
    _ = normalize_to_unit(arr)
    np.testing.assert_array_equal(arr, before)


# --------------------------------------------------------------------------- #
# bilinear_interpolate
# --------------------------------------------------------------------------- #


def test_bilinear_at_integer_coords_returns_pixel():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    for y in range(4):
        for x in range(4):
            assert bilinear_interpolate(img, x, y) == img[y, x]


def test_bilinear_center_is_average_of_corners():
    img = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert bilinear_interpolate(img, 0.5, 0.5) == pytest.approx(15.0)


def test_bilinear_horizontal_midpoint():
    img = np.array([[0.0, 10.0]])
    assert bilinear_interpolate(img, 0.5, 0.0) == pytest.approx(5.0)


def test_bilinear_quarter_along_x():
    img = np.array([[0.0, 100.0]])
    assert bilinear_interpolate(img, 0.25, 0.0) == pytest.approx(25.0)


def test_bilinear_clamps_negative_coords():
    img = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert bilinear_interpolate(img, -1.0, 0.0) == pytest.approx(5.0)
    assert bilinear_interpolate(img, 0.0, -1.0) == pytest.approx(5.0)


def test_bilinear_clamps_out_of_bounds_coords():
    img = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert bilinear_interpolate(img, 100.0, 100.0) == pytest.approx(8.0)


def test_bilinear_returns_python_float():
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(bilinear_interpolate(img, 0.5, 0.5), float)


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #


def test_preprocess_is_identity_when_all_steps_disabled():
    cfg = PreprocessConfig(clahe_kernel_size=None, blur_kernel_size=None)
    arr = np.full((4, 4), 0.5)
    out = preprocess(arr, cfg)
    np.testing.assert_array_equal(out, arr)


def test_preprocess_blur_only_smooths():
    cfg = PreprocessConfig(clahe_kernel_size=None, blur_kernel_size=2)
    arr = np.zeros((9, 9))
    arr[4, 4] = 1.0  # impulse
    out = preprocess(arr, cfg)
    # Smoothing should spread the impulse: original peak shrinks, neighbours rise.
    assert out[4, 4] < 1.0
    assert out[4, 5] > 0.0
    assert out[5, 4] > 0.0


def test_preprocess_does_not_mutate_input():
    cfg = PreprocessConfig()
    arr = np.full((16, 16), 0.5)
    before = arr.copy()
    _ = preprocess(arr, cfg)
    np.testing.assert_array_equal(arr, before)


# --------------------------------------------------------------------------- #
# edge_probability
# --------------------------------------------------------------------------- #


def test_edge_probability_uniform_image_is_zero():
    arr = np.full((8, 8), 0.5)
    mag = edge_probability(arr)
    np.testing.assert_allclose(mag, np.zeros_like(arr))


def test_edge_probability_is_a_pdf_for_non_uniform_image():
    arr = np.zeros((16, 16))
    arr[:, 8:] = 1.0
    mag = edge_probability(arr)
    assert mag.sum() == pytest.approx(1.0)
    assert (mag >= 0).all()


def test_edge_probability_concentrates_on_vertical_edge():
    arr = np.zeros((16, 16))
    arr[:, 8:] = 1.0
    mag = edge_probability(arr)
    peak_col = int(mag.argmax()) % mag.shape[1]
    # Sobel response straddles the boundary at cols 7 or 8.
    assert peak_col in (7, 8)


def test_edge_probability_concentrates_on_horizontal_edge():
    arr = np.zeros((16, 16))
    arr[8:, :] = 1.0
    mag = edge_probability(arr)
    peak_row = int(mag.argmax()) // mag.shape[1]
    assert peak_row in (7, 8)


def test_edge_probability_does_not_mutate_input():
    arr = np.zeros((16, 16))
    arr[:, 8:] = 1.0
    before = arr.copy()
    _ = edge_probability(arr)
    np.testing.assert_array_equal(arr, before)


def test_edge_probability_concentrates_on_diagonal_edge():
    """Edges that aren't axis-aligned must still produce a peak on the edge."""
    n = 32
    y_idx, x_idx = np.indices((n, n))
    arr = (y_idx > x_idx).astype(float)  # lower-triangle bright; diagonal is edge
    mag = edge_probability(arr)
    peak_idx = int(mag.argmax())
    peak_y, peak_x = divmod(peak_idx, n)
    assert abs(peak_x - peak_y) <= 2


# --------------------------------------------------------------------------- #
# Fixtures: synthetic vertical edge
#
# Two helpers because `extract_lines` takes the preprocessed image, while
# `grow_line` (called inside it) takes the derived edge field — tests of each
# pick the matching fixture.
# --------------------------------------------------------------------------- #


def _vertical_edge_image(width=64, height=64, edge_col=32) -> np.ndarray:
    """A smoothed vertical step edge — the kind of input ``extract_lines`` sees."""
    img = np.zeros((height, width))
    img[:, edge_col:] = 1.0
    return gaussian_filter(img, 1)


def _vertical_edge_field(width=64, height=64, edge_col=32):
    """The edge field (mag, gradx, grady) derived from the same image."""
    smooth = _vertical_edge_image(width, height, edge_col)
    mag = edge_probability(smooth)
    grady, gradx = np.gradient(smooth)
    return mag, gradx, grady


# --------------------------------------------------------------------------- #
# grow_line
# --------------------------------------------------------------------------- #


def test_grow_line_on_vertical_edge_produces_long_vertical_segment():
    mag, gx, gy = _vertical_edge_field()
    height = mag.shape[0]
    edge_col = int(mag[height // 2].argmax())
    line = grow_line(mag, gx, gy, edge_col, height // 2, ExtractionConfig())
    dy = abs(line.y2 - line.y1)
    dx = abs(line.x2 - line.x1)
    assert line.length > 10
    assert dy > dx, f"expected mostly-vertical line, got dx={dx}, dy={dy}"


def test_grow_line_on_horizontal_edge_produces_long_horizontal_segment():
    img = np.zeros((64, 64))
    img[32:, :] = 1.0
    smooth = gaussian_filter(img, 1)
    mag = edge_probability(smooth)
    grady, gradx = np.gradient(smooth)
    edge_row = int(np.argmax(mag[:, 32]))
    line = grow_line(mag, gradx, grady, 32, edge_row, ExtractionConfig())
    dy = abs(line.y2 - line.y1)
    dx = abs(line.x2 - line.x1)
    assert line.length > 10
    assert dx > dy, f"expected mostly-horizontal line, got dx={dx}, dy={dy}"


def test_grow_line_centers_on_seed_pixel():
    mag, gx, gy = _vertical_edge_field()
    height = mag.shape[0]
    edge_col = int(mag[height // 2].argmax())
    line = grow_line(mag, gx, gy, edge_col, height // 2, ExtractionConfig())
    # The seed must lie between the two endpoints.
    assert min(line.y1, line.y2) <= height // 2 <= max(line.y1, line.y2)


def test_grow_line_does_not_mutate_inputs():
    mag, gx, gy = _vertical_edge_field()
    mag_before = mag.copy()
    gx_before = gx.copy()
    gy_before = gy.copy()
    height = mag.shape[0]
    edge_col = int(mag[height // 2].argmax())
    _ = grow_line(mag, gx, gy, edge_col, height // 2, ExtractionConfig())
    np.testing.assert_array_equal(mag, mag_before)
    np.testing.assert_array_equal(gx, gx_before)
    np.testing.assert_array_equal(gy, gy_before)


def test_grow_line_seed_at_image_boundary_returns_unit_line():
    """The while-loop bounds check excludes the very first row/col, so a seed
    sitting at the boundary must return a length-1 line without crashing."""
    mag, gx, gy = _vertical_edge_field()
    line = grow_line(mag, gx, gy, 0, 0, ExtractionConfig())
    assert line.length == 1
    assert (line.x1, line.y1) == (0, 0)
    assert (line.x2, line.y2) == (0, 0)


def test_grow_line_with_zero_lpf_keeps_line_perfectly_straight():
    """LPF attack of 0 means mangle never updates, so the line follows the
    seed angle exactly. For a vertical edge that means a vertical line."""
    mag, gx, gy = _vertical_edge_field()
    height = mag.shape[0]
    edge_col = int(mag[height // 2].argmax())
    line = grow_line(
        mag, gx, gy, edge_col, height // 2, ExtractionConfig(lpf_atk=0.0)
    )
    # No drift -> x stays put.
    assert line.x1 == edge_col
    assert line.x2 == edge_col


def test_zero_max_curve_angle_truncates_lines_more_aggressively():
    """Lower ``max_curve_angle_deg`` should make the LPF drift trip the break
    sooner, producing shorter lines. We need a noisy edge to demonstrate this:
    on a perfectly clean edge, ``cangle`` equals ``seed_angle`` at every step
    and the drift check never fires regardless of tolerance."""
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64))
    img[:, 32:] = 1.0
    img = img + rng.normal(0, 0.1, img.shape)
    smooth = gaussian_filter(img, 1)
    mag = edge_probability(smooth)
    grady, gradx = np.gradient(smooth)
    edge_col = int(np.argmax(mag[32]))

    tight = grow_line(mag, gradx, grady, edge_col, 32,
                      ExtractionConfig(max_curve_angle_deg=0.0))
    loose = grow_line(mag, gradx, grady, edge_col, 32,
                      ExtractionConfig(max_curve_angle_deg=90.0))
    assert tight.length < loose.length


# --------------------------------------------------------------------------- #
# extract_lines
# --------------------------------------------------------------------------- #


def test_extract_lines_returns_lines_for_an_edge():
    img = _vertical_edge_image()
    lines = extract_lines(img, ExtractionConfig(min_line_length=5))
    assert len(lines) > 0
    for line in lines:
        assert isinstance(line, Line)
        assert line.length >= 5


def test_extract_lines_respects_min_line_length():
    img = _vertical_edge_image()
    lines = extract_lines(img, ExtractionConfig(min_line_length=8))
    assert all(line.length >= 8 for line in lines)


def test_extract_lines_does_not_mutate_preprocessed_input():
    img = _vertical_edge_image()
    snapshot = img.copy()
    extract_lines(img, ExtractionConfig(min_line_length=5))
    np.testing.assert_array_equal(img, snapshot)


def test_extract_lines_returns_empty_for_uniform_image():
    arr = np.full((32, 32), 0.5)
    lines = extract_lines(arr, ExtractionConfig())
    assert lines == []


def test_extracted_lines_lie_near_the_edge_column():
    img = _vertical_edge_image(edge_col=32)
    lines = extract_lines(img, ExtractionConfig(min_line_length=5))
    assert lines, "expected at least one line on a clean vertical edge"
    for line in lines:
        midpoint_col = (line.x1 + line.x2) / 2.0
        assert abs(midpoint_col - 32) <= 3, line


def test_extract_lines_terminates_when_peak_drops():
    """The peak should decay below threshold, so the loop terminates."""
    img = _vertical_edge_image()
    lines = extract_lines(img, ExtractionConfig(min_line_length=5))
    # If the loop somehow ran forever, we'd never get here. Pin a sanity bound.
    assert 0 < len(lines) < 10_000


def test_extract_lines_exercises_the_short_line_rejection_branch():
    """Force every candidate line to be rejected as too short. The loop must
    still terminate via the smoothed-neighbour reduction path; if that path
    were broken (peak not reduced), the argmax would stay put forever."""
    img = _vertical_edge_image()
    lines = extract_lines(img, ExtractionConfig(min_line_length=10_000))
    assert lines == []


# --------------------------------------------------------------------------- #
# _smoothed_neighbour_value (private but load-bearing in the rejection branch)
# --------------------------------------------------------------------------- #


def test_smoothed_neighbour_value_interior_pixel_averages_4_neighbours():
    mag = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
    # Neighbours of (1, 1) are (0,1)=2, (2,1)=8, (1,0)=4, (1,2)=6 -> mean 5.0
    assert _smoothed_neighbour_value(mag, 1, 1) == pytest.approx(5.0)


def test_smoothed_neighbour_value_corner_pixel_averages_2_neighbours():
    mag = np.array([[1.0, 2.0],
                    [3.0, 4.0]])
    # Neighbours of (0, 0) are (1, 0)=3 and (0, 1)=2
    assert _smoothed_neighbour_value(mag, 0, 0) == pytest.approx(2.5)


def test_smoothed_neighbour_value_edge_pixel_averages_3_neighbours():
    mag = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
    # Neighbours of (0, 1) are (1,1)=5, (0,0)=1, (0,2)=3 -> mean 3.0
    assert _smoothed_neighbour_value(mag, 0, 1) == pytest.approx(3.0)


def test_smoothed_neighbour_value_returns_zero_for_1x1_image():
    """No neighbours: must not divide by zero and must lower the peak so the
    extract_lines loop can terminate."""
    mag = np.array([[5.0]])
    assert _smoothed_neighbour_value(mag, 0, 0) == 0.0


# --------------------------------------------------------------------------- #
# load_image
# --------------------------------------------------------------------------- #


def test_load_image_round_trips_a_png(tmp_path: Path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[1, 1] = [255, 0, 0]
    img[2, 2] = [0, 255, 0]
    png = tmp_path / "tiny.png"
    iio.imwrite(png, img)
    loaded = load_image(png)
    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, img)


def test_load_image_raises_for_missing_file(tmp_path: Path):
    with pytest.raises((FileNotFoundError, OSError)):
        load_image(tmp_path / "does_not_exist.png")


# --------------------------------------------------------------------------- #
# write_svg
# --------------------------------------------------------------------------- #


def test_write_svg_produces_one_line_element_per_line(tmp_path: Path):
    lines = [Line(0, 0, 10, 10, 11), Line(5, 5, 15, 5, 11), Line(7, 1, 7, 9, 9)]
    out = tmp_path / "out.svg"
    write_svg(lines, out, size=(20, 20))
    assert out.exists()
    tree = ET.parse(out)
    line_elts = tree.findall(".//svg:line", SVG_NS)
    assert len(line_elts) == 3
    assert line_elts[0].attrib["x1"] == "0"
    assert line_elts[0].attrib["y2"] == "10"
    assert line_elts[2].attrib["x1"] == "7"


def test_write_svg_with_empty_list_still_writes_valid_file(tmp_path: Path):
    out = tmp_path / "empty.svg"
    write_svg([], out, size=(10, 10))
    assert out.exists()
    tree = ET.parse(out)  # must parse without raising
    line_elts = tree.findall(".//svg:line", SVG_NS)
    assert line_elts == []


def test_write_svg_sets_viewbox_and_size_from_size_argument(tmp_path: Path):
    """Without viewBox + concrete dims, the SVG renders empty in browsers
    because the line coords land outside the default 300x150 viewport."""
    out = tmp_path / "sized.svg"
    write_svg([Line(0, 0, 100, 50, 2)], out, size=(200, 100))
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.attrib["viewBox"] == "0 0 200 100"
    assert root.attrib["width"] == "200px"
    assert root.attrib["height"] == "100px"


def test_write_svg_factors_stroke_onto_parent_group(tmp_path: Path):
    """Stroke attributes live on the parent <g>, not on every <line>, so
    dense files stay compact."""
    out = tmp_path / "grouped.svg"
    write_svg([Line(0, 0, 1, 1, 2)], out, size=(10, 10))
    tree = ET.parse(out)
    g = tree.find(".//svg:g", SVG_NS)
    assert g is not None
    assert g.attrib.get("stroke") == "black"
    assert g.attrib.get("stroke-linecap") == "round"
    line = tree.find(".//svg:line", SVG_NS)
    assert "stroke" not in line.attrib


def test_main_svg_dimensions_match_input_image(tmp_path: Path):
    """The emitted SVG canvas should exactly match the input image dimensions."""
    png = tmp_path / "square.png"
    svg = tmp_path / "out.svg"
    _write_square_png(png, size=96, inset=24)
    main(["-i", str(png), "-o", str(svg)])
    tree = ET.parse(svg)
    root = tree.getroot()
    assert root.attrib["viewBox"] == "0 0 96 96"
    assert root.attrib["width"] == "96px"
    assert root.attrib["height"] == "96px"


def test_grow_line_endpoints_stay_within_image_bounds():
    """The inner walker can step past the image bound by 1-2 pixels before
    the while-condition kicks in. Endpoints must be clamped so they don't
    leak into the SVG and confuse downstream renderers/plotters."""
    rng = np.random.default_rng(7)
    img = np.zeros((32, 32))
    img[:, 16:] = 1.0
    img = img + rng.normal(0, 0.05, img.shape)
    smooth = gaussian_filter(img, 1)
    mag = edge_probability(smooth)
    grady, gradx = np.gradient(smooth)
    cfg = ExtractionConfig(min_line_length=1, max_curve_angle_deg=180.0)
    # Seed somewhere likely to walk past a boundary.
    line = grow_line(mag, gradx, grady, 16, 1, cfg)
    for coord, hi in [(line.x1, 32), (line.x2, 32), (line.y1, 32), (line.y2, 32)]:
        assert 0 <= coord < hi, line


# --------------------------------------------------------------------------- #
# CLI / end-to-end
# --------------------------------------------------------------------------- #


def _write_square_png(path: Path, size: int = 96, inset: int = 24) -> None:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[inset:size - inset, inset:size - inset] = 0
    iio.imwrite(path, img)


def test_parse_args_requires_input_and_output():
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--input", "a.png"])


def test_parse_args_accepts_short_and_long_forms():
    long_form = parse_args(["--input", "a.png", "--output", "b.svg"])
    short_form = parse_args(["-i", "a.png", "-o", "b.svg"])
    assert long_form.input == Path("a.png") == short_form.input
    assert long_form.output == Path("b.svg") == short_form.output


def test_main_writes_svg_for_square_input(tmp_path: Path, capsys):
    png = tmp_path / "square.png"
    svg = tmp_path / "out.svg"
    _write_square_png(png)
    rc = main(["--input", str(png), "--output", str(svg)])
    assert rc == 0
    assert svg.exists()
    tree = ET.parse(svg)
    line_elts = tree.findall(".//svg:line", SVG_NS)
    assert len(line_elts) > 0
    captured = capsys.readouterr()
    assert "img2plot:" in captured.out
    assert str(svg) in captured.out


def test_pipeline_line_endpoints_sit_near_square_edges(tmp_path: Path):
    """Every line endpoint should sit near one of the square's four edges.

    The algorithm is allowed to "cut corners" — once the LPF angle drifts past
    the threshold mid-line, the line kinks toward the next edge — but the two
    ends of a line should always anchor on an edge of the input shape, which
    is what actually marks pen-plotter output as following the shape.
    """
    size, inset = 96, 24
    png = tmp_path / "square.png"
    _write_square_png(png, size=size, inset=inset)
    image = np.asarray(iio.imread(png))
    lines = img_to_lines(image, Config())
    assert lines, "expected lines from a clean black-on-white square"

    edges = (inset, size - inset)
    margin = 5

    def near_any_edge(c: float) -> bool:
        return any(abs(c - e) <= margin for e in edges)

    for line in lines:
        for x, y in ((line.x1, line.y1), (line.x2, line.y2)):
            assert near_any_edge(x) or near_any_edge(y), (line, x, y)


def test_pipeline_lines_are_mostly_axis_aligned_on_a_stripe(tmp_path: Path):
    """A horizontal stripe has no corners, so every line should be horizontal."""
    size = 96
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[40:56, :] = 0
    png = tmp_path / "stripe.png"
    iio.imwrite(png, img)
    image = np.asarray(iio.imread(png))
    lines = img_to_lines(image, Config())
    assert lines, "expected lines from a horizontal stripe"
    for line in lines:
        dx = abs(line.x2 - line.x1)
        dy = abs(line.y2 - line.y1)
        assert dx > dy, line


def test_pipeline_is_deterministic(tmp_path: Path):
    """Running twice on the same input must produce identical line lists."""
    png = tmp_path / "square.png"
    _write_square_png(png)
    image = np.asarray(iio.imread(png))
    cfg = Config()
    lines_a = img_to_lines(image, cfg)
    lines_b = img_to_lines(image, cfg)
    assert lines_a == lines_b


def test_main_raises_for_missing_input(tmp_path: Path):
    """If the input file doesn't exist, ``main`` should error out and not
    leave a stray SVG behind."""
    missing = tmp_path / "nope.png"
    out = tmp_path / "out.svg"
    with pytest.raises((FileNotFoundError, OSError)):
        main(["-i", str(missing), "-o", str(out)])
    assert not out.exists()


def test_disabling_clahe_actually_changes_output():
    """Setting clahe_kernel_size=None must change the pipeline output, not
    silently no-op. Uses a noisy gradient so CLAHE has local histogram to
    equalize."""
    rng = np.random.default_rng(42)
    base = np.tile(np.arange(96, dtype=np.int16)[:, None], (1, 96))
    noise = rng.integers(0, 30, size=base.shape, dtype=np.int16)
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = np.stack([gray, gray, gray], axis=-1)
    cfg_with = Config(preprocess=PreprocessConfig(clahe_kernel_size=32))
    cfg_without = Config(preprocess=PreprocessConfig(clahe_kernel_size=None))
    lines_with = img_to_lines(image, cfg_with)
    lines_without = img_to_lines(image, cfg_without)
    assert lines_with != lines_without


def test_cli_runs_as_a_subprocess(tmp_path: Path):
    """Catches packaging-style regressions (shebang, sys.exit, top-level
    imports) that ``main([...])`` would silently skip past."""
    png = tmp_path / "square.png"
    svg = tmp_path / "out.svg"
    _write_square_png(png)
    script = Path(__file__).parent / "img2plot.py"
    result = subprocess.run(
        [sys.executable, str(script), "-i", str(png), "-o", str(svg)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert svg.exists()
    assert "img2plot:" in result.stdout
