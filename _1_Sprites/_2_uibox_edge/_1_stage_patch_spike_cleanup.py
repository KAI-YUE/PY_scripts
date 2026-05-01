import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/_0_ui_box/test/"
name = "ui_pack"

SOURCE_DIR = os.path.join(root_dir, "./")
PATCH_DIR = os.path.join(root_dir, "./_0_manual_spike_patches/")
OUT_INTERMEDIATE_DIR = os.path.join(root_dir, f"./_1_{name}_patch_spike_removed/")
OUT_DEBUG_DIR = os.path.join(root_dir, f"./_1_{name}_patch_spike_removed_debug/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALPHA_THRESHOLD = 1
OVERWRITE = True

# cv2.TM_SQDIFF_NORMED: 0 is exact, larger is worse.
MATCH_MAX_SQDIFF = 0.005
MATCH_USE_ALPHA = True

# Patch names control the cleanup direction:
#   *_r.png -> scan rows for a horizontal box segment.
#   *_c.png -> scan columns for a vertical box segment.
LINE_NON_TRANSPARENT_ALPHA_THRESHOLD = 1
LINE_MEDIAN_SAMPLE_ALPHA_THRESHOLD = 150
LINE_MEAN_ALPHA_MAX_DELTA = 80
CLEAR_ALPHA_THRESHOLD = 1

DEBUG_OVERLAY_ENABLED = True
DEBUG_BOX_COLOR = (255, 64, 64, 255)


def _collect_images(source_dir: str, exclude_dirs: set[Path] | None = None):
	src = Path(source_dir)
	if not src.exists():
		raise FileNotFoundError(f"Directory not found: {source_dir}")

	exclude_dirs = exclude_dirs or set()
	paths = []
	for p in src.iterdir():
		if not p.is_file():
			continue
		if p.suffix.lower() not in INCLUDE_EXTS:
			continue
		if any(parent in exclude_dirs for parent in p.resolve().parents):
			continue
		paths.append(p)

	return sorted(paths, key=lambda p: p.name.lower())


def _collect_patches(patch_dir: str):
	patch_root = Path(patch_dir)
	patch_root.mkdir(parents=True, exist_ok=True)
	paths = [
		p for p in patch_root.iterdir()
		if p.is_file() and p.suffix.lower() in INCLUDE_EXTS
	]
	return sorted(paths, key=lambda p: p.name.lower())


def _match_array(img: Image.Image) -> np.ndarray:
	rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
	if MATCH_USE_ALPHA:
		return rgba
	return rgba[:, :, :3]


def _find_patch_location(source_img: Image.Image, patch_img: Image.Image):
	source = _match_array(source_img)
	patch = _match_array(patch_img)
	patch_h, patch_w = patch.shape[:2]
	src_h, src_w = source.shape[:2]

	if patch_w > src_w or patch_h > src_h:
		return None

	result = cv2.matchTemplate(source, patch, cv2.TM_SQDIFF_NORMED)
	min_val, _, min_loc, _ = cv2.minMaxLoc(result)
	if min_val > MATCH_MAX_SQDIFF:
		return None

	x, y = min_loc
	return {
		"x": int(x),
		"y": int(y),
		"w": int(patch_w),
		"h": int(patch_h),
		"sqdiff": float(min_val),
	}


def _patch_axis(patch_path: Path) -> str:
	suffix = patch_path.stem.lower().rsplit("_", 1)[-1]
	if suffix in {"r", "row", "h", "horizontal"}:
		return "row"
	if suffix in {"c", "col", "column", "v", "vertical"}:
		return "column"
	raise ValueError(
		f"Patch name must end with _r or _c to choose cleanup axis: {patch_path.name}"
	)


def _line_mean_alpha_spikes(patch_alpha: np.ndarray, axis: str) -> tuple[np.ndarray, float]:
	if axis == "row":
		line_means = patch_alpha.mean(axis=1)
		line_max = patch_alpha.max(axis=1)
	else:
		line_means = patch_alpha.mean(axis=0)
		line_max = patch_alpha.max(axis=0)

	usable = line_max >= LINE_MEDIAN_SAMPLE_ALPHA_THRESHOLD
	if not np.any(usable):
		usable = line_max >= LINE_NON_TRANSPARENT_ALPHA_THRESHOLD
	if not np.any(usable):
		return np.array([], dtype=np.int64), 0.0

	template_median = float(np.median(line_means[usable]))
	not_transparent = line_max >= LINE_NON_TRANSPARENT_ALPHA_THRESHOLD
	not_close = np.abs(line_means - template_median) > LINE_MEAN_ALPHA_MAX_DELTA
	return np.where(not_transparent & not_close)[0], template_median


def _cleanup_patch_region(img: Image.Image, patch_img: Image.Image, match: dict, patch_path: Path) -> tuple[Image.Image, int, str]:
	axis = _patch_axis(patch_path)
	source = np.array(img.convert("RGBA"), dtype=np.uint8)
	patch_alpha = np.array(patch_img.convert("RGBA"), dtype=np.uint8)[:, :, 3]
	x0 = match["x"]
	y0 = match["y"]
	x1 = x0 + match["w"]
	y1 = y0 + match["h"]
	removed = 0

	if axis == "row":
		spike_rows, _ = _line_mean_alpha_spikes(patch_alpha, axis)
		for row in spike_rows:
			yy = y0 + int(row)
			segment = source[yy, x0:x1]
			clear = segment[:, 3] >= CLEAR_ALPHA_THRESHOLD
			removed += int(np.count_nonzero(clear))
			segment[clear] = (0, 0, 0, 0)
	else:
		spike_cols, _ = _line_mean_alpha_spikes(patch_alpha, axis)
		for col in spike_cols:
			xx = x0 + int(col)
			segment = source[y0:y1, xx]
			clear = segment[:, 3] >= CLEAR_ALPHA_THRESHOLD
			removed += int(np.count_nonzero(clear))
			segment[clear] = (0, 0, 0, 0)

	return Image.fromarray(source, "RGBA"), removed, axis


def _draw_debug_overlay(source_img: Image.Image, matches: list[tuple[Path, dict, int, str]]) -> Image.Image:
	overlay = source_img.convert("RGBA").copy()
	draw = ImageDraw.Draw(overlay)
	for patch_path, match, removed, axis in matches:
		x0 = match["x"]
		y0 = match["y"]
		x1 = x0 + match["w"] - 1
		y1 = y0 + match["h"] - 1
		draw.rectangle((x0, y0, x1, y1), outline=DEBUG_BOX_COLOR, width=2)
		draw.text((x0, max(0, y0 - 12)), f"{patch_path.stem} {axis} -{removed}", fill=DEBUG_BOX_COLOR)
	return overlay


def process_image(path: Path, patch_paths: list[Path], out_dir: Path, debug_dir: Path):
	source_img = Image.open(path).convert("RGBA")
	processed = source_img
	matches = []
	missing = []

	for patch_path in patch_paths:
		patch_img = Image.open(patch_path).convert("RGBA")
		match = _find_patch_location(source_img, patch_img)
		if match is None:
			missing.append(patch_path.name)
			continue

		try:
			processed, removed, axis = _cleanup_patch_region(processed, patch_img, match, patch_path)
		except ValueError as exc:
			missing.append(f"{patch_path.name} ({exc})")
			continue
		matches.append((patch_path, match, removed, axis))

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / path.name
	if out_path.exists() and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {out_path}")
	processed.save(out_path, "PNG")

	debug_path = None
	if DEBUG_OVERLAY_ENABLED:
		debug_dir.mkdir(parents=True, exist_ok=True)
		debug_path = debug_dir / path.name
		_draw_debug_overlay(source_img, matches).save(debug_path, "PNG")

	return out_path, debug_path, matches, missing


def main():
	patch_paths = _collect_patches(PATCH_DIR)
	if not patch_paths:
		print(f"No manual patches found. Put patch PNGs here: {PATCH_DIR}")
		return

	paths = _collect_images(SOURCE_DIR)
	if not paths:
		raise RuntimeError(f"No source images found in {SOURCE_DIR}")

	out_dir = Path(OUT_INTERMEDIATE_DIR)
	debug_dir = Path(OUT_DEBUG_DIR)
	for path in paths:
		out_path, debug_path, matches, missing = process_image(path, patch_paths, out_dir, debug_dir)
		match_summary = ", ".join(
			f"{patch.name}@({match['x']},{match['y']}) {axis} removed={removed} sqdiff={match['sqdiff']:.5f}"
			for patch, match, removed, axis in matches
		)
		print(f"{path.name}: cleaned {len(matches)}/{len(patch_paths)} patch(es) -> {out_path}")
		if match_summary:
			print(f"  matches: {match_summary}")
		if missing:
			print(f"  missing: {', '.join(missing)}")
		if debug_path is not None:
			print(f"  debug: {debug_path}")


if __name__ == "__main__":
	main()
