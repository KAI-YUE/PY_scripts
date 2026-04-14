import math
import os
from collections import deque
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/test/"
name = "cards"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DIR = os.path.join(root_dir, f"./_0_{name}_band_cutout/")
MASK_DIR = os.path.join(root_dir, f"./_1_{name}_band_mask/")

# Reference block that contains the target painted band.
TEMPLATE_PATH = os.path.join(root_dir, "./temp.png")
# Full-size rough band estimate. Non-zero alpha marks the approximate band area.
TEMPLATE_SEARCH_PATH = os.path.join(root_dir, "./temp2.png")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
OVERWRITE = True

ALPHA_THRESHOLD = 1
TEMPLATE_USE_ALPHA_ONLY = True

# Search region from the visible outer edge inward, as a percentage
# of the sprite's short edge. Example: 2..15 means only inspect the
# band that lies roughly 2% to 15% away from the silhouette edge.
SEARCH_START_PERCENT = 5.0
SEARCH_END_PERCENT = 12.0
SEARCH_PERCENT_TOLERANCE = 0.5

# Detection settings
COLOR_DISTANCE_THRESHOLD = 32

def _collect_images(source_dir: str):
	src = Path(source_dir)
	if not src.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	template_paths = {
		Path(TEMPLATE_PATH).resolve(),
		Path(TEMPLATE_SEARCH_PATH).resolve(),
	}
	paths = []
	for p in src.iterdir():
		if not p.is_file():
			continue
		if p.suffix.lower() not in INCLUDE_EXTS:
			continue
		if p.resolve() in template_paths:
			continue
		paths.append(p)

	if not paths:
		raise RuntimeError(f"No images found in {source_dir} with extensions {sorted(INCLUDE_EXTS)}")

	return sorted(paths, key=lambda p: p.name.lower())


def _load_template_stats(path: str):
	template = Image.open(path).convert("RGBA")
	rgba = np.array(template, dtype=np.uint8)
	if TEMPLATE_USE_ALPHA_ONLY:
		samples = rgba[rgba[:, :, 3] >= ALPHA_THRESHOLD][:, :3]
	else:
		samples = rgba[:, :, :3].reshape(-1, 3)

	if samples.size == 0:
		raise RuntimeError(f"Template has no usable pixels: {path}")

	sample_count = int(samples.shape[0])
	median_rgb = tuple(int(round(median(samples[:, channel_index]))) for channel_index in range(3))
	mad_rgb = []
	for channel_index in range(3):
		channel_values = samples[:, channel_index]
		channel_median = median_rgb[channel_index]
		channel_abs_dev = np.abs(channel_values - channel_median)
		mad_rgb.append(int(round(median(channel_abs_dev))))

	return {
		"template_size": template.size,
		"sample_count": sample_count,
		"median_rgb": median_rgb,
		"mad_rgb": tuple(mad_rgb),
	}


def _load_search_template(path: str):
	template = Image.open(path).convert("RGBA")
	alpha = np.array(template.getchannel("A"), dtype=np.uint8)
	mask = alpha >= ALPHA_THRESHOLD
	if not mask.any():
		raise RuntimeError(f"Search template has no usable alpha pixels: {path}")
	return {
		"size": template.size,
		"mask": mask,
	}

def _alpha_bounds(alpha: np.ndarray):
	ys, xs = np.nonzero(alpha >= ALPHA_THRESHOLD)
	if xs.size == 0:
		return alpha.shape[1], alpha.shape[0], -1, -1
	return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _search_band_px(alpha: np.ndarray) -> tuple[int, int]:
	min_x, min_y, max_x, max_y = _alpha_bounds(alpha)
	if max_x < 0 or max_y < 0:
		return 0, 0

	content_w = max_x - min_x + 1
	content_h = max_y - min_y + 1
	short_edge = min(content_w, content_h)
	if short_edge <= 0:
		return 0, 0

	start_percent = max(0.0, min(100.0, SEARCH_START_PERCENT))
	end_percent = max(0.0, min(100.0, SEARCH_END_PERCENT))
	if end_percent < start_percent:
		start_percent, end_percent = end_percent, start_percent

	start_px = max(0, int(math.floor(short_edge * start_percent / 100.0)))
	end_px = max(start_px, int(math.ceil(short_edge * end_percent / 100.0)))
	return start_px, end_px


def _short_edge_from_alpha(alpha: np.ndarray) -> int:
	min_x, min_y, max_x, max_y = _alpha_bounds(alpha)
	if max_x < 0 or max_y < 0:
		return 0
	return min(max_x - min_x + 1, max_y - min_y + 1)


def _search_band_from_template(alpha: np.ndarray, edge_dist: np.ndarray, search_template: dict | None):
	default_start_px, default_end_px = _search_band_px(alpha)
	if search_template is None:
		start_percent = max(0.0, SEARCH_START_PERCENT - SEARCH_PERCENT_TOLERANCE)
		end_percent = min(100.0, SEARCH_END_PERCENT + SEARCH_PERCENT_TOLERANCE)
		return default_start_px, default_end_px, start_percent, end_percent

	template_mask = search_template["mask"]
	if template_mask.shape != alpha.shape:
		raise ValueError(
			f"TEMPLATE_SEARCH_PATH size {search_template['size']} does not match source image size {(alpha.shape[1], alpha.shape[0])}"
		)

	valid = template_mask & (alpha >= ALPHA_THRESHOLD) & (edge_dist >= 0)
	if not valid.any():
		start_percent = max(0.0, SEARCH_START_PERCENT - SEARCH_PERCENT_TOLERANCE)
		end_percent = min(100.0, SEARCH_END_PERCENT + SEARCH_PERCENT_TOLERANCE)
		return default_start_px, default_end_px, start_percent, end_percent

	dists = edge_dist[valid]
	start_px = int(dists.min())
	end_px = int(dists.max())
	if end_px < start_px:
		start_px, end_px = end_px, start_px

	short_edge = _short_edge_from_alpha(alpha)
	if short_edge <= 0:
		return start_px, end_px, 0.0, 0.0

	start_percent = max(0.0, (start_px / float(short_edge)) * 100.0 - SEARCH_PERCENT_TOLERANCE)
	end_percent = min(100.0, (end_px / float(short_edge)) * 100.0 + SEARCH_PERCENT_TOLERANCE)
	start_px = max(0, int(math.floor(short_edge * start_percent / 100.0)))
	end_px = max(start_px, int(math.ceil(short_edge * end_percent / 100.0)))
	return start_px, end_px, start_percent, end_percent


def _distance_from_visible_edge(alpha: np.ndarray):
	inside = alpha >= ALPHA_THRESHOLD
	h, w = inside.shape
	dist = np.full((h, w), -1, dtype=np.int32)
	queue = deque()

	neighbors = (
		(-1, -1), (0, -1), (1, -1),
		(-1, 0),           (1, 0),
		(-1, 1),  (0, 1),  (1, 1),
	)

	for y in range(h):
		for x in range(w):
			if not inside[y, x]:
				continue

			is_boundary = False
			for dx, dy in neighbors:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					is_boundary = True
					break
				if not inside[ny, nx]:
					is_boundary = True
					break

			if is_boundary:
				dist[y, x] = 0
				queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y, x]

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if not inside[ny, nx]:
				continue
			if dist[ny, nx] != -1:
				continue

			dist[ny, nx] = base_dist + 1
			queue.append((nx, ny))

	return dist

def _build_band_mask(img: Image.Image, stats: dict, search_template: dict | None = None):
	rgba = np.array(img, dtype=np.int16)
	alpha = rgba[:, :, 3]
	edge_dist = _distance_from_visible_edge(alpha)
	search_start_px, search_end_px, search_start_percent, search_end_percent = _search_band_from_template(
		alpha,
		edge_dist,
		search_template,
	)

	target_r, target_g, target_b = stats["median_rgb"]
	color_distance_threshold_sq = COLOR_DISTANCE_THRESHOLD * COLOR_DISTANCE_THRESHOLD

	search_mask = (
		(alpha >= ALPHA_THRESHOLD)
		& (edge_dist >= search_start_px)
		& (edge_dist <= search_end_px)
	)
	dr = rgba[:, :, 0] - target_r
	dg = rgba[:, :, 1] - target_g
	db = rgba[:, :, 2] - target_b
	dist_sq = dr * dr + dg * dg + db * db
	mask = search_mask & (dist_sq <= color_distance_threshold_sq)
	return (
		mask,
		int(mask.any()),
		int(mask.sum()),
		search_start_px,
		search_end_px,
		search_start_percent,
		search_end_percent,
	)


def _mask_to_image(mask) -> Image.Image:
	return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def _apply_zero_alpha_cutout(img: Image.Image, mask):
	out = np.array(img, dtype=np.uint8, copy=True)
	out[mask, 3] = 0
	return Image.fromarray(out, mode="RGBA"), int(mask.sum())


def process_image(path: Path, out_dir: Path, mask_dir: Path, stats: dict, search_template: dict | None = None):
	img = Image.open(path).convert("RGBA")
	(
		mask,
		kept_components,
		kept_pixels,
		search_start_px,
		search_end_px,
		search_start_percent,
		search_end_percent,
	) = _build_band_mask(img, stats, search_template=search_template)
	cutout, removed_pixels = _apply_zero_alpha_cutout(img, mask)

	out_path = out_dir / path.name
	mask_path = mask_dir / path.name

	if (out_path.exists() or mask_path.exists()) and not OVERWRITE:
		raise FileExistsError(f"Output already exists for {path.name}")

	cutout.save(out_path, "PNG")
	_mask_to_image(mask).save(mask_path, "PNG")

	return {
		"name": path.name,
		"size": img.size,
		"components": kept_components,
		"mask_pixels": kept_pixels,
		"removed_pixels": removed_pixels,
		"search_start_px": search_start_px,
		"search_end_px": search_end_px,
		"search_start_percent": search_start_percent,
		"search_end_percent": search_end_percent,
		"out_path": out_path,
		"mask_path": mask_path,
	}


def main():
	stats = _load_template_stats(TEMPLATE_PATH)
	search_template = _load_search_template(TEMPLATE_SEARCH_PATH)
	paths = _collect_images(SOURCE_DIR)
	out_dir = Path(OUT_DIR)
	mask_dir = Path(MASK_DIR)
	out_dir.mkdir(parents=True, exist_ok=True)
	mask_dir.mkdir(parents=True, exist_ok=True)

	print(
		f"template={Path(TEMPLATE_PATH).name} "
		f"| median_rgb={stats['median_rgb']} "
		f"| mad_rgb={stats['mad_rgb']} "
		f"| color_dist_thresh={COLOR_DISTANCE_THRESHOLD:.2f} "
		f"| search={SEARCH_START_PERCENT:.2f}%..{SEARCH_END_PERCENT:.2f}% "
		f"| samples={stats['sample_count']}"
	)

	for path in paths:
		result = process_image(path, out_dir, mask_dir, stats, search_template=search_template)
		print(
			f"{result['name']}: {result['size'][0]}x{result['size'][1]} | "
			f"search_px={result['search_start_px']}..{result['search_end_px']} | "
			f"components={result['components']} | "
			f"mask={result['mask_pixels']} px | "
			f"removed={result['removed_pixels']} px | "
			f"{result['out_path']}"
		)

	print(f"Processed {len(paths)} image(s) to {OUT_DIR}")


if __name__ == "__main__":
	main()
