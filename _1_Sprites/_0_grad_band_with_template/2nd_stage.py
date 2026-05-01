import os
from collections import deque
from pathlib import Path
from statistics import median

from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/test/"
name = "cards"

SOURCE_DIR = os.path.join(root_dir, "./")
CUTOUT_DIR = os.path.join(root_dir, f"./_0_{name}_band_cutout/")
MASK_DIR = os.path.join(root_dir, f"./_1_{name}_band_mask/")

OUT_BAND_DIR = os.path.join(root_dir, f"./_2_{name}_band_protected/")
OUT_COMPOSITE_DIR = os.path.join(root_dir, f"./_3_{name}_band_composite/")

TEMPLATE_PATH = os.path.join(root_dir, "./temp.png")
TEMPLATE_SEARCH_PATH = os.path.join(root_dir, "./temp2.png")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALPHA_THRESHOLD = 1
OVERWRITE = True

# Tint source
TINT_COLOR_OVERRIDE = None
TINT_DARKEN = 0.10

# Protection widths
INWARD_BAND_PERCENT = 1.50
OUTWARD_BAND_PERCENT = 1.50
OUTWARD_PADDING = 4

# Inward protection
INWARD_TINT_ENABLED = True
INWARD_TINT_STRENGTH = 0.35
INWARD_ALPHA_FLOOR = 144
INWARD_TINT_BIAS = 0.35

# Outward protection
OUTWARD_TINT_ENABLED = True
OUTWARD_TINT_STRENGTH = 0.45
OUTWARD_ALPHA_MAX = 164


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


def _load_tint_color(template_path: str):
	if TINT_COLOR_OVERRIDE is not None:
		return TINT_COLOR_OVERRIDE

	template = Image.open(template_path).convert("RGBA")
	pix = template.load()
	w, h = template.size
	samples = []

	for y in range(h):
		for x in range(w):
			r, g, b, a = pix[x, y]
			if a < ALPHA_THRESHOLD:
				continue
			samples.append((r, g, b))

	if not samples:
		raise RuntimeError(f"Template has no usable pixels: {template_path}")

	base = tuple(int(round(median(channel))) for channel in zip(*samples))
	return tuple(max(0, min(255, int(round(channel * (1.0 - TINT_DARKEN))))) for channel in base)


def _alpha_bounds(img: Image.Image):
	alpha = img.getchannel("A")
	pix = alpha.load()
	w, h = img.size
	min_x = w
	min_y = h
	max_x = -1
	max_y = -1

	for y in range(h):
		for x in range(w):
			if pix[x, y] < ALPHA_THRESHOLD:
				continue
			if x < min_x:
				min_x = x
			if y < min_y:
				min_y = y
			if x > max_x:
				max_x = x
			if y > max_y:
				max_y = y

	return min_x, min_y, max_x, max_y


def _band_width_px(img: Image.Image, percent: float) -> int:
	min_x, min_y, max_x, max_y = _alpha_bounds(img)
	if max_x < 0 or max_y < 0:
		return 0

	content_w = max_x - min_x + 1
	content_h = max_y - min_y + 1
	short_edge = min(content_w, content_h)
	return max(1, int(round(short_edge * percent / 100.0)))


def _expand_canvas(img: Image.Image, padding: int) -> Image.Image:
	if padding <= 0:
		return img.copy()

	w, h = img.size
	out = Image.new("RGBA", (w + padding * 2, h + padding * 2), (0, 0, 0, 0))
	out.paste(img, (padding, padding))
	return out


def _crop_center(img: Image.Image, size: tuple[int, int]) -> Image.Image:
	target_w, target_h = size
	w, h = img.size
	if (w, h) == (target_w, target_h):
		return img.copy()

	left = max(0, (w - target_w) // 2)
	top = max(0, (h - target_h) // 2)
	right = left + target_w
	bottom = top + target_h
	return img.crop((left, top, right, bottom))


def _reconstruct_band(source_img: Image.Image, mask_img: Image.Image) -> tuple[Image.Image, int]:
	if source_img.size != mask_img.size:
		raise ValueError(f"Mask size {mask_img.size} does not match source size {source_img.size}")

	source = source_img.convert("RGBA")
	mask = mask_img.convert("L")
	src_pix = source.load()
	mask_pix = mask.load()
	w, h = source.size
	band = Image.new("RGBA", source.size, (0, 0, 0, 0))
	band_pix = band.load()
	band_pixels = 0

	for y in range(h):
		for x in range(w):
			if mask_pix[x, y] < ALPHA_THRESHOLD:
				continue
			band_pix[x, y] = src_pix[x, y]
			band_pixels += 1

	return band, band_pixels


def _apply_inward_gradient_band(img: Image.Image, tint_color: tuple[int, int, int]) -> tuple[Image.Image, int]:
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	band_px = _band_width_px(img, INWARD_BAND_PERCENT)

	if band_px <= 0:
		return img.copy(), 0

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= ALPHA_THRESHOLD

	neighbors = (
		(-1, -1), (0, -1), (1, -1),
		(-1, 0),           (1, 0),
		(-1, 1),  (0, 1),  (1, 1),
	)

	for y in range(h):
		for x in range(w):
			if not is_inside(x, y):
				continue

			is_boundary = False
			for dx, dy in neighbors:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					is_boundary = True
					break
				if not is_inside(nx, ny):
					is_boundary = True
					break

			if is_boundary:
				dist[y][x] = 0
				queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= band_px - 1:
			continue

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if not is_inside(nx, ny):
				continue
			if dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			queue.append((nx, ny))

	out = img.copy()
	out_pix = out.load()
	alpha_out = out.getchannel("A")
	out_alpha = alpha_out.load()
	tr, tg, tb = tint_color

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d == -1:
				continue

			falloff = float(d + 1) / float(band_px)
			target_alpha = max(
				0,
				min(255, int(round(INWARD_ALPHA_FLOOR + (255 - INWARD_ALPHA_FLOOR) * falloff)))
			)
			out_alpha[x, y] = min(out_alpha[x, y], target_alpha)

			if INWARD_TINT_ENABLED:
				edge_mix = max(0.0, min(1.0, 1.0 - (d / float(max(1, band_px - 1)))))
				edge_mix = max(0.0, min(1.0, edge_mix + INWARD_TINT_BIAS * (1.0 - edge_mix)))
				mix = INWARD_TINT_STRENGTH * edge_mix
				r, g, b, a = out_pix[x, y]
				out_pix[x, y] = (
					int(round(r * (1.0 - mix) + tr * mix)),
					int(round(g * (1.0 - mix) + tg * mix)),
					int(round(b * (1.0 - mix) + tb * mix)),
					a,
				)

	out.putalpha(alpha_out)
	return out, band_px


def _apply_outward_gradient_band(img: Image.Image, tint_color: tuple[int, int, int]) -> tuple[Image.Image, int]:
	band_px = _band_width_px(img, OUTWARD_BAND_PERCENT)
	if band_px <= 0:
		return img.copy(), 0

	padding = max(OUTWARD_PADDING, band_px)
	out = _expand_canvas(img, padding)
	alpha = out.getchannel("A")
	src = alpha.load()
	out_pix = out.load()
	alpha_out = alpha.copy()
	out_alpha = alpha_out.load()
	w, h = out.size

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	seed_rgb = [[None for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= ALPHA_THRESHOLD

	neighbors = (
		(-1, -1), (0, -1), (1, -1),
		(-1, 0),           (1, 0),
		(-1, 1),  (0, 1),  (1, 1),
	)

	for y in range(h):
		for x in range(w):
			if not is_inside(x, y):
				continue

			is_boundary = False
			for dx, dy in neighbors:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					is_boundary = True
					break
				if not is_inside(nx, ny):
					is_boundary = True
					break

			if is_boundary:
				dist[y][x] = 0
				r, g, b, _ = out_pix[x, y]
				seed_rgb[y][x] = (r, g, b)
				queue.append((x, y))

	tr, tg, tb = tint_color
	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= band_px:
			continue

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if is_inside(nx, ny):
				continue
			if dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			seed_rgb[ny][nx] = seed_rgb[y][x]
			queue.append((nx, ny))

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d <= 0 or d > band_px:
				continue

			r, g, b = seed_rgb[y][x]
			falloff = max(0.0, 1.0 - (d / float(band_px + 1)))
			alpha_value = max(0, min(255, int(round(OUTWARD_ALPHA_MAX * falloff))))

			if OUTWARD_TINT_ENABLED:
				mix = OUTWARD_TINT_STRENGTH * falloff
				r = int(round(r * (1.0 - mix) + tr * mix))
				g = int(round(g * (1.0 - mix) + tg * mix))
				b = int(round(b * (1.0 - mix) + tb * mix))

			out_pix[x, y] = (r, g, b, alpha_value)
			out_alpha[x, y] = alpha_value

	out.putalpha(alpha_out)
	return out, band_px


def _composite_band(cutout_img: Image.Image, band_layer: Image.Image) -> Image.Image:
	base = cutout_img.convert("RGBA")
	out = base.copy()
	out.alpha_composite(band_layer)
	return out


def process_image(path: Path, cutout_dir: Path, mask_dir: Path, out_band_dir: Path, out_composite_dir: Path, tint_color):
	source_img = Image.open(path).convert("RGBA")
	mask_path = mask_dir / path.name

	if not mask_path.exists():
		raise FileNotFoundError(f"1st-stage mask not found: {mask_path}")

	mask_img = Image.open(mask_path).convert("L")
	band_img, band_pixels = _reconstruct_band(source_img, mask_img)
	band_inward, inward_px = _apply_inward_gradient_band(band_img, tint_color)
	band_protected, outward_px = _apply_outward_gradient_band(band_inward, tint_color)
	band_protected = _crop_center(band_protected, source_img.size)
	composite = _composite_band(source_img, band_protected)

	out_band_path = out_band_dir / path.name
	out_composite_path = out_composite_dir / path.name

	if (out_band_path.exists() or out_composite_path.exists()) and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {path.name}")

	band_protected.save(out_band_path, "PNG")
	composite.save(out_composite_path, "PNG")

	return {
		"name": path.name,
		"size": source_img.size,
		"band_pixels": band_pixels,
		"inward_px": inward_px,
		"outward_px": outward_px,
		"out_band_path": out_band_path,
		"out_composite_path": out_composite_path,
	}


def main():
	paths = _collect_images(SOURCE_DIR)
	tint_color = _load_tint_color(TEMPLATE_PATH)
	cutout_dir = Path(CUTOUT_DIR)
	mask_dir = Path(MASK_DIR)
	out_band_dir = Path(OUT_BAND_DIR)
	out_composite_dir = Path(OUT_COMPOSITE_DIR)

	out_band_dir.mkdir(parents=True, exist_ok=True)
	out_composite_dir.mkdir(parents=True, exist_ok=True)

	print(
		f"tint={tint_color} | "
		f"inward={INWARD_BAND_PERCENT:.2f}% | "
		f"outward={OUTWARD_BAND_PERCENT:.2f}%"
	)

	for path in paths:
		result = process_image(
			path,
			cutout_dir=cutout_dir,
			mask_dir=mask_dir,
			out_band_dir=out_band_dir,
			out_composite_dir=out_composite_dir,
			tint_color=tint_color,
		)
		print(
			f"{result['name']}: {result['size'][0]}x{result['size'][1]} | "
			f"band_pixels={result['band_pixels']} | "
			f"inward={result['inward_px']}px | "
			f"outward={result['outward_px']}px | "
			f"{result['out_composite_path']}"
		)

	print(f"Processed {len(paths)} image(s) to {OUT_COMPOSITE_DIR}")


if __name__ == "__main__":
	main()
