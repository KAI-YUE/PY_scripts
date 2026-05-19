import os
from collections import deque
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/_0_ui_box/_1_icons/dots/"
name = "ui_box"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DIR = os.path.join(root_dir, f"./_3_{name}_mask_dilated/")
OUT_DEBUG_DIR = os.path.join(root_dir, f"./_3_{name}_mask_dilated_debug/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MASK_SUFFIX = "-mask"
OUTPUT_MASK_SUFFIX = "-mask"

MASK_SUFFIX = ""
OUTPUT_MASK_SUFFIX = ""

ALPHA_THRESHOLD = 1
OUTPUT_ALPHA = 255
OVERWRITE = True
HARD_SET_SPRITE_RGBA = True
HARD_SET_RGBA_VALUE = (255, 255, 255, 255)

CANVAS_EXPAND_ENABLED = True
CANVAS_EXTRA_PADDING = 1

# Binary morphology. Close fills small zigzag cuts/gaps, open removes small
# teeth/spikes, and final dilation expands the mask edge outward.
MORPH_CLOSE_ENABLED = True
MORPH_CLOSE_SIZE = 5            # Odd integer >= 3.
MORPH_OPEN_ENABLED = False
MORPH_OPEN_SIZE = 3             # Odd integer >= 3.
FINAL_DILATE_SIZE = 3           # 1 disables; odd integer.

# Potential filtering treats the alpha mask as a soft field: blur first, then
# threshold back to binary. This rounds stair-step edges after morphology.
POTENTIAL_FILTER_ENABLED = True
POTENTIAL_BLUR_RADIUS = 1.15
POTENTIAL_THRESHOLD = 96        # 0..255. Lower grows; higher shrinks.
POTENTIAL_PASSES = 1

# Component filtering removes tiny islands and fills tiny transparent holes.
FILTER_SMALL_ISLANDS = True
MIN_ISLAND_AREA = 24
FILTER_SMALL_HOLES = True
MAX_HOLE_AREA = 64

SAVE_DEBUG_OVERLAY = True
ADDED_COLOR = (80, 220, 120, 255)
REMOVED_COLOR = (255, 80, 80, 255)
KEPT_COLOR = (255, 255, 255, 110)


def _collect_masks(source_dir: str):
	src = Path(source_dir)
	if not src.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	paths = []
	for p in src.iterdir():
		if not p.is_file() or p.suffix.lower() not in INCLUDE_EXTS:
			continue
		if MASK_SUFFIX and not p.stem.endswith(MASK_SUFFIX):
			continue
		paths.append(p)

	if not paths:
		raise RuntimeError(
			f"No mask images found in {source_dir}; expected files ending with {MASK_SUFFIX!r}"
		)

	return sorted(paths, key=lambda p: p.name.lower())


def _validate_filter_size(name: str, size: int, allow_one: bool = False):
	if allow_one and size == 1:
		return
	if size < 3 or size % 2 == 0:
		raise ValueError(f"{name} must be an odd integer >= 3")


def _binary_alpha(img: Image.Image) -> Image.Image:
	alpha = img.getchannel("A")
	return alpha.point(lambda a: 255 if a >= ALPHA_THRESHOLD else 0, "L")


def _mask_bounds(mask: Image.Image) -> tuple[int, int, int, int] | None:
	pix = mask.load()
	w, h = mask.size
	min_x = w
	min_y = h
	max_x = -1
	max_y = -1

	for y in range(h):
		for x in range(w):
			if pix[x, y] == 0:
				continue
			if x < min_x:
				min_x = x
			if y < min_y:
				min_y = y
			if x > max_x:
				max_x = x
			if y > max_y:
				max_y = y

	if max_x < 0:
		return None
	return min_x, min_y, max_x, max_y


def _filter_radius(size: int) -> int:
	if size <= 1:
		return 0
	return (size - 1) // 2


def _estimated_growth_px() -> int:
	growth = 0

	if MORPH_CLOSE_ENABLED:
		growth += _filter_radius(MORPH_CLOSE_SIZE)
	if FINAL_DILATE_SIZE > 1:
		growth += _filter_radius(FINAL_DILATE_SIZE)
	if POTENTIAL_FILTER_ENABLED and POTENTIAL_BLUR_RADIUS > 0 and POTENTIAL_PASSES > 0:
		growth += int(round(POTENTIAL_BLUR_RADIUS * 3.0)) * POTENTIAL_PASSES

	return max(0, growth + CANVAS_EXTRA_PADDING)


def _estimate_canvas_padding(mask: Image.Image) -> tuple[int, int, int, int]:
	if not CANVAS_EXPAND_ENABLED:
		return 0, 0, 0, 0

	bounds = _mask_bounds(mask)
	if bounds is None:
		return 0, 0, 0, 0

	growth = _estimated_growth_px()
	if growth <= 0:
		return 0, 0, 0, 0

	min_x, min_y, max_x, max_y = bounds
	w, h = mask.size
	left = max(0, growth - min_x)
	top = max(0, growth - min_y)
	right = max(0, max_x + growth - (w - 1))
	bottom = max(0, max_y + growth - (h - 1))
	return left, top, right, bottom


def _expand_canvas(img: Image.Image, padding: tuple[int, int, int, int]) -> Image.Image:
	left, top, right, bottom = padding
	if left == 0 and top == 0 and right == 0 and bottom == 0:
		return img

	w, h = img.size
	out = Image.new("RGBA", (w + left + right, h + top + bottom), (0, 0, 0, 0))
	out.paste(img, (left, top))
	return out


def _hard_set_sprite_rgba(img: Image.Image) -> Image.Image:
	if not HARD_SET_SPRITE_RGBA:
		return img

	r, g, b, a = HARD_SET_RGBA_VALUE
	src = img.convert("RGBA")
	alpha = src.getchannel("A")
	mask = alpha.point(lambda value: 255 if value >= ALPHA_THRESHOLD else 0, "L")
	out = Image.new("RGBA", src.size, (0, 0, 0, 0))
	fill = Image.new("RGBA", src.size, (r, g, b, a))
	out.paste(fill, (0, 0), mask)
	return out


def _apply_morphology(mask: Image.Image) -> Image.Image:
	out = mask.copy()

	if MORPH_CLOSE_ENABLED:
		_validate_filter_size("MORPH_CLOSE_SIZE", MORPH_CLOSE_SIZE)
		out = out.filter(ImageFilter.MaxFilter(MORPH_CLOSE_SIZE))
		out = out.filter(ImageFilter.MinFilter(MORPH_CLOSE_SIZE))

	if MORPH_OPEN_ENABLED:
		_validate_filter_size("MORPH_OPEN_SIZE", MORPH_OPEN_SIZE)
		out = out.filter(ImageFilter.MinFilter(MORPH_OPEN_SIZE))
		out = out.filter(ImageFilter.MaxFilter(MORPH_OPEN_SIZE))

	_validate_filter_size("FINAL_DILATE_SIZE", FINAL_DILATE_SIZE, allow_one=True)
	if FINAL_DILATE_SIZE > 1:
		out = out.filter(ImageFilter.MaxFilter(FINAL_DILATE_SIZE))

	return out


def _apply_potential_filter(mask: Image.Image) -> Image.Image:
	if not POTENTIAL_FILTER_ENABLED:
		return mask
	if POTENTIAL_BLUR_RADIUS <= 0 or POTENTIAL_PASSES <= 0:
		return mask

	out = mask.copy()
	for _ in range(POTENTIAL_PASSES):
		potential = out.filter(ImageFilter.GaussianBlur(POTENTIAL_BLUR_RADIUS))
		out = potential.point(lambda a: 255 if a >= POTENTIAL_THRESHOLD else 0, "L")
	return out


def _neighbors():
	return (
		(-1, -1), (0, -1), (1, -1),
		(-1,  0),          (1,  0),
		(-1,  1), (0,  1), (1,  1),
	)


def _remove_small_islands(mask: Image.Image) -> tuple[Image.Image, int]:
	if not FILTER_SMALL_ISLANDS or MIN_ISLAND_AREA <= 0:
		return mask, 0

	src = mask.load()
	w, h = mask.size
	visited = [[False for _ in range(w)] for _ in range(h)]
	out = mask.copy()
	out_pix = out.load()
	removed = 0

	for y in range(h):
		for x in range(w):
			if visited[y][x] or src[x, y] == 0:
				continue

			visited[y][x] = True
			queue = deque([(x, y)])
			pixels = [(x, y)]

			while queue:
				cx, cy = queue.popleft()
				for dx, dy in _neighbors():
					nx = cx + dx
					ny = cy + dy
					if nx < 0 or ny < 0 or nx >= w or ny >= h:
						continue
					if visited[ny][nx] or src[nx, ny] == 0:
						continue
					visited[ny][nx] = True
					queue.append((nx, ny))
					pixels.append((nx, ny))

			if len(pixels) >= MIN_ISLAND_AREA:
				continue
			for px, py in pixels:
				out_pix[px, py] = 0
			removed += len(pixels)

	return out, removed


def _fill_small_holes(mask: Image.Image) -> tuple[Image.Image, int]:
	if not FILTER_SMALL_HOLES or MAX_HOLE_AREA <= 0:
		return mask, 0

	src = mask.load()
	w, h = mask.size
	visited = [[False for _ in range(w)] for _ in range(h)]
	out = mask.copy()
	out_pix = out.load()
	filled = 0

	for y in range(h):
		for x in range(w):
			if visited[y][x] or src[x, y] > 0:
				continue

			visited[y][x] = True
			queue = deque([(x, y)])
			pixels = [(x, y)]
			touches_edge = x == 0 or y == 0 or x == w - 1 or y == h - 1

			while queue:
				cx, cy = queue.popleft()
				if cx == 0 or cy == 0 or cx == w - 1 or cy == h - 1:
					touches_edge = True

				for dx, dy in _neighbors():
					nx = cx + dx
					ny = cy + dy
					if nx < 0 or ny < 0 or nx >= w or ny >= h:
						continue
					if visited[ny][nx] or src[nx, ny] > 0:
						continue
					visited[ny][nx] = True
					queue.append((nx, ny))
					pixels.append((nx, ny))

			if touches_edge or len(pixels) > MAX_HOLE_AREA:
				continue
			for px, py in pixels:
				out_pix[px, py] = 255
			filled += len(pixels)

	return out, filled


def _apply_component_filters(mask: Image.Image) -> tuple[Image.Image, int, int]:
	out, removed_islands = _remove_small_islands(mask)
	out, filled_holes = _fill_small_holes(out)
	return out, removed_islands, filled_holes


def _median_mask_rgb(img: Image.Image) -> tuple[int, int, int]:
	pix = img.load()
	w, h = img.size
	rs = []
	gs = []
	bs = []

	for y in range(h):
		for x in range(w):
			r, g, b, a = pix[x, y]
			if a < ALPHA_THRESHOLD:
				continue
			rs.append(r)
			gs.append(g)
			bs.append(b)

	if not rs:
		return 255, 255, 255

	rs.sort()
	gs.sort()
	bs.sort()
	mid = len(rs) // 2
	return rs[mid], gs[mid], bs[mid]


def _compose_mask_rgba(source_img: Image.Image, mask: Image.Image) -> Image.Image:
	source = source_img.convert("RGBA")
	source_pix = source.load()
	mask_pix = mask.load()
	w, h = source.size
	default_rgb = HARD_SET_RGBA_VALUE[:3] if HARD_SET_SPRITE_RGBA else _median_mask_rgb(source)
	output_alpha = HARD_SET_RGBA_VALUE[3] if HARD_SET_SPRITE_RGBA else OUTPUT_ALPHA
	out = Image.new("RGBA", source.size, (0, 0, 0, 0))
	out_pix = out.load()

	for y in range(h):
		for x in range(w):
			if mask_pix[x, y] == 0:
				continue
			r, g, b, a = source_pix[x, y]
			if a < ALPHA_THRESHOLD:
				r, g, b = default_rgb
			out_pix[x, y] = (r, g, b, output_alpha)

	return out


def _debug_overlay(original: Image.Image, before: Image.Image, after: Image.Image) -> Image.Image:
	out = Image.new("RGBA", original.size, (0, 0, 0, 0))
	out_pix = out.load()
	before_pix = before.load()
	after_pix = after.load()
	w, h = original.size

	for y in range(h):
		for x in range(w):
			was_mask = before_pix[x, y] > 0
			is_mask = after_pix[x, y] > 0
			if was_mask and is_mask:
				out_pix[x, y] = KEPT_COLOR
			elif is_mask:
				out_pix[x, y] = ADDED_COLOR
			elif was_mask:
				out_pix[x, y] = REMOVED_COLOR

	return Image.alpha_composite(original.convert("RGBA"), out)


def _changed_counts(before: Image.Image, after: Image.Image) -> tuple[int, int, int]:
	added = ImageChops.subtract(after, before)
	removed = ImageChops.subtract(before, after)
	return (
		sum(1 for value in added.getdata() if value > 0),
		sum(1 for value in removed.getdata() if value > 0),
		sum(1 for value in after.getdata() if value > 0),
	)


def _output_name(path: Path) -> str:
	if not OUTPUT_MASK_SUFFIX or path.stem.endswith(OUTPUT_MASK_SUFFIX):
		return f"{path.stem}.png"
	if MASK_SUFFIX and path.stem.endswith(MASK_SUFFIX):
		base = path.stem[:-len(MASK_SUFFIX)]
		return f"{base}{OUTPUT_MASK_SUFFIX}.png"
	return f"{path.stem}{OUTPUT_MASK_SUFFIX}.png"


def process_mask(path: Path, out_dir: Path, debug_dir: Path):
	original = Image.open(path).convert("RGBA")
	source = _hard_set_sprite_rgba(original)
	before = _binary_alpha(source)
	canvas_padding = _estimate_canvas_padding(before)
	if any(canvas_padding):
		original = _expand_canvas(original, canvas_padding)
		source = _expand_canvas(source, canvas_padding)
		before = _binary_alpha(source)
	processed = _apply_morphology(before)
	processed = _apply_potential_filter(processed)
	processed, removed_islands, filled_holes = _apply_component_filters(processed)
	out_img = _compose_mask_rgba(source, processed)

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / _output_name(path)
	if out_path.exists() and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {out_path}")
	out_img.save(out_path, "PNG")

	debug_path = None
	if SAVE_DEBUG_OVERLAY:
		debug_dir.mkdir(parents=True, exist_ok=True)
		debug_path = debug_dir / _output_name(path)
		_debug_overlay(original, before, processed).save(debug_path, "PNG")

	added, removed, final_area = _changed_counts(before, processed)
	return out_path, debug_path, added, removed, final_area, removed_islands, filled_holes, canvas_padding


def main():
	paths = _collect_masks(SOURCE_DIR)
	out_dir = Path(OUT_DIR)
	debug_dir = Path(OUT_DEBUG_DIR)

	for path in paths:
		out_path, debug_path, added, removed, final_area, removed_islands, filled_holes, canvas_padding = process_mask(
			path,
			out_dir,
			debug_dir,
		)
		print(
			f"{path.name}: added={added} removed={removed} final_area={final_area} | "
			f"islands_removed={removed_islands} holes_filled={filled_holes} | "
			f"canvas_pad={canvas_padding} | {out_path}"
		)
		if debug_path is not None:
			print(f"  debug: {debug_path}")

	print(f"Processed {len(paths)} mask image(s) to {OUT_DIR}")


if __name__ == "__main__":
	main()
