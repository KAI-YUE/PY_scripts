import os
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/-1_field_landscape/rocks_stuff/export/"
root_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/ui_box/_1_icons/"
name = "grass_dec"
name = "tree_terrain"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DEBUG_DIR = os.path.join(root_dir, f"./{name}_alpha_edge_debug/")
OUT_FADE_DIR = os.path.join(root_dir, f"./{name}_alpha_edge_fade/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALPHA_THRESHOLD = 1
EDGE_DETECT_ALPHA_THRESHOLD = 254
EDGE_NEIGHBOR_MODE = 8          # 4 or 8
EDGE_MASK_WIDTH = 1             # debug / visualization band width

FADE_DIRECTION = "inward"      # "outward" | "inward" | "both"
INWARD_FADE_WIDTH = 3           # fade width into the foreground
OUTWARD_FADE_WIDTH = 6          # protection fade width into transparent pixels

FADE_COLOR_MODE = "median"      # "sampled" | "median" | "transparent"
INWARD_ALPHA_FLOOR = 96
FADE_MAX_ALPHA = 96
OVERWRITE = True

USE_MASK_MORPHOLOGY = True
MORPH_MODE = "close"            # "open" | "close" | "none"
MORPH_RADIUS = 1

SAVE_INSIDE_MASK = True
SAVE_OUTSIDE_MASK = True
SAVE_EDGE_OVERLAY = True
SAVE_CLEAN_MASK = True
SAVE_PROTECTED_RESULT = True

INSIDE_EDGE_COLOR = (255, 80, 80, 255)
OUTSIDE_EDGE_COLOR = (80, 200, 255, 255)


def _collect_images(source_dir: str):
	src = Path(source_dir)
	if not src.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	paths = []
	for p in src.iterdir():
		if p.is_file() and p.suffix.lower() in INCLUDE_EXTS:
			paths.append(p)

	if not paths:
		raise RuntimeError(f"No images found in {source_dir} with extensions {sorted(INCLUDE_EXTS)}")

	return sorted(paths, key=lambda p: p.name.lower())


def _neighbors():
	if EDGE_NEIGHBOR_MODE == 4:
		return ((0, -1), (-1, 0), (1, 0), (0, 1))
	if EDGE_NEIGHBOR_MODE == 8:
		return (
			(-1, -1), (0, -1), (1, -1),
			(-1, 0),           (1, 0),
			(-1, 1),  (0, 1),  (1, 1),
		)
	raise ValueError(f"EDGE_NEIGHBOR_MODE must be 4 or 8, got {EDGE_NEIGHBOR_MODE}")


def _new_mask(size: tuple[int, int]) -> Image.Image:
	return Image.new("L", size, 0)


def _build_detection_mask(img: Image.Image) -> Image.Image:
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	mask = _new_mask(img.size)
	mask_pix = mask.load()

	for y in range(h):
		for x in range(w):
			if src[x, y] >= EDGE_DETECT_ALPHA_THRESHOLD:
				mask_pix[x, y] = 255

	return mask


def _apply_mask_morphology(mask: Image.Image) -> Image.Image:
	if not USE_MASK_MORPHOLOGY or MORPH_MODE == "none" or MORPH_RADIUS <= 0:
		return mask.copy()

	size = MORPH_RADIUS * 2 + 1
	if MORPH_MODE == "open":
		return mask.filter(ImageFilter.MinFilter(size=size)).filter(ImageFilter.MaxFilter(size=size))
	if MORPH_MODE == "close":
		return mask.filter(ImageFilter.MaxFilter(size=size)).filter(ImageFilter.MinFilter(size=size))

	raise ValueError(f"MORPH_MODE must be 'open', 'close', or 'none', got {MORPH_MODE}")


def _find_boundary_masks(mask: Image.Image) -> tuple[Image.Image, Image.Image]:
	src = mask.load()
	w, h = mask.size
	inside = _new_mask(mask.size)
	outside = _new_mask(mask.size)
	inside_pix = inside.load()
	outside_pix = outside.load()
	neighbors = _neighbors()

	def is_fg(x: int, y: int) -> bool:
		return src[x, y] > 0

	for y in range(h):
		for x in range(w):
			current_is_fg = is_fg(x, y)
			intersects_bg = False
			intersects_fg = False

			for dx, dy in neighbors:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					if current_is_fg:
						intersects_bg = True
					continue

				neighbor_is_fg = is_fg(nx, ny)
				if current_is_fg and not neighbor_is_fg:
					intersects_bg = True
					break
				if (not current_is_fg) and neighbor_is_fg:
					intersects_fg = True
					break

			if current_is_fg and intersects_bg:
				inside_pix[x, y] = 255
			elif (not current_is_fg) and intersects_fg:
				outside_pix[x, y] = 255

	return inside, outside


def _grow_mask(mask: Image.Image, max_dist: int) -> Image.Image:
	if max_dist <= 1:
		return mask.copy()

	src = mask.load()
	w, h = mask.size
	out = mask.copy()
	out_pix = out.load()
	neighbors = _neighbors()

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	for y in range(h):
		for x in range(w):
			if src[x, y] > 0:
				dist[y][x] = 0
				queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= max_dist - 1:
			continue

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			out_pix[nx, ny] = 255
			queue.append((nx, ny))

	return out


def _mask_to_rgba(mask: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
	w, h = mask.size
	out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
	mask_pix = mask.load()
	out_pix = out.load()
	r, g, b = rgb

	for y in range(h):
		for x in range(w):
			if mask_pix[x, y] > 0:
				out_pix[x, y] = (r, g, b, 255)

	return out


def _compose_overlay(
	img: Image.Image,
	inside_mask: Image.Image,
	outside_mask: Image.Image,
) -> Image.Image:
	base = img.copy()
	base_pix = base.load()
	inside_pix = inside_mask.load()
	outside_pix = outside_mask.load()
	w, h = img.size

	for y in range(h):
		for x in range(w):
			if inside_pix[x, y] > 0:
				base_pix[x, y] = INSIDE_EDGE_COLOR
			elif outside_pix[x, y] > 0:
				base_pix[x, y] = OUTSIDE_EDGE_COLOR

	return base


def _median_edge_rgb(img: Image.Image, inside_mask: Image.Image) -> tuple[int, int, int] | None:
	src_pix = img.load()
	mask_pix = inside_mask.load()
	w, h = img.size
	rs = []
	gs = []
	bs = []

	for y in range(h):
		for x in range(w):
			if mask_pix[x, y] <= 0:
				continue
			r, g, b, _ = src_pix[x, y]
			rs.append(r)
			gs.append(g)
			bs.append(b)

	if not rs:
		return None

	rs.sort()
	gs.sort()
	bs.sort()
	mid = len(rs) // 2
	return rs[mid], gs[mid], bs[mid]


def _build_outward_protection(img: Image.Image, inside_mask: Image.Image) -> Image.Image:
	if OUTWARD_FADE_WIDTH <= 0:
		return img.copy()

	alpha = img.getchannel("A")
	src_alpha = alpha.load()
	src_pix = img.load()
	w, h = img.size
	out = img.copy()
	out_pix = out.load()
	neighbors = _neighbors()
	median_rgb = _median_edge_rgb(img, inside_mask) if FADE_COLOR_MODE == "median" else None

	if FADE_COLOR_MODE not in {"sampled", "median", "transparent"}:
		raise ValueError(
			f"FADE_COLOR_MODE must be 'sampled', 'median', or 'transparent', got {FADE_COLOR_MODE}"
		)

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	seed_rgb = [[None for _ in range(w)] for _ in range(h)]
	queue = deque()

	for y in range(h):
		for x in range(w):
			if inside_mask.getpixel((x, y)) <= 0:
				continue

			dist[y][x] = 0
			if FADE_COLOR_MODE == "median":
				seed_rgb[y][x] = median_rgb
			elif FADE_COLOR_MODE == "transparent":
				seed_rgb[y][x] = (0, 0, 0)
			else:
				r, g, b, _ = src_pix[x, y]
				seed_rgb[y][x] = (r, g, b)
			queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= OUTWARD_FADE_WIDTH:
			continue

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if src_alpha[nx, ny] >= ALPHA_THRESHOLD:
				continue
			if dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			seed_rgb[ny][nx] = seed_rgb[y][x]
			queue.append((nx, ny))

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d <= 0:
				continue

			falloff = max(0.0, 1.0 - ((d - 1) / float(max(1, OUTWARD_FADE_WIDTH))))
			target_alpha = max(0, min(255, int(round(FADE_MAX_ALPHA * falloff))))
			if target_alpha <= 0:
				continue

			seed = seed_rgb[y][x]
			if seed is None:
				continue

			r, g, b = seed
			out_pix[x, y] = (r, g, b, target_alpha)

	return out


def _build_inward_fade(img: Image.Image, inside_mask: Image.Image) -> Image.Image:
	if INWARD_FADE_WIDTH <= 0:
		return img.copy()

	mask_pix = inside_mask.load()
	alpha_img = img.getchannel("A")
	src_alpha = alpha_img.load()
	w, h = img.size
	out = img.copy()
	out_alpha_img = out.getchannel("A")
	out_alpha = out_alpha_img.load()
	neighbors = _neighbors()

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	for y in range(h):
		for x in range(w):
			if mask_pix[x, y] <= 0:
				continue
			dist[y][x] = 0
			queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= INWARD_FADE_WIDTH - 1:
			continue

		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if src_alpha[nx, ny] < ALPHA_THRESHOLD:
				continue
			if dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			queue.append((nx, ny))

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d == -1:
				continue

			falloff = float(d + 1) / float(max(1, INWARD_FADE_WIDTH))
			target_alpha = max(
				0,
				min(255, int(round(INWARD_ALPHA_FLOOR + (255 - INWARD_ALPHA_FLOOR) * falloff))),
			)
			out_alpha[x, y] = min(out_alpha[x, y], target_alpha)

	out.putalpha(out_alpha_img)
	return out


def _build_protected_result(img: Image.Image, inside_mask: Image.Image) -> Image.Image:
	if FADE_DIRECTION not in {"outward", "inward", "both"}:
		raise ValueError(f"FADE_DIRECTION must be 'outward', 'inward', or 'both', got {FADE_DIRECTION}")

	if FADE_DIRECTION == "outward":
		return _build_outward_protection(img, inside_mask)
	if FADE_DIRECTION == "inward":
		return _build_inward_fade(img, inside_mask)

	return _build_inward_fade(_build_outward_protection(img, inside_mask), inside_mask)


def process_image(path: Path, debug_dir: Path, fade_dir: Path):
	img = Image.open(path).convert("RGBA")
	detection_mask = _build_detection_mask(img)
	cleaned_mask = _apply_mask_morphology(detection_mask)
	inside_edge, outside_edge = _find_boundary_masks(cleaned_mask)
	inside_band = _grow_mask(inside_edge, EDGE_MASK_WIDTH)
	outside_band = _grow_mask(outside_edge, EDGE_MASK_WIDTH)

	debug_dir.mkdir(parents=True, exist_ok=True)
	fade_dir.mkdir(parents=True, exist_ok=True)
	stem = path.stem

	inside_path = debug_dir / f"{stem}_inside_mask.png"
	outside_path = debug_dir / f"{stem}_outside_mask.png"
	overlay_path = debug_dir / f"{stem}_edge_overlay.png"
	clean_mask_path = debug_dir / f"{stem}_clean_mask.png"
	fade_path = fade_dir / path.name

	if not OVERWRITE:
		for candidate in (inside_path, outside_path, overlay_path, clean_mask_path, fade_path):
			if candidate.exists():
				raise FileExistsError(f"Output already exists: {candidate}")

	if SAVE_CLEAN_MASK:
		cleaned_mask.save(clean_mask_path, "PNG")
	if SAVE_INSIDE_MASK:
		_mask_to_rgba(inside_band, INSIDE_EDGE_COLOR[:3]).save(inside_path, "PNG")
	if SAVE_OUTSIDE_MASK:
		_mask_to_rgba(outside_band, OUTSIDE_EDGE_COLOR[:3]).save(outside_path, "PNG")
	if SAVE_EDGE_OVERLAY:
		_compose_overlay(img, inside_band, outside_band).save(overlay_path, "PNG")
	if SAVE_PROTECTED_RESULT:
		_build_protected_result(img, inside_edge).save(fade_path, "PNG")

	inside_count = sum(1 for value in inside_band.getdata() if value > 0)
	outside_count = sum(1 for value in outside_band.getdata() if value > 0)
	return path.name, img.size, inside_count, outside_count


def main():
	paths = _collect_images(SOURCE_DIR)
	debug_dir = Path(OUT_DEBUG_DIR)
	fade_dir = Path(OUT_FADE_DIR)

	for path in paths:
		name, size, inside_count, outside_count = process_image(path, debug_dir, fade_dir)
		print(
			f"{name}: {size[0]}x{size[1]} | "
			f"inside_edge_px={inside_count} | outside_edge_px={outside_count}"
		)

	print(f"Processed {len(paths)} image(s) | debug={OUT_DEBUG_DIR} | fade={OUT_FADE_DIR}")


if __name__ == "__main__":
	main()
