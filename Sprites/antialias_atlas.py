import os
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/"
# root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/"
name = "cards"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_TRIM_DIR = os.path.join(root_dir, f"./{name}_trimmed/")
OUT_DIR = os.path.join(root_dir, f"./{name}_antialias/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PADDING = 4
TARGET_LONG_EDGE = 128

TRIM_ALPHA = True
ALPHA_THRESHOLD = 1
ALPHA_HARD_TRIM = True
ALPHA_HARD_TRIM_PERCENT = 12

TRIM_KEEP = 0
SAVE_TRIM_ONLY = False
OVERWRITE = True

SOFT_EXTRUDE_ALPHA = True
SOFT_EXTRUDE_MAX_ALPHA = 96

EDGE_BLUR = False
EDGE_BLUR_RADIUS = 4
EDGE_BLUR_WIDTH = 12
EDGE_BLUR_MODE = "rgba"   # "alpha" | "rgba"


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


def _maybe_resize(img: Image.Image) -> Image.Image:
	if not TARGET_LONG_EDGE:
		return img

	w, h = img.size
	long_edge = max(w, h)
	if long_edge <= 0 or long_edge == TARGET_LONG_EDGE:
		return img

	scale = TARGET_LONG_EDGE / float(long_edge)
	nw = max(1, int(round(w * scale)))
	nh = max(1, int(round(h * scale)))
	return img.resize((nw, nh), resample=Image.Resampling.BICUBIC)


def _trim_to_alpha_bounds(img: Image.Image) -> Image.Image:
	if not TRIM_ALPHA:
		return img

	w, h = img.size
	pix = img.load()

	def has_visible_alpha(x: int, y: int) -> bool:
		return pix[x, y][3] >= ALPHA_THRESHOLD

	def alpha_percent_for_col(x: int) -> float:
		count = 0
		for y in range(h):
			if has_visible_alpha(x, y):
				count += 1
		return (count / float(h)) * 100.0 if h > 0 else 0.0

	def alpha_percent_for_row(y: int) -> float:
		count = 0
		for x in range(w):
			if has_visible_alpha(x, y):
				count += 1
		return (count / float(w)) * 100.0 if w > 0 else 0.0

	def loose_alpha_bounds():
		min_x = w
		min_y = h
		max_x = -1
		max_y = -1

		for y in range(h):
			for x in range(w):
				if has_visible_alpha(x, y):
					if x < min_x:
						min_x = x
					if y < min_y:
						min_y = y
					if x > max_x:
						max_x = x
					if y > max_y:
						max_y = y

		return min_x, min_y, max_x, max_y

	if ALPHA_HARD_TRIM:
		required_fill_percent = max(0.0, min(100.0, 100.0 - ALPHA_HARD_TRIM_PERCENT))

		min_x = 0
		while min_x < w and alpha_percent_for_col(min_x) < required_fill_percent:
			min_x += 1

		max_x = w - 1
		while max_x >= min_x and alpha_percent_for_col(max_x) < required_fill_percent:
			max_x -= 1

		min_y = 0
		while min_y < h and alpha_percent_for_row(min_y) < required_fill_percent:
			min_y += 1

		max_y = h - 1
		while max_y >= min_y and alpha_percent_for_row(max_y) < required_fill_percent:
			max_y -= 1

		# Some images never reach the required edge fill rate, especially at 100%.
		# Fall back to loose alpha bounds instead of returning an empty crop.
		if min_x >= w or min_y >= h or max_x < min_x or max_y < min_y:
			min_x, min_y, max_x, max_y = loose_alpha_bounds()
	else:
		min_x, min_y, max_x, max_y = loose_alpha_bounds()

	if max_x < 0 or max_y < 0:
		return img.copy()

	left = max(0, min_x - TRIM_KEEP)
	top = max(0, min_y - TRIM_KEEP)
	right = min(w, max_x + 1 + TRIM_KEEP)
	bottom = min(h, max_y + 1 + TRIM_KEEP)

	if left >= right or top >= bottom:
		return img.copy()

	return img.crop((left, top, right, bottom))


def _edge_band_mask(img: Image.Image, width: int) -> Image.Image:
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	mask = Image.new("L", (w, h), 0)
	mask_pix = mask.load()

	for y in range(h):
		for x in range(w):
			if src[x, y] <= 0:
				continue

			near_boundary = False
			for dy in range(-width, width + 1):
				ny = y + dy
				if ny < 0 or ny >= h:
					near_boundary = True
					break
				for dx in range(-width, width + 1):
					nx = x + dx
					if nx < 0 or nx >= w:
						near_boundary = True
						break
					if src[nx, ny] <= 0:
						near_boundary = True
						break
				if near_boundary:
					break

			if near_boundary:
				mask_pix[x, y] = 255

	return mask


def _apply_edge_blur(img: Image.Image) -> Image.Image:
	if not EDGE_BLUR or EDGE_BLUR_RADIUS <= 0 or EDGE_BLUR_WIDTH <= 0:
		return img

	edge_mask = _edge_band_mask(img, EDGE_BLUR_WIDTH)
	original_alpha = img.getchannel("A")

	if EDGE_BLUR_MODE == "alpha":
		blurred_alpha = original_alpha.filter(ImageFilter.GaussianBlur(radius=EDGE_BLUR_RADIUS))
		out = img.copy()
		out.putalpha(Image.composite(blurred_alpha, original_alpha, edge_mask))
		return out

	if EDGE_BLUR_MODE == "rgba":
		blurred = img.filter(ImageFilter.GaussianBlur(radius=EDGE_BLUR_RADIUS))
		return Image.composite(blurred, img, edge_mask)

	raise ValueError(f"Unknown EDGE_BLUR_MODE: {EDGE_BLUR_MODE}")


def _expand_canvas(img: Image.Image, padding: int) -> Image.Image:
	if padding <= 0:
		return img.copy()

	w, h = img.size
	expanded = Image.new("RGBA", (w + padding * 2, h + padding * 2), (0, 0, 0, 0))
	expanded.paste(img, (padding, padding))
	return expanded


def _extrude_edges(img: Image.Image, padding: int) -> Image.Image:
	if padding <= 0:
		return img.copy()

	out = _expand_canvas(img, padding)
	pix = out.load()
	w, h = out.size

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	seed_color = [[None for _ in range(w)] for _ in range(h)]
	queue = deque()

	for y in range(h):
		for x in range(w):
			r, g, b, a = pix[x, y]
			if a > 0:
				dist[y][x] = 0
				seed_color[y][x] = (r, g, b, a)
				queue.append((x, y))

	neighbors = (
		(-1, -1), (0, -1), (1, -1),
		(-1,  0),          (1,  0),
		(-1,  1), (0,  1), (1,  1),
	)

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]

		if base_dist >= padding:
			continue

		color = seed_color[y][x]
		for dx, dy in neighbors:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if dist[ny][nx] != -1:
				continue

			new_dist = base_dist + 1
			dist[ny][nx] = new_dist
			seed_color[ny][nx] = color

			r, g, b, a = color
			if SOFT_EXTRUDE_ALPHA:
				falloff = max(0.0, 1.0 - (new_dist / float(padding + 1)))
				alpha = min(a, int(round(SOFT_EXTRUDE_MAX_ALPHA * falloff)))
				pix[nx, ny] = (r, g, b, alpha)
			else:
				pix[nx, ny] = color

			queue.append((nx, ny))

	return out


def process_image(path: Path, out_dir: Path, padding: int):
	original = Image.open(path).convert("RGBA")
	trimmed = _trim_to_alpha_bounds(original)
	trimmed = _apply_edge_blur(trimmed)

	trim_dir = Path(OUT_TRIM_DIR)
	trim_dir.mkdir(parents=True, exist_ok=True)
	trim_path = trim_dir / path.name
	if trim_path.exists() and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {trim_path}")
	trimmed.save(trim_path, "PNG")

	if SAVE_TRIM_ONLY:
		return path.name, original.size, trimmed.size, trim_path, None

	img = _maybe_resize(trimmed)
	processed = _extrude_edges(img, padding)

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / path.name
	if out_path.exists() and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {out_path}")

	processed.save(out_path, "PNG")
	return path.name, original.size, processed.size, trim_path, out_path


def main():
	paths = _collect_images(SOURCE_DIR)
	out_dir = Path(OUT_DIR)

	for path in paths:
		name, original_size, processed_size, trim_path, out_path = process_image(path, out_dir, PADDING)
		if SAVE_TRIM_ONLY:
			print(
				f"{name}: {original_size[0]}x{original_size[1]} -> "
				f"{processed_size[0]}x{processed_size[1]} | trimmed {trim_path}"
			)
		else:
			print(
				f"{name}: {original_size[0]}x{original_size[1]} -> "
				f"{processed_size[0]}x{processed_size[1]} | trimmed {trim_path} | antialias {out_path}"
			)

	if SAVE_TRIM_ONLY:
		print(f"Saved {len(paths)} trimmed image(s) to {OUT_TRIM_DIR}")
	else:
		print(f"Processed {len(paths)} image(s) with PADDING={PADDING}px")


if __name__ == "__main__":
	main()
