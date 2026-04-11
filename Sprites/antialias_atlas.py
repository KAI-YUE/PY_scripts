import os
from collections import deque
from pathlib import Path

from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/"
name = "cards"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_TRIM_DIR = os.path.join(root_dir, f"./{name}_trimmed/")
OUT_DIR = os.path.join(root_dir, f"./{name}_antialias/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PADDING = 4
TARGET_LONG_EDGE = None
TRIM_ALPHA = True
ALPHA_THRESHOLD = 1
ALPHA_HARD_TRIM = True
ALPHA_HARD_TRIM_PERCENT = 20.0
TRIM_KEEP = 0
SAVE_TRIM_ONLY = True
OVERWRITE = True


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

	if ALPHA_HARD_TRIM:
		min_x = 0
		while min_x < w and alpha_percent_for_col(min_x) <= ALPHA_HARD_TRIM_PERCENT:
			min_x += 1

		max_x = w - 1
		while max_x >= min_x and alpha_percent_for_col(max_x) <= ALPHA_HARD_TRIM_PERCENT:
			max_x -= 1

		min_y = 0
		while min_y < h and alpha_percent_for_row(min_y) <= ALPHA_HARD_TRIM_PERCENT:
			min_y += 1

		max_y = h - 1
		while max_y >= min_y and alpha_percent_for_row(max_y) <= ALPHA_HARD_TRIM_PERCENT:
			max_y -= 1
	else:
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

	if max_x < 0 or max_y < 0:
		return img.copy()

	left = max(0, min_x - TRIM_KEEP)
	top = max(0, min_y - TRIM_KEEP)
	right = min(w, max_x + 1 + TRIM_KEEP)
	bottom = min(h, max_y + 1 + TRIM_KEEP)
	return img.crop((left, top, right, bottom))


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

			dist[ny][nx] = base_dist + 1
			seed_color[ny][nx] = color
			pix[nx, ny] = color
			queue.append((nx, ny))

	return out


def process_image(path: Path, out_dir: Path, padding: int):
	original = Image.open(path).convert("RGBA")
	trimmed = _trim_to_alpha_bounds(original)

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
