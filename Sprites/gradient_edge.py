import os
from collections import deque
from pathlib import Path

from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/colorful_design/"
root_dir = "/mnt/ssd/HMeshi/-1_field_landscape/rocks_stuff/export/"
name = "cards"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DIR = os.path.join(root_dir, f"./{name}_gradient_band/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALPHA_THRESHOLD = 1
# GRADIENT_BAND_PERCENT = 2
GRADIENT_BAND_PERCENT = 1
OVERWRITE = True
EDGE_TINT_ENABLED = True
EDGE_TINT_COLOR = (200, 210, 230)

EDGE_TINT_STRENGTH = 0.55
INWARD_ALPHA_FLOOR = 96
INWARD_TINT_BIAS = 0.55
GRADIENT_DIRECTION = "inward"   # "inward" | "outward"
# GRADIENT_DIRECTION = "outward" 
OUTWARD_PADDING = 2


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
			if pix[x, y] >= ALPHA_THRESHOLD:
				if x < min_x:
					min_x = x
				if y < min_y:
					min_y = y
				if x > max_x:
					max_x = x
				if y > max_y:
					max_y = y

	return min_x, min_y, max_x, max_y


def _band_width_px(img: Image.Image) -> int:
	min_x, min_y, max_x, max_y = _alpha_bounds(img)
	if max_x < 0 or max_y < 0:
		return 0

	content_w = max_x - min_x + 1
	content_h = max_y - min_y + 1
	short_edge = min(content_w, content_h)
	return max(1, int(round(short_edge * GRADIENT_BAND_PERCENT / 100.0)))


def _expand_canvas(img: Image.Image, padding: int) -> Image.Image:
	if padding <= 0:
		return img.copy()

	w, h = img.size
	out = Image.new("RGBA", (w + padding * 2, h + padding * 2), (0, 0, 0, 0))
	out.paste(img, (padding, padding))
	return out


def _apply_inward_gradient_band(img: Image.Image) -> tuple[Image.Image, int]:
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	band_px = _band_width_px(img)

	if band_px <= 0:
		return img.copy(), 0

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= ALPHA_THRESHOLD

	neighbors = (
		(-1, -1), (0, -1), (1, -1),
		(-1,  0),          (1,  0),
		(-1,  1), (0,  1), (1,  1),
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

			if EDGE_TINT_ENABLED:
				edge_mix = max(0.0, min(1.0, 1.0 - (d / float(max(1, band_px - 1)))))
				edge_mix = max(0.0, min(1.0, edge_mix + INWARD_TINT_BIAS * (1.0 - edge_mix)))
				mix = EDGE_TINT_STRENGTH * edge_mix
				r, g, b, a = out_pix[x, y]
				tr, tg, tb = EDGE_TINT_COLOR
				out_pix[x, y] = (
					int(round(r * (1.0 - mix) + tr * mix)),
					int(round(g * (1.0 - mix) + tg * mix)),
					int(round(b * (1.0 - mix) + tb * mix)),
					a,
				)

	out.putalpha(alpha_out)
	return out, band_px


def _apply_outward_gradient_band(img: Image.Image) -> tuple[Image.Image, int]:
	band_px = _band_width_px(img)
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
		(-1,  0),          (1,  0),
		(-1,  1), (0,  1), (1,  1),
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

			new_dist = base_dist + 1
			dist[ny][nx] = new_dist
			seed_rgb[ny][nx] = seed_rgb[y][x]
			queue.append((nx, ny))

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d <= 0 or d > band_px:
				continue

			r, g, b = seed_rgb[y][x]
			falloff = max(0.0, 1.0 - (d / float(band_px + 1)))
			alpha_value = max(0, min(255, int(round(255 * falloff))))

			if EDGE_TINT_ENABLED:
				mix = EDGE_TINT_STRENGTH * falloff
				tr, tg, tb = EDGE_TINT_COLOR
				r = int(round(r * (1.0 - mix) + tr * mix))
				g = int(round(g * (1.0 - mix) + tg * mix))
				b = int(round(b * (1.0 - mix) + tb * mix))

			out_pix[x, y] = (r, g, b, alpha_value)
			out_alpha[x, y] = alpha_value

	out.putalpha(alpha_out)
	return out, band_px


def process_image(path: Path, out_dir: Path):
	img = Image.open(path).convert("RGBA")
	if GRADIENT_DIRECTION == "inward":
		processed, band_px = _apply_inward_gradient_band(img)
	elif GRADIENT_DIRECTION == "outward":
		processed, band_px = _apply_outward_gradient_band(img)
	else:
		raise ValueError(f"Unknown GRADIENT_DIRECTION: {GRADIENT_DIRECTION}")

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / path.name
	if out_path.exists() and not OVERWRITE:
		raise FileExistsError(f"Output already exists: {out_path}")

	processed.save(out_path, "PNG")
	return path.name, img.size, band_px, out_path


def main():
	paths = _collect_images(SOURCE_DIR)
	out_dir = Path(OUT_DIR)

	for path in paths:
		name, size, band_px, out_path = process_image(path, out_dir)
		print(
			f"{name}: {size[0]}x{size[1]} | "
			f"band={band_px}px ({GRADIENT_BAND_PERCENT:.2f}%) | {out_path}"
		)

	print(f"Processed {len(paths)} image(s) to {OUT_DIR}")


if __name__ == "__main__":
	main()
