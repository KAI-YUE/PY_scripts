import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from PIL import Image

EXTERNAL_OUT_DIR = True
out_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/gemini_uibox/_2_export/ui_pack_gradient_band/"

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/_0_ui_box/test/"
name = "ui_pack"

LAZY_ATLAS_BAKE = True          # True -> run _-1_concate_atlas.py after processing

if not EXTERNAL_OUT_DIR:
	out_dir = os.path.join(root_dir, f"./{name}_gradient_band/")

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DIR = out_dir

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALPHA_THRESHOLD = 1
GRADIENT_BAND_PERCENT = 1

OVERWRITE = True
EDGE_TINT_ENABLED = True

# UI-box edges can touch two different transparent regions:
# exterior-connected transparent pixels are the outer bound, while enclosed
# transparent holes are the inner mask bound.
OUTER_BOUND_TINT_COLOR = (119, 136, 153)   # base environment side
INNER_BOUND_TINT_COLOR = (244, 243, 242)         # inner mask side

EDGE_TINT_STRENGTH = 0.55
INWARD_ALPHA_FLOOR = 96
INWARD_TINT_BIAS = 0.55

COLOR_MODE = "original"     # "original" | "hard_set_color"

GRADIENT_DIRECTION = "inward"     # "inward" | "outward"
OUTWARD_PADDING = 2

BOUND_OUTER = 1
BOUND_INNER = 2

NEIGHBORS = (
	(-1, -1), (0, -1), (1, -1),
	(-1,  0),          (1,  0),
	(-1,  1), (0,  1), (1,  1),
)


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


def _median_channel(values: list[int]) -> int:
	values = sorted(values)
	count = len(values)
	mid = count // 2
	if count % 2:
		return values[mid]
	return int(round((values[mid - 1] + values[mid]) / 2.0))


def _hard_set_sprite_color(img: Image.Image) -> Image.Image:
	pix = img.load()
	w, h = img.size
	red = []
	green = []
	blue = []

	for y in range(h):
		for x in range(w):
			r, g, b, a = pix[x, y]
			if a >= ALPHA_THRESHOLD:
				red.append(r)
				green.append(g)
				blue.append(b)

	if not red:
		return img.copy()

	median_rgb = (
		_median_channel(red),
		_median_channel(green),
		_median_channel(blue),
	)

	out = img.copy()
	out_pix = out.load()
	mr, mg, mb = median_rgb
	for y in range(h):
		for x in range(w):
			_, _, _, a = out_pix[x, y]
			out_pix[x, y] = (mr, mg, mb, a)

	return out


def _prepare_color_mode(img: Image.Image) -> Image.Image:
	if COLOR_MODE == "original":
		return img
	if COLOR_MODE == "hard_set_color":
		return _hard_set_sprite_color(img)
	raise ValueError(f"Unknown COLOR_MODE: {COLOR_MODE}")


def _classify_transparent_regions(img: Image.Image):
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	region = [[-1 for _ in range(w)] for _ in range(h)]
	region_is_outer = []

	def is_transparent(x: int, y: int) -> bool:
		return src[x, y] < ALPHA_THRESHOLD

	for y in range(h):
		for x in range(w):
			if not is_transparent(x, y) or region[y][x] != -1:
				continue

			region_id = len(region_is_outer)
			touches_canvas_edge = x == 0 or y == 0 or x == w - 1 or y == h - 1
			region[y][x] = region_id
			queue = deque([(x, y)])

			while queue:
				cx, cy = queue.popleft()
				if cx == 0 or cy == 0 or cx == w - 1 or cy == h - 1:
					touches_canvas_edge = True

				for dx, dy in NEIGHBORS:
					nx = cx + dx
					ny = cy + dy
					if nx < 0 or ny < 0 or nx >= w or ny >= h:
						continue
					if not is_transparent(nx, ny) or region[ny][nx] != -1:
						continue

					region[ny][nx] = region_id
					queue.append((nx, ny))

			region_is_outer.append(touches_canvas_edge)

	return region, region_is_outer


def _boundary_tint(boundary_kind: int) -> tuple[int, int, int]:
	if boundary_kind == BOUND_INNER:
		return INNER_BOUND_TINT_COLOR
	return OUTER_BOUND_TINT_COLOR


def _transparent_boundary_kind(region: list[list[int]], region_is_outer: list[bool], x: int, y: int) -> int:
	region_id = region[y][x]
	if region_id < 0:
		return BOUND_OUTER
	if region_is_outer[region_id]:
		return BOUND_OUTER
	return BOUND_INNER


def _apply_inward_gradient_band(img: Image.Image) -> tuple[Image.Image, int]:
	alpha = img.getchannel("A")
	src = alpha.load()
	w, h = img.size
	band_px = _band_width_px(img)

	if band_px <= 0:
		return img.copy(), 0

	transparent_region, region_is_outer = _classify_transparent_regions(img)
	dist = [[-1 for _ in range(w)] for _ in range(h)]
	boundary_kind = [[BOUND_OUTER for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= ALPHA_THRESHOLD

	for y in range(h):
		for x in range(w):
			if not is_inside(x, y):
				continue

			kind = None
			for dx, dy in NEIGHBORS:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					kind = BOUND_OUTER
					break
				if not is_inside(nx, ny):
					kind = _transparent_boundary_kind(transparent_region, region_is_outer, nx, ny)
					break

			if kind is not None:
				dist[y][x] = 0
				boundary_kind[y][x] = kind
				queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= band_px - 1:
			continue

		for dx, dy in NEIGHBORS:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if not is_inside(nx, ny) or dist[ny][nx] != -1:
				continue

			dist[ny][nx] = base_dist + 1
			boundary_kind[ny][nx] = boundary_kind[y][x]
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
				tr, tg, tb = _boundary_tint(boundary_kind[y][x])
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

	transparent_region, region_is_outer = _classify_transparent_regions(out)
	dist = [[-1 for _ in range(w)] for _ in range(h)]
	seed_rgb = [[None for _ in range(w)] for _ in range(h)]
	boundary_kind = [[BOUND_OUTER for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= ALPHA_THRESHOLD

	for y in range(h):
		for x in range(w):
			if not is_inside(x, y):
				continue

			kind = None
			for dx, dy in NEIGHBORS:
				nx = x + dx
				ny = y + dy
				if nx < 0 or ny < 0 or nx >= w or ny >= h:
					kind = BOUND_OUTER
					break
				if not is_inside(nx, ny):
					kind = _transparent_boundary_kind(transparent_region, region_is_outer, nx, ny)
					break

			if kind is not None:
				dist[y][x] = 0
				r, g, b, _ = out_pix[x, y]
				seed_rgb[y][x] = (r, g, b)
				boundary_kind[y][x] = kind
				queue.append((x, y))

	while queue:
		x, y = queue.popleft()
		base_dist = dist[y][x]
		if base_dist >= band_px:
			continue

		for dx, dy in NEIGHBORS:
			nx = x + dx
			ny = y + dy
			if nx < 0 or ny < 0 or nx >= w or ny >= h:
				continue
			if is_inside(nx, ny) or dist[ny][nx] != -1:
				continue

			new_dist = base_dist + 1
			dist[ny][nx] = new_dist
			seed_rgb[ny][nx] = seed_rgb[y][x]
			boundary_kind[ny][nx] = boundary_kind[y][x]
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
				tr, tg, tb = _boundary_tint(boundary_kind[y][x])
				r = int(round(r * (1.0 - mix) + tr * mix))
				g = int(round(g * (1.0 - mix) + tg * mix))
				b = int(round(b * (1.0 - mix) + tb * mix))

			out_pix[x, y] = (r, g, b, alpha_value)
			out_alpha[x, y] = alpha_value

	out.putalpha(alpha_out)
	return out, band_px


def process_image(path: Path, out_dir: Path):
	img = Image.open(path).convert("RGBA")
	img = _prepare_color_mode(img)
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


def _run_lazy_ATLAS_bake(out_dir: Path):
	atlas_script = Path(__file__).with_name("_-1_concate_atlas.py")
	if not atlas_script.exists():
		raise FileNotFoundError(f"Atlas bake script not found: {atlas_script}")

	env = os.environ.copy()
	env["ATLAS_ROOT_DIR"] = str(out_dir)
	env["ATLAS_NAME"] = name

	print(f"Lazy atlas bake: {atlas_script}")
	subprocess.run([sys.executable, str(atlas_script)], check=True, env=env)


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
	if LAZY_ATLAS_BAKE:
		_run_lazy_ATLAS_bake(out_dir)


if __name__ == "__main__":
	main()
