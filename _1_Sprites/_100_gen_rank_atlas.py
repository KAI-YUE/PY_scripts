# rank_sheet_builder.py
# Spyder-friendly: edit CONFIG, then Run.
# Requires: pip install pillow

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from collections import deque
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
font_dir  = "/home/kyue/Documents/fonts/"
FONT_DIR  = os.path.join(font_dir, "Bearpaw")      # folder containing .ttf/.otf
FONT_FILE = None						         # None = auto-pick first font in folder, or set e.g. "MyFont.ttf"

SUIT_DIR = r"/mnt/ssd/HMeshi/_0_card_design/_2_card_suit_icons/suits/10suits/"
# SUIT_DIR = r"/mnt/ssd/HMeshi/_0_card_design/_2_card_suit_icons/suits/test"					# folder with suit icons, e.g. _1_fire.png, _6_diamond.png
OUT_PNG = r"/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/hm/card/ranks/ranks.png"
OUT_JSON = r"/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/hm/card/ranks/ranks.json"

CHARS =  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "X", "V"]			# 12 columns
CELL_W = 150
CELL_H = 150
TEN_GAP = int(-3)
CELL_PADDING = 1						 # spacing inside each cell around glyph (visual margin)
# FONT_SIZE = 30							# base font size before supersampling
FONT_SIZE = 300
TEN_DENT = 1
SUPERSAMPLE = 2							# 1 = direct render, 2 = smoother downscale
BG_TRANSPARENT = True
ATLAS_GLYPH_COLOR = (255, 255, 255, 255)	# neutral atlas color; tint in LOVE at runtime

# Color sampling from suit icons
MIN_ALPHA_FOR_SAMPLE = 24				# ignore very transparent pixels
IGNORE_DARK_OUTLINES = True				# helps if suit icons have black outlines
DARK_THRESHOLD = 40						# pixels darker than this may be ignored
QUANT_BITS = 1							# color quantization for "dominant" color (4 bits => buckets of 16)
ALPHA_WEIGHTED = True

CENTER_GLYPH = True

PRE_BLUR_ENABLED = True
PRE_BLUR_RADIUS = 1					# radius in final pixels; internally scaled by SUPERSAMPLE
PRE_BLUR_KEEP_SOLID_COLOR = True		# blur alpha, then repaint glyph with one flat color

# Protective edge / gradient band
PROTECTIVE_EDGE_ENABLED = True
# PROTECTIVE_EDGE_ENABLED = False
EDGE_DIRECTION = "outward"				# "outward" | "inward"
EDGE_BAND_PERCENT = 3.0					# percent of glyph short edge
EDGE_ALPHA_THRESHOLD = 1
OUTWARD_CORE_ALPHA_THRESHOLD = 96		# harden antialiased glyph edge before outward fade

OUTWARD_FADE_TARGET_COLOR = (255, 255, 255, 0)	# RGB target that outward band fades toward

# Optional manual overrides if a sampled color is not what you want
# Keys should match the parsed suit key (e.g. "fire", "diamond")
MANUAL_COLORS = {
	# "diamond": (220, 20, 60, 255),
	"water": (82, 180, 180, 255),
	# "smoke": (109, 109, 109, 255),
}

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SUIT_INDEX_INFO = {}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def _natural_key(s: str):
	return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _find_font_path(font_dir: str, font_file: str | None):
	fd = Path(font_dir)
	if not fd.exists():
		raise FileNotFoundError(f"FONT_DIR not found: {font_dir}")

	if font_file:
		fp = fd / font_file
		if not fp.exists():
			raise FileNotFoundError(f"FONT_FILE not found: {fp}")
		return fp

	cands = []
	for p in fd.iterdir():
		if p.is_file() and p.suffix.lower() in {".ttf", ".otf", ".ttc"}:
			cands.append(p)

	if not cands:
		raise RuntimeError(f"No font files (.ttf/.otf/.ttc) found in {font_dir}")

	cands.sort(key=lambda p: _natural_key(p.name))
	return cands[0]


def _collect_suits(suit_dir: str):
	sd = Path(suit_dir)
	if not sd.exists():
		raise FileNotFoundError(f"SUIT_DIR not found: {suit_dir}")

	items = []
	for p in sd.iterdir():
		if p.is_file() and p.suffix.lower() in INCLUDE_EXTS:
			items.append(p)

	if not items:
		raise RuntimeError(f"No suit images found in {suit_dir}")

	items.sort(key = lambda p: _natural_key(p.name))
	return items


def _parse_suit_key(filename: str):
	# Examples:
	# "_1_fire.png" -> "fire"   (stores index=1)
	# "_6_diamond.png" -> "diamond" (stores index=6)
	# "spade.png" -> "spade"   (stores index=None)

	stem = Path(filename).stem
	m = re.match(r"^_?(?P<idx>\d+)_(?P<key>.+)$", stem)

	if m:
		idx = int(m.group("idx"))
		key = m.group("key")

		SUIT_INDEX_INFO[key] = {
			"index": idx,
			"stem": stem,
			"filename": filename,
		}
		return key

	# No numeric prefix found
	SUIT_INDEX_INFO[stem] = {
		"index": None,
		"stem": stem,
		"filename": filename,
	}
	return stem


def _sample_color_from_icon(img: Image.Image):
	"""
	Robust-ish color sampling:
	- ignores transparent pixels
	- optionally ignores very dark outline pixels
	- picks dominant quantized color bucket
	- returns RGBA
	"""
	rgba = img.convert("RGBA")
	w, h = rgba.size
	pix = rgba.load()

	# bucket -> [weight_sum, r_sum, g_sum, b_sum, a_sum]
	buckets = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
	fallback = [0.0, 0.0, 0.0, 0.0, 0.0]

	shift = max(0, 8 - QUANT_BITS)

	for y in range(h):
		for x in range(w):
			r, g, b, a = pix[x, y]
			if a < MIN_ALPHA_FOR_SAMPLE:
				continue

			# fallback accumulates all visible pixels
			wgt_all = (a / 255.0) if ALPHA_WEIGHTED else 1.0
			fallback[0] += wgt_all
			fallback[1] += r * wgt_all
			fallback[2] += g * wgt_all
			fallback[3] += b * wgt_all
			fallback[4] += a * wgt_all

			if IGNORE_DARK_OUTLINES and max(r, g, b) < DARK_THRESHOLD:
				continue

			key = (r >> shift, g >> shift, b >> shift)
			wgt = (a / 255.0) if ALPHA_WEIGHTED else 1.0
			bk = buckets[key]
			bk[0] += wgt
			bk[1] += r * wgt
			bk[2] += g * wgt
			bk[3] += b * wgt
			bk[4] += a * wgt

	# Prefer dominant non-dark bucket
	if buckets:
		best_key, best = max(buckets.items(), key=lambda kv: kv[1][0])
		wgt = max(best[0], 1e-6)
		r = int(round(best[1] / wgt))
		g = int(round(best[2] / wgt))
		b = int(round(best[3] / wgt))
		a = int(round(best[4] / wgt))
		return (r, g, b, a)

	# Fallback if icon is too dark / filtered out
	if fallback[0] > 0:
		wgt = max(fallback[0], 1e-6)
		r = int(round(fallback[1] / wgt))
		g = int(round(fallback[2] / wgt))
		b = int(round(fallback[3] / wgt))
		a = int(round(fallback[4] / wgt))
		return (r, g, b, a)

	# Last resort
	return (255, 255, 255, 255)

def _make_font(font_path: Path, px_size: int):
	return ImageFont.truetype(str(font_path), px_size)

def _cell_width_for_char(ch: str, base_w: int) -> int:
	if ch == "10":
		return int(base_w * 1.2)
	return base_w


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
			if pix[x, y] < EDGE_ALPHA_THRESHOLD:
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


def _band_width_px(img: Image.Image) -> int:
	min_x, min_y, max_x, max_y = _alpha_bounds(img)
	if max_x < 0 or max_y < 0:
		return 0

	content_w = max_x - min_x + 1
	content_h = max_y - min_y + 1
	short_edge = min(content_w, content_h)
	return max(1, int(round(short_edge * EDGE_BAND_PERCENT / 100.0)))


def _harden_alpha(img: Image.Image, threshold: int) -> Image.Image:
	w, h = img.size
	src = img.convert("RGBA")
	src_pix = src.load()
	out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
	out_pix = out.load()

	for y in range(h):
		for x in range(w):
			r, g, b, a = src_pix[x, y]
			if a >= threshold:
				out_pix[x, y] = (r, g, b, 255)

	return out


def _apply_pre_blur(img: Image.Image, fill_rgba = None) -> Image.Image:
	if not PRE_BLUR_ENABLED:
		return img

	ss = max(1, int(SUPERSAMPLE))
	radius = max(0.0, float(PRE_BLUR_RADIUS)) * ss
	if radius <= 0.0:
		return img

	blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
	if not PRE_BLUR_KEEP_SOLID_COLOR or fill_rgba is None:
		return blurred

	alpha = blurred.getchannel("A")
	out = Image.new("RGBA", blurred.size, (0, 0, 0, 0))
	solid = (
		int(fill_rgba[0]),
		int(fill_rgba[1]),
		int(fill_rgba[2]),
		255,
	)
	out.paste(solid, (0, 0), alpha)
	return out


def _apply_inward_gradient_band(img: Image.Image) -> tuple[Image.Image, int]:
	alpha = img.getchannel("A")
	w, h = img.size
	band_px = _band_width_px(img)

	if band_px <= 0:
		return img.copy(), 0

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= EDGE_ALPHA_THRESHOLD

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
				if nx < 0 or ny < 0 or nx >= w or ny >= h or not is_inside(nx, ny):
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
	alpha_out = out.getchannel("A")
	out_alpha = alpha_out.load()

	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d == -1:
				continue

			falloff = float(d + 1) / float(band_px)
			target_alpha = max(0, min(255, int(round(255 * falloff))))
			out_alpha[x, y] = min(out_alpha[x, y], target_alpha)

	out.putalpha(alpha_out)
	return out, band_px


def _apply_outward_gradient_band(img: Image.Image) -> tuple[Image.Image, int]:
	core = _harden_alpha(img, OUTWARD_CORE_ALPHA_THRESHOLD)
	band_px = _band_width_px(core)
	if band_px <= 0:
		return img.copy(), 0

	alpha = core.getchannel("A")
	src = alpha.load()
	out = core.copy()
	out_pix = out.load()
	alpha_out = alpha.copy()
	out_alpha = alpha_out.load()
	w, h = out.size

	dist = [[-1 for _ in range(w)] for _ in range(h)]
	queue = deque()

	def is_inside(x: int, y: int) -> bool:
		return src[x, y] >= EDGE_ALPHA_THRESHOLD

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
				if nx < 0 or ny < 0 or nx >= w or ny >= h or not is_inside(nx, ny):
					is_boundary = True
					break

			if is_boundary:
				dist[y][x] = 0
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

			dist[ny][nx] = base_dist + 1
			queue.append((nx, ny))

	sr, sg, sb = ATLAS_GLYPH_COLOR[:3]
	tr, tg, tb = OUTWARD_FADE_TARGET_COLOR[:3]
	for y in range(h):
		for x in range(w):
			d = dist[y][x]
			if d <= 0 or d > band_px:
				continue

			falloff = max(0.0, 1.0 - (d / float(band_px + 1)))
			inv = 1.0 - falloff
			r = max(0, min(255, int(round(sr * falloff + tr * inv))))
			g = max(0, min(255, int(round(sg * falloff + tg * inv))))
			b = max(0, min(255, int(round(sb * falloff + tb * inv))))
			alpha_value = max(0, min(255, int(round(255 * falloff))))

			if alpha_value > out_alpha[x, y]:
				out_pix[x, y] = (r, g, b, alpha_value)
				out_alpha[x, y] = alpha_value

	out.putalpha(alpha_out)
	return out, band_px


def _apply_protective_edge(img: Image.Image) -> Image.Image:
	if not PROTECTIVE_EDGE_ENABLED:
		return img
	if EDGE_DIRECTION == "inward":
		out, _ = _apply_inward_gradient_band(img)
		return out
	if EDGE_DIRECTION == "outward":
		out, _ = _apply_outward_gradient_band(img)
		return out
	raise ValueError(f"Unknown EDGE_DIRECTION: {EDGE_DIRECTION}")


def _render_glyph_cell(ch: str, font, color_rgba, cell_w: int, cell_h: int):
	ss = max(1, int(SUPERSAMPLE))
	cell_w = _cell_width_for_char(ch, cell_w)

	W = cell_w * ss
	H = cell_h * ss
	pad = CELL_PADDING * ss

	cell = Image.new("RGBA", (W, H), (0, 0, 0, 0))
	draw = ImageDraw.Draw(cell)
	
	if ch == "10":
		gap = TEN_GAP	# tighten spacing; tweak: -1, -2, -3

		bbox1 = draw.textbbox((0, 0), "1", font=font)
		bbox0 = draw.textbbox((0, 0), "0", font=font)

		w1 = bbox1[2] - bbox1[0]
		h1 = bbox1[3] - bbox1[1]
		w0 = bbox0[2] - bbox0[0]
		h0 = bbox0[3] - bbox0[1]

		pair_w = w1 + gap + w0
		pair_h = max(h1, h0)

		if CENTER_GLYPH:
			tx = (W - pair_w) * 0.5
			ty = (H - pair_h) * 0.5
		else:
			tx = pad
			ty = pad

		kwargs = {
			"fill": color_rgba,
			"font": font,
		}

		draw.text((tx - bbox1[0], ty - bbox1[1]), "1", **kwargs)
		draw.text((tx + w1 + gap - bbox0[0], ty - bbox0[1]), "0", **kwargs)

		if ss > 1:
			cell = cell.resize((cell_w, cell_h), resample=Image.Resampling.LANCZOS)

		cell = _apply_pre_blur(cell, fill_rgba=color_rgba)
		return _apply_protective_edge(cell)

	# Measure glyph bbox
	# Use anchorless bbox and center manually for predictable output.
	bbox = draw.textbbox((0, 0), ch, font=font)
	if bbox is None:
		cell = cell.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
		cell = _apply_pre_blur(cell, fill_rgba=color_rgba)
		return _apply_protective_edge(cell)

	bw = bbox[2] - bbox[0]
	bh = bbox[3] - bbox[1]

	if CENTER_GLYPH:
		tx = (W - bw) * 0.5 - bbox[0]
		ty = (H - bh) * 0.5 - bbox[1]
	else:
		tx = pad - bbox[0]
		ty = pad - bbox[1]

	kwargs = {
		"fill": color_rgba,
		"font": font,
	}

	draw.text((tx, ty), ch, **kwargs)

	if ss > 1:
		cell = cell.resize((cell_w, cell_h), resample=Image.Resampling.LANCZOS)

	cell = _apply_pre_blur(cell, fill_rgba=color_rgba)
	return _apply_protective_edge(cell)

def build_rank_sheet():
	font_path = _find_font_path(FONT_DIR, FONT_FILE)
	suit_paths = _collect_suits(SUIT_DIR)

	print(f"Using font: {font_path.name}")
	print(f"Found {len(suit_paths)} suit icons")

	ss = max(1, int(SUPERSAMPLE))
	font = _make_font(font_path, FONT_SIZE * ss)

	suits = []
	for sp in suit_paths:
		img = Image.open(sp).convert("RGBA")
		suit_key = _parse_suit_key(sp.name)

		if suit_key in MANUAL_COLORS:
			color = MANUAL_COLORS[suit_key]
		else:
			color = _sample_color_from_icon(img)

		suits.append({ "file": sp.name, "suit_key": suit_key, "color": color })

	n_suits = len(suits)
	n_cols = len(CHARS)

	# --------------------------------------------------
	# Variable-width columns (e.g. "10" is wider)
	# --------------------------------------------------
	col_widths = [_cell_width_for_char(ch, CELL_W) for ch in CHARS]

	col_x = []
	x_acc = 0
	for w in col_widths:
		col_x.append(x_acc)
		x_acc += w + CELL_PADDING

	sheet_w = x_acc
	sheet_h = CELL_H

	bg = (0, 0, 0, 0) if BG_TRANSPARENT else (255, 255, 255, 255)
	sheet = Image.new("RGBA", (sheet_w, sheet_h), bg)

	# --------------------------------------------------
	# Draw sheet
	# --------------------------------------------------
	render_color = tuple(ATLAS_GLYPH_COLOR)
	y = 0
	for col_i, ch in enumerate(CHARS):
		cell_w = col_widths[col_i]
		cell = _render_glyph_cell(ch, font, render_color, cell_w, CELL_H)
		if ch == "10":
			col_x[col_i] -= TEN_DENT
		if ch == "X":
			col_x[col_i] += 20*TEN_DENT

		x = col_x[col_i]
		sheet.alpha_composite(cell, (x, y))

	out_png = Path(OUT_PNG)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	sheet.save(out_png, "PNG")

	# --------------------------------------------------
	# Save metadata for runtime lookup
	# --------------------------------------------------
	meta = {
		"atlas_color": list(ATLAS_GLYPH_COLOR),
		"columns": [],
		"rows": [],
		"suits": {},
		"frames": {}
	}

	for suit in suits:
		suit_key = suit["suit_key"]
		suit_info = SUIT_INDEX_INFO[suit_key]
		meta["suits"][suit_key] = {
			"index": suit_info["index"],
			# "filename": suit_info["filename"],
			"color": list(suit["color"]),
		}

	for col_i, ch in enumerate(CHARS):
		meta["frames"][ch] = {
			"x": col_x[col_i],
			"y": 0,
			"w": col_widths[col_i],
			"h": CELL_H,
			"char": ch,
			"row": 0,
			"col": col_i,
		}

	out_json = Path(OUT_JSON)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent = 2)

	print(f"Wrote sheet: { out_png }")
	print(f"Wrote meta : { out_json }")
	print(f"Grid: 1 x { n_cols } (rows x cols)")
	print(f"Suits: { n_suits } stored in JSON")
	print(f"Variable widths: { col_widths }")
	

if __name__ == "__main__":
	build_rank_sheet()
