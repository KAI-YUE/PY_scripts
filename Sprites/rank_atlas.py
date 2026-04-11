# rank_sheet_builder.py
# Spyder-friendly: edit CONFIG, then Run.
# Requires: pip install pillow

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
font_dir = "/home/kyue/Documents/fonts/"
FONT_DIR = os.path.join(font_dir, "Bearpaw")      # folder containing .ttf/.otf
FONT_FILE = None						         # None = auto-pick first font in folder, or set e.g. "MyFont.ttf"

SUIT_DIR = r"/mnt/ssd/HMeshi/_0_card_design/card_suit_icons/tmp"					# folder with suit icons, e.g. _1_fire.png, _6_diamond.png
OUT_PNG = r"/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/hm/card/ranks/ranks.png"
OUT_JSON = r"/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/hm/card/ranks/ranks.json"

CHARS =  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "X", "V"]			# 12 columns
CELL_W = 20
CELL_H = 20
TEN_GAP = int(-14)
TEN_GAP = int(-3)
CELL_PADDING = 2						# spacing inside each cell around glyph (visual margin)
# FONT_SIZE = 30							# base font size before supersampling
FONT_SIZE = 50
TEN_DENT = 1
SUPERSAMPLE = 2							# 1 = direct render, 2 = smoother downscale
BG_TRANSPARENT = True

# Color sampling from suit icons
MIN_ALPHA_FOR_SAMPLE = 24				# ignore very transparent pixels
IGNORE_DARK_OUTLINES = True				# helps if suit icons have black outlines
DARK_THRESHOLD = 40						# pixels darker than this may be ignored
QUANT_BITS = 4							# color quantization for "dominant" color (4 bits => buckets of 16)
ALPHA_WEIGHTED = True

# Rank styling
ADD_STROKE = False						# optional outline
STROKE_WIDTH = 2						# in final pixels (auto scaled by SUPERSAMPLE)
STROKE_FILL = (0, 0, 0, 255)

# Rendering mode
PIXEL_STYLE = False						# True => NEAREST downscale, False => LANCZOS
CENTER_GLYPH = True

# Optional manual overrides if a sampled color is not what you want
# Keys should match the parsed suit key (e.g. "fire", "diamond")
MANUAL_COLORS = {
	# "diamond": (220, 20, 60, 255),
	"water": (82, 180, 180, 255),
	"smoke": (109, 109, 109, 255),
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

		bbox1 = draw.textbbox((0, 0), "1", font=font, stroke_width=(STROKE_WIDTH * ss if ADD_STROKE else 0))
		bbox0 = draw.textbbox((0, 0), "0", font=font, stroke_width=(STROKE_WIDTH * ss if ADD_STROKE else 0))

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
		if ADD_STROKE:
			kwargs["stroke_width"] = STROKE_WIDTH * ss
			kwargs["stroke_fill"] = STROKE_FILL

		draw.text((tx - bbox1[0], ty - bbox1[1]), "1", **kwargs)
		draw.text((tx + w1 + gap - bbox0[0], ty - bbox0[1]), "0", **kwargs)

		if ss > 1:
			resample = Image.Resampling.NEAREST if PIXEL_STYLE else Image.Resampling.LANCZOS
			cell = cell.resize((cell_w, cell_h), resample=resample)

		return cell

	# Measure glyph bbox
	# Use anchorless bbox and center manually for predictable output.
	bbox = draw.textbbox((0, 0), ch, font=font, stroke_width=(STROKE_WIDTH * ss if ADD_STROKE else 0))
	if bbox is None:
		return cell.resize((cell_w, cell_h), Image.Resampling.NEAREST if PIXEL_STYLE else Image.Resampling.LANCZOS)

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
	if ADD_STROKE:
		kwargs["stroke_width"] = STROKE_WIDTH * ss
		kwargs["stroke_fill"] = STROKE_FILL

	draw.text((tx, ty), ch, **kwargs)

	if ss > 1:
		resample = Image.Resampling.NEAREST if PIXEL_STYLE else Image.Resampling.LANCZOS
		cell = cell.resize((cell_w, cell_h), resample=resample)

	return cell

def build_rank_sheet():
	font_path = _find_font_path(FONT_DIR, FONT_FILE)
	suit_paths = _collect_suits(SUIT_DIR)

	print(f"Using font: {font_path.name}")
	print(f"Found {len(suit_paths)} suit icons")

	ss = max(1, int(SUPERSAMPLE))
	font = _make_font(font_path, FONT_SIZE * ss)

	rows = []
	for sp in suit_paths:
		img = Image.open(sp).convert("RGBA")
		suit_key = _parse_suit_key(sp.name)

		if suit_key in MANUAL_COLORS:
			color = MANUAL_COLORS[suit_key]
		else:
			color = _sample_color_from_icon(img)

		rows.append({ "file": sp.name, "suit_key": suit_key, "color": color })

	n_rows = len(rows)
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
	sheet_h = n_rows * (CELL_H + CELL_PADDING)

	bg = (0, 0, 0, 0) if BG_TRANSPARENT else (255, 255, 255, 255)
	sheet = Image.new("RGBA", (sheet_w, sheet_h), bg)

	# --------------------------------------------------
	# Draw sheet
	# --------------------------------------------------
	for row_i, row in enumerate(rows):
		color = tuple(row["color"])
		y = row_i * (CELL_H + CELL_PADDING)

		for col_i, ch in enumerate(CHARS):
			cell_w = col_widths[col_i]
			cell = _render_glyph_cell(ch, font, color, cell_w, CELL_H)
			if ch == "10" and row_i == 0:
				col_x[col_i] -= TEN_DENT
				
			x = col_x[col_i]
			sheet.alpha_composite(cell, (x, y))

	out_png = Path(OUT_PNG)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	sheet.save(out_png, "PNG")

	# --------------------------------------------------
	# Save metadata for runtime lookup
	# --------------------------------------------------
	meta = {
		"columns": [],
		"rows": [],
		"frames": {}
	}

	# Row metadata + per-frame metadata
	for row_i, row in enumerate(rows):

		for col_i, ch in enumerate(CHARS):
			_s  = row["suit_key"]
			key = "_{:d}_{:s}:{:s}".format(SUIT_INDEX_INFO[_s]["index"], _s, ch)
			meta["frames"][key] = {
				"x": col_x[col_i],
				"y": row_i * (CELL_H + CELL_PADDING),
				"w": col_widths[col_i],
				"h": CELL_H,
				"suit_key": row["suit_key"],
				"char": ch,
				"row": row_i,
				"col": col_i,
			}

	out_json = Path(OUT_JSON)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent = 2)

	print(f"Wrote sheet: { out_png }")
	print(f"Wrote meta : { out_json }")
	print(f"Grid: { n_rows } x { n_cols } (rows x cols)")
	print(f"Variable widths: { col_widths }")
	

if __name__ == "__main__":
	build_rank_sheet()
