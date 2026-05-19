# text_atlas_builder.py
# Spyder-friendly: edit CONFIG, then Run.
# Requires: pip install pillow

import os
import re
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
font_dir = "/home/kyue/Documents/_0_fonts/"
FONT_DIR = os.path.join(font_dir, "kitchen")	# folder containing .ttf/.otf/.ttc
FONT_FILE = None								# None = auto-pick first font in folder

OUT_PNG = r"/home/kyue/Downloads/text_atlas.png"
OUT_JSON = r"/home/kyue/Downloads/text_atlas.json"

CHARS = (
	list("abcdefghijklmnopqrstuvwxyz")
	+ list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	+ list("0123456789")
)

CELL_W = 196
CELL_H = 196
CELL_GAP = 2					# small gap prevents atlas bleeding
CELL_PADDING = 8

COLS = 13						# 65 glyphs -> 5 clean rows
FONT_SIZE = 172

SUPERSAMPLE = 3					# smoother downscale
BG_TRANSPARENT = False
BG_COLOR = (0, 102, 51, 255)
GLYPH_COLOR = (255, 255, 255, 255)

CENTER_GLYPH = True

PRE_BLUR_ENABLED = False
PRE_BLUR_RADIUS = 0.65
PRE_BLUR_KEEP_SOLID_COLOR = True

AUTO_FIT_FONT = True			# shrink font only if a glyph does not fit


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


def _make_font(font_path: Path, px_size: int):
	return ImageFont.truetype(str(font_path), px_size)


def _measure_char(ch: str, font):
	dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
	draw = ImageDraw.Draw(dummy)
	bbox = draw.textbbox((0, 0), ch, font=font)
	if bbox is None:
		return 0, 0, (0, 0, 0, 0)

	w = bbox[2] - bbox[0]
	h = bbox[3] - bbox[1]
	return w, h, bbox


def _find_fitted_font_size(font_path: Path):
	if not AUTO_FIT_FONT:
		return FONT_SIZE

	ss = max(1, int(SUPERSAMPLE))
	max_w = (CELL_W - CELL_PADDING * 2) * ss
	max_h = (CELL_H - CELL_PADDING * 2) * ss

	size = FONT_SIZE * ss

	while size > 4:
		font = _make_font(font_path, size)
		ok = True

		for ch in CHARS:
			w, h, _ = _measure_char(ch, font)
			if w > max_w or h > max_h:
				ok = False
				break

		if ok:
			return max(1, size // ss)

		size -= ss

	return 4


def _apply_pre_blur(img: Image.Image, fill_rgba=None):
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


def _char_key(ch: str):
	if ch == "(":
		return "open_paren"
	if ch == ")":
		return "close_paren"
	if ch == "#":
		return "hash"
	return ch


def _render_glyph_cell(ch: str, font, color_rgba):
	ss = max(1, int(SUPERSAMPLE))

	W = CELL_W * ss
	H = CELL_H * ss
	pad = CELL_PADDING * ss

	cell = Image.new("RGBA", (W, H), (0, 0, 0, 0))
	draw = ImageDraw.Draw(cell)

	bbox = draw.textbbox((0, 0), ch, font=font)

	if bbox is None:
		if ss > 1:
			cell = cell.resize((CELL_W, CELL_H), resample=Image.Resampling.LANCZOS)
		return _apply_pre_blur(cell, fill_rgba=color_rgba)

	bw = bbox[2] - bbox[0]
	bh = bbox[3] - bbox[1]

	if CENTER_GLYPH:
		tx = (W - bw) * 0.5 - bbox[0]
		ty = (H - bh) * 0.5 - bbox[1]
	else:
		tx = pad - bbox[0]
		ty = pad - bbox[1]

	draw.text(
		(tx, ty),
		ch,
		fill=color_rgba,
		font=font,
	)

	if ss > 1:
		cell = cell.resize((CELL_W, CELL_H), resample=Image.Resampling.LANCZOS)

	cell = _apply_pre_blur(cell, fill_rgba=color_rgba)
	return cell


def build_text_atlas():
	font_path = _find_font_path(FONT_DIR, FONT_FILE)
	fitted_font_size = _find_fitted_font_size(font_path)

	print(f"Using font: {font_path.name}")
	print(f"Font size: {fitted_font_size}")
	print(f"Glyph count: {len(CHARS)}")

	ss = max(1, int(SUPERSAMPLE))
	font = _make_font(font_path, fitted_font_size * ss)

	n_chars = len(CHARS)
	n_cols = int(COLS)
	n_rows = int(math.ceil(n_chars / n_cols))

	sheet_w = n_cols * CELL_W + (n_cols - 1) * CELL_GAP
	sheet_h = n_rows * CELL_H + (n_rows - 1) * CELL_GAP

	bg = (0, 0, 0, 0) if BG_TRANSPARENT else BG_COLOR
	sheet = Image.new("RGBA", (sheet_w, sheet_h), bg)

	meta = {
		"atlas_color": list(GLYPH_COLOR),
		"background_color": list(bg),
		"cell_w": CELL_W,
		"cell_h": CELL_H,
		"cell_gap": CELL_GAP,
		"cols": n_cols,
		"rows": n_rows,
		"chars": CHARS,
		"frames": {},
		"aliases": {},
	}

	for i, ch in enumerate(CHARS):
		row = i // n_cols
		col = i % n_cols

		x = col * (CELL_W + CELL_GAP)
		y = row * (CELL_H + CELL_GAP)

		cell = _render_glyph_cell(ch, font, GLYPH_COLOR)
		sheet.alpha_composite(cell, (x, y))

		frame = {
			"x": x,
			"y": y,
			"w": CELL_W,
			"h": CELL_H,
			"char": ch,
			"key": _char_key(ch),
			"row": row,
			"col": col,
		}

		meta["frames"][ch] = frame

		alias = _char_key(ch)
		if alias != ch:
			meta["aliases"][alias] = ch

	out_png = Path(OUT_PNG)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	sheet.save(out_png, "PNG")

	out_json = Path(OUT_JSON)
	out_json.parent.mkdir(parents=True, exist_ok=True)

	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"Wrote sheet: {out_png}")
	print(f"Wrote meta : {out_json}")
	print(f"Grid: {n_rows} x {n_cols} rows x cols")
	print(f"Atlas size: {sheet_w} x {sheet_h}")


if __name__ == "__main__":
	build_text_atlas()
