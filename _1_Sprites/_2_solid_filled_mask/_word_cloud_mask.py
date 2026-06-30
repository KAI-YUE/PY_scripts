# word_cloud_mask.py

import os
from collections import deque
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter


# ----------------------------
# CONFIG (edit these)
# ----------------------------
SOURCE_DIR = os.environ.get(
	"WORD_GLYPH_SOURCE_DIR",
	"/mnt/ssd/HMeshi/_2_UI_Uten/_4_gampad_btns/_-1_prev/_0_title_page/_00_export",
)
OUTPUT_DIR = os.environ.get("WORD_MASK_OUTPUT_DIR", os.path.join(SOURCE_DIR, "_word_cloud_mask"))
OUTPUT_PREFIX = os.environ.get("WORD_OUTPUT_PREFIX", "_pr_")

WORDS = {
	"press": ["_pr_eng0_p.png", "_pr_eng1_r.png", "_pr_eng2_e.png", "_pr_eng3_s.png", "_pr_eng3_s_2.png"],
	"any": ["_pr_eng4_A.png", "_pr_eng5_n.png", "_pr_eng6_y.png"],
	"button": ["_pr_eng7_B.png", "_pr_eng8_u.png", "_pr_eng9_t.png", "_pr_eng10_t.png", "_pr_eng11_o.png", "_pr_eng12_n.png"],
}

ALPHA_THRESHOLD = int(os.environ.get("WORD_ALPHA_THRESHOLD", "8"))
LETTER_TRACKING = int(os.environ.get("WORD_LETTER_TRACKING", "-5"))
CANVAS_PAD = int(os.environ.get("WORD_CANVAS_PAD", "18"))
LETTER_GROW_PX = {
	"u": 6,
	"n": 6,
	"t": 4,
	"r": 4,
	"y": 4,
	"a": 6,
}
CLOSE_GAP_PX = int(os.environ.get("WORD_CLOSE_GAP_PX", "7"))
EXPAND_PX = int(os.environ.get("WORD_EXPAND_PX", "4"))
SOFT_BLUR = float(os.environ.get("WORD_SOFT_BLUR", "2.8"))
SOFT_ALPHA = float(os.environ.get("WORD_SOFT_ALPHA", "0.85"))
FINAL_BLUR = float(os.environ.get("WORD_FINAL_BLUR", "0.8"))
MASK_RGBA = tuple(int(v) for v in os.environ.get("WORD_MASK_RGBA", "255,255,255,255").split(","))
OVERWRITE = True


# ----------------------------
# HELPERS
# ----------------------------
def _alpha_mask(img: Image.Image) -> Image.Image:
	return img.getchannel("A").point(lambda a: 255 if a > ALPHA_THRESHOLD else 0, "L")


def _grow(mask: Image.Image, px: int) -> Image.Image:
	if px <= 0:
		return mask
	return mask.filter(ImageFilter.MaxFilter(px * 2 + 1))


def _close(mask: Image.Image, px: int) -> Image.Image:
	grown = _grow(mask, px)
	if px <= 0:
		return grown
	return grown.filter(ImageFilter.MinFilter(px * 2 + 1))


def _outside_transparent(mask: Image.Image) -> Image.Image:
	w, h = mask.size
	src = mask.load()
	outside = Image.new("L", (w, h), 0)
	dst = outside.load()
	q = deque()

	def push(x: int, y: int) -> None:
		if src[x, y] == 0 and dst[x, y] == 0:
			dst[x, y] = 255; q.append((x, y))

	for x in range(w):
		push(x, 0); push(x, h - 1)
	for y in range(h):
		push(0, y); push(w - 1, y)

	while q:
		x, y = q.popleft()
		for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
			if 0 <= nx < w and 0 <= ny < h:
				push(nx, ny)

	return outside


def _fill_holes(mask: Image.Image) -> Image.Image:
	outside = _outside_transparent(mask)
	transparent = mask.point(lambda v: 0 if v else 255, "L")
	holes = ImageChops.multiply(ImageChops.invert(outside), transparent)
	return ImageChops.lighter(mask, holes)


def _soften(mask: Image.Image) -> Image.Image:
	mask = _grow(mask, EXPAND_PX)
	if SOFT_BLUR > 0:
		halo = mask.filter(ImageFilter.GaussianBlur(SOFT_BLUR))
		halo = halo.point(lambda a: min(255, int(a * SOFT_ALPHA)), "L")
		mask = ImageChops.lighter(mask, halo)
	if FINAL_BLUR > 0:
		mask = mask.filter(ImageFilter.GaussianBlur(FINAL_BLUR))
	return mask


def _tint(mask: Image.Image) -> Image.Image:
	color = Image.new("RGBA", mask.size, MASK_RGBA)
	color.putalpha(mask.point(lambda a: int(a * MASK_RGBA[3] / 255), "L"))
	return color


def _glyph_name(path: Path) -> str:
	return path.stem.rsplit("_", 1)[-1].lower()


def _letter_grow_px(path: Path) -> int:
	return LETTER_GROW_PX.get(_glyph_name(path), 0)


def _load_letter(path: Path) -> tuple[Image.Image, tuple[int, int]]:
	img = Image.open(path).convert("RGBA")
	mask = _grow(_alpha_mask(img), _letter_grow_px(path))
	bbox = mask.getbbox()
	if bbox is None:
		raise RuntimeError(f"Empty glyph alpha: {path}")
	left, top, right, bottom = bbox
	return mask.crop(bbox), (left, top)


def _word_mask(source_dir: Path, files: list[str]) -> Image.Image:
	letters = [_load_letter(source_dir / name) for name in files]
	width = CANVAS_PAD * 2 + sum(mask.size[0] for mask, _ in letters) + LETTER_TRACKING * (len(letters) - 1)
	height = CANVAS_PAD * 2 + max(top + mask.size[1] for mask, (_, top) in letters)
	out = Image.new("L", (width, height), 0)
	x = CANVAS_PAD

	for mask, (_, top) in letters:
		out.paste(mask, (x, CANVAS_PAD + top), mask)
		x += mask.size[0] + LETTER_TRACKING

	return _soften(_fill_holes(_close(out, CLOSE_GAP_PX)))


def _write(img: Image.Image, path: Path) -> None:
	if path.exists() and not OVERWRITE:
		print(f"Skipped existing: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
	img.save(tmp_path, "PNG", compress_level=6)
	with Image.open(tmp_path) as verify_img:
		verify_img.verify()
	with Image.open(tmp_path) as verify_img:
		verify_img.load()
	os.replace(tmp_path, path)
	print(f"Wrote: {path}")


# ----------------------------
# MAIN
# ----------------------------
def word_cloud_mask() -> None:
	source_dir = Path(SOURCE_DIR)
	output_dir = Path(OUTPUT_DIR)
	if not source_dir.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	for name, files in WORDS.items():
		mask = _word_mask(source_dir, files)
		_write(_tint(mask), output_dir / f"{OUTPUT_PREFIX}{name}_cloud_mask.png")

	print(f"Processed {len(WORDS)} words")
	print(f"Output: {output_dir}")


if __name__ == "__main__":
	word_cloud_mask()
