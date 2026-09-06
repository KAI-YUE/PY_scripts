# solid_filled_mask.py

import os
from collections import deque
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter


# ----------------------------
# CONFIG (edit these)
# ----------------------------
SOURCE_DIR = os.environ.get(
	"GLYPH_SOURCE_DIR",
	# "/mnt/ssd/HMeshi/_2_UI_Uten/_4_gampad_btns/_-1_prev/_0_title_page/_00_export",
	"/mnt/ssd/HMeshi/_2_UI_Uten/_4_gampad_btns/inter_btn_pack_pieces/tape_edge/"
)
OUTPUT_DIR = os.environ.get("GLYPH_OUTPUT_DIR", os.path.join(SOURCE_DIR, "_solid_filled_mask"))
GLOB_PATTERN = os.environ.get("GLYPH_GLOB", "_pr_eng*.png")

ALPHA_THRESHOLD = int(os.environ.get("GLYPH_ALPHA_THRESHOLD", "8"))
FILL_HOLES = os.environ.get("GLYPH_FILL_HOLES", "1") != "0"
SEAL_GAP_PX = int(os.environ.get("GLYPH_SEAL_GAP_PX", "7"))
EXPAND_PX = int(os.environ.get("GLYPH_EXPAND_PX", "4"))
EXPAND_PX_BY_GLYPH = {
	"u": 10,
	"n": 10,
	"t": 8,
	"r": 8,
	"y": 8,
	"a": 10
}
BLUR_RADIUS = float(os.environ.get("GLYPH_BLUR_RADIUS", "1"))
SOFT_CONTOUR_BLUR = float(os.environ.get("GLYPH_SOFT_CONTOUR_BLUR", "2.4"))
SOFT_CONTOUR_ALPHA = float(os.environ.get("GLYPH_SOFT_CONTOUR_ALPHA", "0.85"))
MASK_RGBA = tuple(int(v) for v in os.environ.get("GLYPH_MASK_RGBA", "255,255,255,255").split(","))

OUTPUT_TYPES = {"sealed_underlay"}		# "solid_mask" | "underlay" | "open_mask" | "open_underlay" | "sealed_mask" | "sealed_underlay"
OVERWRITE = True
GENERATED_SUFFIXES = (
	"_solid_mask",
	"_underlay",
	"_open_mask",
	"_open_underlay",
	"_sealed_mask",
	"_sealed_underlay",
)


# ----------------------------
# HELPERS
# ----------------------------
def _alpha_mask(img: Image.Image) -> Image.Image:
	alpha = img.getchannel("A")
	return alpha.point(lambda a: 255 if a > ALPHA_THRESHOLD else 0, "L")


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


def _expand_mask(mask: Image.Image, px: int = EXPAND_PX) -> Image.Image:
	if px <= 0:
		return mask
	return mask.filter(ImageFilter.MaxFilter(px * 2 + 1))


def _soft_contour_mask(mask: Image.Image) -> Image.Image:
	if SOFT_CONTOUR_BLUR <= 0:
		return mask
	halo = mask.filter(ImageFilter.GaussianBlur(SOFT_CONTOUR_BLUR))
	halo = halo.point(lambda a: min(255, int(a * SOFT_CONTOUR_ALPHA)), "L")
	return ImageChops.lighter(mask, halo)


def _underlay_mask(mask: Image.Image, path: Path) -> Image.Image:
	return _soft_contour_mask(_expand_mask(mask, _expand_px_for(path)))


def _grow_mask(mask: Image.Image, px: int) -> Image.Image:
	if px <= 0:
		return mask
	return mask.filter(ImageFilter.MaxFilter(px * 2 + 1))


def _sealed_mask(mask: Image.Image) -> Image.Image:
	sealed = _fill_holes(_grow_mask(mask, SEAL_GAP_PX))
	new_fill = ImageChops.subtract(sealed, _grow_mask(mask, SEAL_GAP_PX))
	return _fill_holes(ImageChops.lighter(mask, new_fill))


def _tint_mask(mask: Image.Image) -> Image.Image:
	if BLUR_RADIUS > 0:
		mask = mask.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
	color = Image.new("RGBA", mask.size, MASK_RGBA)
	color.putalpha(mask.point(lambda a: int(a * MASK_RGBA[3] / 255), "L"))
	return color


def _output_path(output_dir: Path, src_path: Path, suffix: str) -> Path:
	return output_dir / f"{src_path.stem}{suffix}.png"


def _is_source_glyph(path: Path) -> bool:
	return not any(path.stem.endswith(suffix) for suffix in GENERATED_SUFFIXES)


def _glyph_name(path: Path) -> str:
	return path.stem.rsplit("_", 1)[-1].lower()


def _expand_px_for(path: Path) -> int:
	return EXPAND_PX_BY_GLYPH.get(_glyph_name(path), EXPAND_PX)


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
def solid_filled_mask() -> None:
	source_dir = Path(SOURCE_DIR)
	output_dir = Path(OUTPUT_DIR)
	if not source_dir.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	paths = sorted(p for p in source_dir.glob(GLOB_PATTERN) if p.is_file() and _is_source_glyph(p))
	if not paths:
		raise RuntimeError(f"No glyphs found in {source_dir} matching {GLOB_PATTERN}")

	for path in paths:
		img = Image.open(path).convert("RGBA")
		open_mask = _alpha_mask(img)
		needs_solid = bool({"solid_mask", "underlay"} & OUTPUT_TYPES)
		needs_sealed = bool({"sealed_mask", "sealed_underlay"} & OUTPUT_TYPES)
		solid_mask = _fill_holes(open_mask) if needs_solid and FILL_HOLES else open_mask
		sealed_mask = _sealed_mask(open_mask) if needs_sealed else open_mask

		if "solid_mask" in OUTPUT_TYPES:
			_write(solid_mask, _output_path(output_dir, path, "_solid_mask"))
		if "underlay" in OUTPUT_TYPES:
			_write(_tint_mask(_underlay_mask(solid_mask, path)), _output_path(output_dir, path, "_underlay"))
		if "open_mask" in OUTPUT_TYPES:
			_write(open_mask, _output_path(output_dir, path, "_open_mask"))
		if "open_underlay" in OUTPUT_TYPES:
			_write(_tint_mask(_underlay_mask(open_mask, path)), _output_path(output_dir, path, "_open_underlay"))
		if "sealed_mask" in OUTPUT_TYPES:
			_write(sealed_mask, _output_path(output_dir, path, "_sealed_mask"))
		if "sealed_underlay" in OUTPUT_TYPES:
			_write(_tint_mask(_underlay_mask(sealed_mask, path)), _output_path(output_dir, path, "_sealed_underlay"))

	print(f"Processed {len(paths)} glyphs")
	print(f"Output: {output_dir}")


if __name__ == "__main__":
	solid_filled_mask()
