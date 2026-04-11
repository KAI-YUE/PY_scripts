# atlas_pack.py

import os
import json
from pathlib import Path
from PIL import Image

# ----------------------------
# CONFIG (edit these)
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/card_suit_icons/tmp/"
# root_dir = "/mnt/ssd/HMeshi/_0_card_design/cardback_geom_abstract/tmp/"
root_dir = "/home/kyue/Downloads/tmp/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/city/_-2_export/pawns/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_0_henshin_heros/_0_fire/egg/_0_1_snail-rider/for_sheet/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/city/_-2_export/sheet/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/"
# root_dir = "/mnt/ssd/HMeshi/-1_field_landscape/grass/exported/"
root_dir = "/mnt/ssd/HMeshi/_7_live2love/love2d-fxaa/tmp/"
name = "suits"
name = "cards"
# name = "pawns"
# name = "meshi"
# name = "machi"
# name = "grass"
name = "map"
SOURCE_DIR = os.path.join(root_dir, "./")			# folder with 0.png, 1.png, etc.
OUT_ATLAS_PNG = os.path.join(root_dir, "./{:s}.png".format(name))
OUT_ATLAS_JSON = os.path.join(root_dir, "./{:s}.json".format(name))

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_ATLAS_WIDTH = 2048					# typical: 1024/2048/4096
PADDING = 2								# space between sprites to avoid bleeding
SORT_MODE = "height"					# "name" | "height" | "area"
POWER_OF_TWO = False					# round atlas size up to next power of two

# Optional: if your art is pixel-art and you use nearest filtering, PADDING=1 is often ok.


# ----------------------------
# ADD TO CONFIG
# ----------------------------
RESIZE_MODE = "fit_long_edge"		# "none" | "scale" | "fit_long_edge" | "fit_box"
SCALE = 0.5							# used when RESIZE_MODE == "scale"
TARGET_LONG_EDGE = 32				# used when RESIZE_MODE == "fit_long_edge"
TARGET_LONG_EDGE = 90				# used when RESIZE_MODE == "fit_long_edge"
TARGET_LONG_EDGE = 128
TARGET_LONG_EDGE = 512
TARGET_W = 1024						# used when RESIZE_MODE == "fit_box"
TARGET_H = 1024						# used when RESIZE_MODE == "fit_box"
PIXEL_ART = False					# True -> NEAREST, 

# recalculate the long edge by substracting the 2xpadding
if TARGET_LONG_EDGE > 128:
	TARGET_LONG_EDGE -= 2*PADDING

def _resample_filter():
	# Pillow resampling choice
	# return Image.Resampling.NEAREST if PIXEL_ART else Image.Resampling.LANCZOS
    return Image.Resampling.NEAREST if PIXEL_ART else Image.Resampling.BICUBIC


def _maybe_resize(img: Image.Image) -> Image.Image:
	if RESIZE_MODE == "none":
		return img

	w, h = img.size
	resample = _resample_filter()

	if RESIZE_MODE == "scale":
		nw = max(1, int(round(w * SCALE)))
		nh = max(1, int(round(h * SCALE)))
		if (nw, nh) == (w, h):
			return img
		return img.resize((nw, nh), resample=resample)

	if RESIZE_MODE == "fit_long_edge":
		long_edge = max(w, h)
		# if long_edge <= TARGET_LONG_EDGE:
		# 	return img
		scale = TARGET_LONG_EDGE / float(long_edge)
		nw = max(1, int(round(w * scale)))
		nh = max(1, int(round(h * scale)))
		return img.resize((nw, nh), resample=resample)

	if RESIZE_MODE == "fit_box":
		# Fit inside TARGET_W x TARGET_H, keep aspect ratio
		if w <= TARGET_W and h <= TARGET_H:
			return img
		scale = min(TARGET_W / float(w), TARGET_H / float(h))
		nw = max(1, int(round(w * scale)))
		nh = max(1, int(round(h * scale)))
		return img.resize((nw, nh), resample=resample)

	raise ValueError(f"Unknown RESIZE_MODE: {RESIZE_MODE}")


def _next_pow2(n: int) -> int:
	if n <= 0:
		return 1
	p = 1
	while p < n:
		p <<= 1
	return p


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

	# Load all images (RGBA)
	items = []
	for p in paths:
		img = Image.open(p).convert("RGBA")
		img = _maybe_resize(img)
		w, h = img.size
		items.append({
			"name": p.name,
			"path": str(p),
			"img": img,
			"w": w,
			"h": h,
		})

	# Deterministic ordering
	if SORT_MODE == "name":
		items.sort(key=lambda it: it["name"].lower())
	elif SORT_MODE == "area":
		items.sort(key=lambda it: (it["w"] * it["h"], it["h"], it["w"]), reverse=True)
	else:
		# "height" default
		items.sort(key=lambda it: (it["h"], it["w"]), reverse=True)

	return items


def _shelf_pack(items, max_width: int, padding: int):
	# Simple shelf packing: left-to-right, wrap to next row when width exceeds.
	x = padding
	y = padding
	row_h = 0
	used_w = 0

	placements = {}

	for it in items:
		w = it["w"]
		h = it["h"]

		if w + padding * 2 > max_width:
			raise RuntimeError(
				f"Sprite '{it['name']}' width {w}px too large for MAX_ATLAS_WIDTH={max_width}px"
			)

		# Wrap row
		if x + w + padding > max_width:
			x = padding
			y = y + row_h + padding
			row_h = 0

		placements[it["name"]] = { "x": x, "y": y, "w": w, "h": h, }

		x = x + w + padding
		row_h = max(row_h, h)
		used_w = max(used_w, x)

	used_h = y + row_h + padding
	used_w = max(used_w, padding)  # safety

	return placements, used_w, used_h


def build_atlas():
	items = _collect_images(SOURCE_DIR)
	placements, atlas_w, atlas_h = _shelf_pack(items, MAX_ATLAS_WIDTH, PADDING)

	# Optionally round up to power-of-two sizes
	final_w = _next_pow2(atlas_w) if POWER_OF_TWO else atlas_w
	final_h = _next_pow2(atlas_h) if POWER_OF_TWO else atlas_h

	atlas = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))

	# Paste sprites
	for it in items:
		fr = placements[it["name"]]
		atlas.paste(it["img"], (fr["x"], fr["y"]))

	# Save atlas PNG
	out_png = Path(OUT_ATLAS_PNG)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	atlas.save(out_png, "PNG")

	# Write metadata JSON
	meta = {
		"meta": {
			"image": str(out_png.name),
			"size": {"w": final_w, "h": final_h},
		},
		"frames": {}
	}

	for name, fr in placements.items():
		# Store rectangles (top-left origin)
		meta["frames"][name[:-4]] = {
			"frame": {"x": fr["x"], "y": fr["y"], "w": fr["w"], "h": fr["h"]}
		}

	out_json = Path(OUT_ATLAS_JSON)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	# Print summary
	total_area = sum(it["w"] * it["h"] for it in items)
	atlas_area = final_w * final_h
	util = (total_area / atlas_area) * 100.0 if atlas_area > 0 else 0.0
	print(f"Packed {len(items)} sprites into {final_w}x{final_h} atlas")
	print(f"Utilization (rough): {util:.1f}%")
	print(f"Wrote: {out_png} and {out_json}")


if __name__ == "__main__":
	build_atlas()
