# atlas_pack.py

import os
import json
from pathlib import Path
from PIL import Image

# ----------------------------
# CONFIG (edit these)
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_0_card_design/card_suit_icons/tmp/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/colorful_design/cards_gradient_band/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_2_card_suit_icons/suits/export/"
root_dir = "/mnt/ssd/HMeshi/_0_card_design/_1_cardback_geom_abstract/export/cards_gradient_band/"
root_dir = "/mnt/ssd/HMeshi/-1_field_landscape/grass/tiles/patch_quilting_output/"
name = "suits"
name = "cards"
# name = "pawns"
# name = "meshi"
# name = "machi"
# name = "grass"
# name = "map"
name = "grass_tiles"
SOURCE_DIR = os.path.join(root_dir, "./")			# folder with 0.png, 1.png, etc.
OUT_ATLAS_PNG = os.path.join(root_dir, "./{:s}.png".format(name))
OUT_ATLAS_JSON = os.path.join(root_dir, "./{:s}.json".format(name))

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_ATLAS_WIDTH = 2048					# typical: 1024/2048/4096
PADDING = 2								# space between sprites to avoid bleeding
SORT_MODE = "name"					# "name" | "height" | "area"
POWER_OF_TWO = False					# round atlas size up to next power of two
INPUT_HAS_EXTRUDED_PADDING = True
INPUT_HAS_EXTRUDED_PADDING = False
INPUT_EXTRUDED_PADDING = 4


# ----------------------------
# ADD TO CONFIG
# ----------------------------
RESIZE_MODE = "fit_long_edge"		# "none" | "scale" | "fit_long_edge" | "fit_box"
SCALE = 0.5							# used when RESIZE_MODE == "scale"
TARGET_LONG_EDGE = [128]			# used when RESIZE_MODE == "fit_long_edge"
TARGET_LONG_EDGE = [32, 64, 128, 256]
TARGET_LONG_EDGE = [256]
# TARGET_LONG_EDGE = 512				
# TARGET_LONG_EDGE = 128
# TARGET_LONG_EDGE = 256
# TARGET_LONG_EDGE = 512
TARGET_W = 1024						# used when RESIZE_MODE == "fit_box"
TARGET_H = 1024						# used when RESIZE_MODE == "fit_box"
PIXEL_ART = False					# True -> NEAREST, 

def _resample_filter():
	# Pillow resampling choice
	# return Image.Resampling.NEAREST if PIXEL_ART else Image.Resampling.LANCZOS
    return Image.Resampling.NEAREST if PIXEL_ART else Image.Resampling.BICUBIC


def _normalize_target_long_edges():
	if isinstance(TARGET_LONG_EDGE, (list, tuple)):
		values = list(TARGET_LONG_EDGE)
	else:
		values = [TARGET_LONG_EDGE]

	normalized = []
	for value in values:
		if value is None:
			continue
		edge = int(value)
		if edge <= 0:
			continue
		if edge > 128:
			edge -= 2 * PADDING
		normalized.append(edge)

	if not normalized:
		raise ValueError("TARGET_LONG_EDGE must contain at least one positive value")

	return normalized


def _maybe_resize(img: Image.Image, target_long_edge: int | None = None) -> Image.Image:
	if INPUT_HAS_EXTRUDED_PADDING:
		return img

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
		if not target_long_edge:
			return img
		long_edge = max(w, h)
		# if long_edge <= target_long_edge:
		# 	return img
		scale = target_long_edge / float(long_edge)
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


def _collect_images(source_dir: str, target_long_edge: int | None = None):
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
		img = _maybe_resize(img, target_long_edge=target_long_edge)
		w, h = img.size
		if INPUT_HAS_EXTRUDED_PADDING and (w <= INPUT_EXTRUDED_PADDING * 2 or h <= INPUT_EXTRUDED_PADDING * 2):
			raise RuntimeError(
				f"Sprite '{p.name}' is too small for INPUT_EXTRUDED_PADDING={INPUT_EXTRUDED_PADDING}px: {w}x{h}"
			)
		content_w = max(1, w - INPUT_EXTRUDED_PADDING * 2) if INPUT_HAS_EXTRUDED_PADDING else w
		content_h = max(1, h - INPUT_EXTRUDED_PADDING * 2) if INPUT_HAS_EXTRUDED_PADDING else h
		items.append({
			"name": p.name,
			"path": str(p),
			"img": img,
			"w": w,
			"h": h,
			"content_x": INPUT_EXTRUDED_PADDING if INPUT_HAS_EXTRUDED_PADDING else 0,
			"content_y": INPUT_EXTRUDED_PADDING if INPUT_HAS_EXTRUDED_PADDING else 0,
			"content_w": content_w,
			"content_h": content_h,
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


def _output_paths(target_long_edge: int | None = None):
	if RESIZE_MODE == "fit_long_edge" and target_long_edge:
		out_dir = Path(root_dir) / str(target_long_edge)
		return out_dir / f"{name}.png", out_dir / f"{name}.json"

	return Path(OUT_ATLAS_PNG), Path(OUT_ATLAS_JSON)


def build_atlas(target_long_edge: int | None = None):
	items = _collect_images(SOURCE_DIR, target_long_edge=target_long_edge)
	items_by_name = {it["name"]: it for it in items}
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
	out_png, out_json = _output_paths(target_long_edge=target_long_edge)
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
		item = items_by_name[name]
		# Store rectangles (top-left origin)
		meta["frames"][name[:-4]] = {
			"frame": {
				"x": fr["x"] + item["content_x"],
				"y": fr["y"] + item["content_y"],
				"w": item["content_w"],
				"h": item["content_h"],
			},
			# "sourceSize": {"w": item["content_w"], "h": item["content_h"]},
			# "spriteSourceSize": {"x": 0, "y": 0, "w": item["content_w"], "h": item["content_h"]},
			# "atlasRect": {"x": fr["x"], "y": fr["y"], "w": fr["w"], "h": fr["h"]},
		}

	out_json.parent.mkdir(parents=True, exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	# Print summary
	total_area = sum(it["w"] * it["h"] for it in items)
	atlas_area = final_w * final_h
	util = (total_area / atlas_area) * 100.0 if atlas_area > 0 else 0.0
	target_label = f" target_long_edge={target_long_edge}" if target_long_edge else ""
	print(f"Packed {len(items)} sprites into {final_w}x{final_h} atlas{target_label}")
	print(f"Utilization (rough): {util:.1f}%")
	print(f"Wrote: {out_png} and {out_json}")


if __name__ == "__main__":
	if RESIZE_MODE == "fit_long_edge":
		for target_long_edge in _normalize_target_long_edges():
			build_atlas(target_long_edge=target_long_edge)
	else:
		build_atlas()
