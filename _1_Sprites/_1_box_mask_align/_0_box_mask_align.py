import os
import math
from pathlib import Path

from PIL import Image

GLOBAL_SCALE = 1

# ----------------------------
# CONFIG
# ----------------------------
root_dir = "/mnt/ssd/HMeshi/_2_UI_Uten/tobe_aligned/"
name = "ui_box"

SOURCE_DIR = os.path.join(root_dir, "./")
OUT_DIR = os.path.join(root_dir, f"./{name}_mask_aligned/")

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MASK_SUFFIX = "-mask"
OUTPUT_MASK_SUFFIX = "-mask"

ALPHA_THRESHOLD = 1
BOUNDARY_MIN_VALID_PERCENT = 2.0
CROP_PADDING = 2
FIT_SCALE = 0.98


ONLY_SCALE_DOWN_WHEN_MASK_LARGER = True
OVERWRITE = True
RESAMPLE_MODE = "bicubic"       # "nearest" | "bilinear" | "bicubic" | "lanczos"
COPY_BASE_IMAGE = True


def _resample_filter():
	if RESAMPLE_MODE == "nearest":
		return Image.Resampling.NEAREST
	if RESAMPLE_MODE == "bilinear":
		return Image.Resampling.BILINEAR
	if RESAMPLE_MODE == "bicubic":
		return Image.Resampling.BICUBIC
	if RESAMPLE_MODE == "lanczos":
		return Image.Resampling.LANCZOS
	raise ValueError(f"Unknown RESAMPLE_MODE: {RESAMPLE_MODE}")


def _collect_pairs(source_dir: str):
	src = Path(source_dir)
	if not src.exists():
		raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

	pairs = []
	for base_path in src.iterdir():
		if not base_path.is_file() or base_path.suffix.lower() not in INCLUDE_EXTS:
			continue
		if base_path.stem.endswith(MASK_SUFFIX):
			continue

		mask_path = base_path.with_name(f"{base_path.stem}{MASK_SUFFIX}{base_path.suffix}")
		if mask_path.exists():
			pairs.append((base_path, mask_path))

	if not pairs:
		raise RuntimeError(
			f"No image pairs found in {source_dir}; expected files like 1.png and 1{MASK_SUFFIX}.png"
		)

	return sorted(pairs, key=lambda pair: pair[0].name.lower())


def _alpha_bounds(img: Image.Image):
	alpha = img.getchannel("A")
	pix = alpha.load()
	w, h = img.size
	col_counts = [0 for _ in range(w)]
	row_counts = [0 for _ in range(h)]

	for y in range(h):
		row_count = 0
		for x in range(w):
			if pix[x, y] < ALPHA_THRESHOLD:
				continue
			col_counts[x] += 1
			row_count += 1
		row_counts[y] = row_count

	if not any(col_counts):
		return None

	min_col_count = max(1, int(math.ceil(h * BOUNDARY_MIN_VALID_PERCENT / 100.0)))
	min_row_count = max(1, int(math.ceil(w * BOUNDARY_MIN_VALID_PERCENT / 100.0)))

	valid_cols = [x for x, count in enumerate(col_counts) if count >= min_col_count]
	valid_rows = [y for y, count in enumerate(row_counts) if count >= min_row_count]
	if not valid_cols or not valid_rows:
		return _fallback_alpha_bounds(col_counts, row_counts)

	min_x = valid_cols[0]
	max_x = valid_cols[-1]
	min_y = valid_rows[0]
	max_y = valid_rows[-1]
	return min_x, min_y, max_x, max_y


def _fallback_alpha_bounds(col_counts: list[int], row_counts: list[int]):
	valid_cols = [x for x, count in enumerate(col_counts) if count > 0]
	valid_rows = [y for y, count in enumerate(row_counts) if count > 0]
	if not valid_cols or not valid_rows:
		return None
	return valid_cols[0], valid_rows[0], valid_cols[-1], valid_rows[-1]


def _bounds_size(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
	min_x, min_y, max_x, max_y = bounds
	return max_x - min_x + 1, max_y - min_y + 1


def _bounds_center(bounds: tuple[int, int, int, int]) -> tuple[float, float]:
	min_x, min_y, max_x, max_y = bounds
	return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def _expand_bounds(
	bounds: tuple[int, int, int, int],
	size: tuple[int, int],
	padding: int,
) -> tuple[int, int, int, int]:
	min_x, min_y, max_x, max_y = bounds
	w, h = size
	return (
		max(0, min_x - padding),
		max(0, min_y - padding),
		min(w - 1, max_x + padding),
		min(h - 1, max_y + padding),
	)


def _scaled_mask_crop(
	mask: Image.Image,
	mask_bounds: tuple[int, int, int, int],
	base_bounds: tuple[int, int, int, int],
) -> tuple[Image.Image, float]:
	mask_w, mask_h = _bounds_size(mask_bounds)
	base_w, base_h = _bounds_size(base_bounds)
	scale = 1.0

	if mask_w > base_w or mask_h > base_h or not ONLY_SCALE_DOWN_WHEN_MASK_LARGER:
		scale = min(base_w / float(mask_w), base_h / float(mask_h)) * FIT_SCALE

	scale = max(0.001, min(1.0, scale))
	min_x, min_y, max_x, max_y = mask_bounds
	crop = mask.crop((min_x, min_y, max_x + 1, max_y + 1))

	new_w = max(1, int(round(mask_w * scale)))
	new_h = max(1, int(round(mask_h * scale)))
	if (new_w, new_h) != crop.size:
		crop = crop.resize((new_w, new_h), resample=_resample_filter())

	return crop, scale


def _paste_clipped(dst: Image.Image, src: Image.Image, xy: tuple[int, int]):
	x, y = xy
	src_w, src_h = src.size
	dst_w, dst_h = dst.size

	left = max(0, x)
	top = max(0, y)
	right = min(dst_w, x + src_w)
	bottom = min(dst_h, y + src_h)
	if left >= right or top >= bottom:
		return

	src_box = (left - x, top - y, right - x, bottom - y)
	dst.paste(src.crop(src_box), (left, top))


def _apply_global_scale(img: Image.Image) -> Image.Image:
	if GLOBAL_SCALE <= 0:
		raise ValueError(f"GLOBAL_SCALE must be greater than 0, got {GLOBAL_SCALE}")
	if GLOBAL_SCALE == 1.0:
		return img

	w, h = img.size
	new_w = max(1, int(round(w * GLOBAL_SCALE)))
	new_h = max(1, int(round(h * GLOBAL_SCALE)))
	if (new_w, new_h) == img.size:
		return img
	return img.resize((new_w, new_h), resample=_resample_filter())


def align_mask_to_base(base: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, float]:
	base_bounds = _alpha_bounds(base)
	mask_bounds = _alpha_bounds(mask)
	if base_bounds is None:
		raise RuntimeError("Base image has no alpha-positive pixels")
	if mask_bounds is None:
		raise RuntimeError("Mask image has no alpha-positive pixels")

	crop_bounds = _expand_bounds(base_bounds, base.size, CROP_PADDING)
	crop_min_x, crop_min_y, crop_max_x, crop_max_y = crop_bounds
	cropped_base = base.crop((crop_min_x, crop_min_y, crop_max_x + 1, crop_max_y + 1))
	crop, scale = _scaled_mask_crop(mask, mask_bounds, base_bounds)
	base_cx, base_cy = _bounds_center(base_bounds)
	local_cx = base_cx - crop_min_x
	local_cy = base_cy - crop_min_y
	paste_x = int(round(local_cx - (crop.size[0] - 1) / 2.0))
	paste_y = int(round(local_cy - (crop.size[1] - 1) / 2.0))

	out = Image.new("RGBA", cropped_base.size, (0, 0, 0, 0))
	_paste_clipped(out, crop, (paste_x, paste_y))
	cropped_base = _apply_global_scale(cropped_base)
	out = _apply_global_scale(out)
	return cropped_base, out, scale


def process_pair(base_path: Path, mask_path: Path, out_dir: Path):
	base = Image.open(base_path).convert("RGBA")
	mask = Image.open(mask_path).convert("RGBA")
	cropped_base, aligned_mask, scale = align_mask_to_base(base, mask)

	out_dir.mkdir(parents=True, exist_ok=True)
	base_out = out_dir / base_path.name
	mask_out = out_dir / f"{base_path.stem}{OUTPUT_MASK_SUFFIX}.png"

	if not OVERWRITE:
		for out_path in (base_out, mask_out):
			if out_path.exists():
				raise FileExistsError(f"Output already exists: {out_path}")

	if COPY_BASE_IMAGE:
		cropped_base.save(base_out, "PNG")
	aligned_mask.save(mask_out, "PNG")
	return base_path.name, mask_path.name, cropped_base.size, scale, mask_out


def main():
	pairs = _collect_pairs(SOURCE_DIR)
	out_dir = Path(OUT_DIR)

	for base_path, mask_path in pairs:
		base_name, mask_name, size, scale, out_path = process_pair(base_path, mask_path, out_dir)
		print(
			f"{base_name} + {mask_name}: {size[0]}x{size[1]} | "
			f"mask_fit_scale={scale:.4f} | global_scale={GLOBAL_SCALE:.4f} | {out_path}"
		)

	print(f"Processed {len(pairs)} image pair(s) to {OUT_DIR}")


if __name__ == "__main__":
	main()
