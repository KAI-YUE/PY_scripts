from pathlib import Path

from PIL import Image


# ----------------------------
# CONFIG
# ----------------------------
SOURCE_DIR = "/mnt/ssd/HMeshi/-1_field_landscape/rocks_stuff/tmp/"
OUTPUT_DIR = None
MODE = "scale"  # "scale" or "max_edge"
SCALE = 0.45
MAX_EDGE = 512
RESAMPLE = "lanczos"  # "nearest" | "bilinear" | "bicubic" | "lanczos"
OVERWRITE = False
RECURSIVE = True
SKIP_UPSCALE = True

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_resample(name: str) -> int:
	mapping = {
		"nearest": Image.Resampling.NEAREST,
		"bilinear": Image.Resampling.BILINEAR,
		"bicubic": Image.Resampling.BICUBIC,
		"lanczos": Image.Resampling.LANCZOS,
	}
	return mapping[name]


def resolve_mode_values() -> tuple[float | None, int | None]:
	if MODE == "scale":
		return SCALE, None
	if MODE == "max_edge":
		return None, MAX_EDGE
	raise ValueError(f"MODE must be 'scale' or 'max_edge', got {MODE!r}")


def collect_images(source: Path, recursive: bool) -> list[Path]:
	if not source.exists():
		raise FileNotFoundError(f"Source folder not found: {source}")
	if not source.is_dir():
		raise NotADirectoryError(f"Source path is not a folder: {source}")

	pattern = "**/*" if recursive else "*"
	return sorted(
		[
			path
			for path in source.glob(pattern)
			if path.is_file() and path.suffix.lower() in INCLUDE_EXTS
		],
		key=lambda path: str(path).lower(),
	)


def target_size(
	size: tuple[int, int],
	scale: float | None,
	max_edge: int | None,
	skip_upscale: bool,
) -> tuple[int, int]:
	width, height = size

	if scale is not None:
		if scale <= 0:
			raise ValueError("--scale must be greater than 0")
		if skip_upscale and scale >= 1.0:
			return width, height
		return (
			max(1, int(round(width * scale))),
			max(1, int(round(height * scale))),
		)

	if max_edge is None or max_edge <= 0:
		raise ValueError("--max-edge must be greater than 0")

	long_edge = max(width, height)
	if long_edge == 0:
		return width, height
	if skip_upscale and long_edge <= max_edge:
		return width, height

	scale_ratio = max_edge / float(long_edge)
	return (
		max(1, int(round(width * scale_ratio))),
		max(1, int(round(height * scale_ratio))),
	)


def output_path_for(source_root: Path, output_root: Path, image_path: Path) -> Path:
	relative = image_path.relative_to(source_root)
	return output_root / relative


def save_image(img: Image.Image, out_path: Path, original_format: str | None) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)

	save_format = original_format
	save_img = img
	if (original_format or "").upper() == "JPEG" and img.mode in ("RGBA", "LA"):
		save_img = img.convert("RGB")

	if out_path.suffix.lower() in {".jpg", ".jpeg"} and save_img.mode in ("RGBA", "LA"):
		save_img = save_img.convert("RGB")
		save_format = "JPEG"

	save_img.save(out_path, format=save_format)


def main() -> None:
	source_root = Path(SOURCE_DIR).resolve()

	if OVERWRITE and OUTPUT_DIR is not None:
		raise ValueError("Use either OVERWRITE=True or set OUTPUT_DIR, not both")

	scale, max_edge = resolve_mode_values()
	if scale is not None and scale == 1.0 and not OVERWRITE:
		print("Warning: SCALE=1.0 will copy images without changing their size.")

	output_root = source_root if OVERWRITE else (
		Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else source_root.with_name(f"{source_root.name}_downsampled")
	)

	images = collect_images(source_root, RECURSIVE)
	if not images:
		raise RuntimeError(f"No images found in {source_root} with extensions {sorted(INCLUDE_EXTS)}")

	resample = resolve_resample(RESAMPLE)
	processed = 0
	skipped = 0

	for image_path in images:
		with Image.open(image_path) as img:
			dst_size = target_size(
				img.size,
				scale=scale,
				max_edge=max_edge,
				skip_upscale=SKIP_UPSCALE,
			)

			out_path = output_path_for(source_root, output_root, image_path)

			if dst_size == img.size:
				if OVERWRITE:
					skipped += 1
					continue
				save_image(img.copy(), out_path, img.format)
				skipped += 1
				continue

			downsampled = img.resize(dst_size, resample=resample)
			save_image(downsampled, out_path, img.format)
			processed += 1
			print(f"{image_path} -> {out_path} ({img.size[0]}x{img.size[1]} -> {dst_size[0]}x{dst_size[1]})")

	print(f"Found {len(images)} images")
	print(f"Resized {processed} images")
	print(f"Unchanged {skipped} images")
	print(f"Output: {output_root}")


if __name__ == "__main__":
	main()
