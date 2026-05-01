import os

from pathlib import Path

from PIL import Image


#%% ----------------------------
# CONFIG
# ----------------------------
SOURCE_PATH = "/mnt/ssd/HMeshi/-1_field_landscape/grass/bird/"
OUTPUT_DIR = os.path.join(SOURCE_PATH, "green")
OUTPUT_SUFFIX = "_dark_green_bg"

BACKGROUND_COLOR = (8, 48, 24)  # dark green RGB
OUTPUT_FORMAT = "png"           # "png" | "jpg" | "webp" | "same"
OVERWRITE = False
RECURSIVE = True

INCLUDE_EXTS = {".png", ".webp", ".tif", ".tiff"}


#%% ----------------------------
# HELPERS
# ----------------------------
def collect_images(source: Path, recursive: bool) -> list[Path]:
	if not source.exists():
		raise FileNotFoundError(f"Source path not found: {source}")

	if source.is_file():
		if source.suffix.lower() not in INCLUDE_EXTS:
			raise ValueError(f"Unsupported image extension: {source.suffix}")
		return [source]

	if not source.is_dir():
		raise NotADirectoryError(f"Source path is not a file or folder: {source}")

	pattern = "**/*" if recursive else "*"
	return sorted(
		[
			path
			for path in source.glob(pattern)
			if path.is_file() and path.suffix.lower() in INCLUDE_EXTS
		],
		key=lambda path: str(path).lower(),
	)


def output_root_for(source: Path) -> Path:
	if OUTPUT_DIR:
		return Path(OUTPUT_DIR).resolve()

	if source.is_file():
		return source.parent

	return source.with_name(f"{source.name}{OUTPUT_SUFFIX}")


def output_path_for(source_root: Path, output_root: Path, image_path: Path) -> Path:
	if OVERWRITE:
		return image_path

	if source_root.is_file():
		relative = image_path.name
	else:
		relative = image_path.relative_to(source_root)

	if OUTPUT_FORMAT == "same":
		return output_root / relative

	return (output_root / relative).with_suffix(f".{OUTPUT_FORMAT}")


def image_to_rgba(img: Image.Image) -> Image.Image:
	if img.mode == "RGBA":
		return img.copy()

	if img.mode == "LA":
		return img.convert("RGBA")

	if img.mode == "P" and "transparency" in img.info:
		return img.convert("RGBA")

	if img.mode in ("RGB", "L"):
		return img.convert("RGBA")

	return img.convert("RGBA")


def apply_background(img: Image.Image) -> Image.Image:
	rgba = image_to_rgba(img)
	background = Image.new("RGBA", rgba.size, (*BACKGROUND_COLOR, 255))
	background.alpha_composite(rgba)
	return background.convert("RGB")


def save_image(img: Image.Image, out_path: Path, original_format: str | None) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)

	save_format = original_format if OUTPUT_FORMAT == "same" else OUTPUT_FORMAT.upper()
	if save_format == "JPG":
		save_format = "JPEG"

	img.save(out_path, format=save_format)


#%% ----------------------------
# MAIN
# ----------------------------
def main() -> None:
	source = Path(SOURCE_PATH).resolve()

	if OVERWRITE and OUTPUT_DIR is not None:
		raise ValueError("Use either OVERWRITE=True or set OUTPUT_DIR, not both")

	if OUTPUT_FORMAT not in {"png", "jpg", "webp", "same"}:
		raise ValueError(f"OUTPUT_FORMAT must be 'png', 'jpg', 'webp', or 'same', got {OUTPUT_FORMAT!r}")

	images = collect_images(source, RECURSIVE)
	if not images:
		raise RuntimeError(f"No images found in {source} with extensions {sorted(INCLUDE_EXTS)}")

	output_root = output_root_for(source)
	processed = 0

	for image_path in images:
		out_path = output_path_for(source, output_root, image_path)

		with Image.open(image_path) as img:
			flattened = apply_background(img)
			save_image(flattened, out_path, img.format)

		processed += 1
		print(f"{image_path} -> {out_path}")

	print(f"Processed {processed} images")
	print(f"Background color: {BACKGROUND_COLOR}")
	print(f"Output: {output_root}")


if __name__ == "__main__":
	main()
