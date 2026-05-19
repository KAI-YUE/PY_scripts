from pathlib import Path
import json

from PIL import Image

#%% ----------------------------
# CONFIG
# ----------------------------
SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/gemini_uibox/_2_export/_2_strokes/"
OUTPUT_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/gemini_uibox/_2_export/_2_strokes/cropped"

RECURSIVE = False

ALPHA_THRESHOLD = 1         # pixels below this alpha are treated as empty
PADDING_PX = 2              # keep 1-2 pixels around sprite

OVERWRITE = False           # if True, replaces original files
FORCE_PNG_OUTPUT = True     # recommended for sprites with alpha
WRITE_METADATA = True       # saves crop offset/original size info

INCLUDE_EXTS = {".png", ".webp", ".bmp", ".tif", ".tiff"}


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


def image_to_rgba(img: Image.Image) -> Image.Image:
	if img.mode == "RGBA":
		return img.copy()
	return img.convert("RGBA")


def alpha_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
	rgba = image_to_rgba(img)
	alpha = rgba.getchannel("A")

	if ALPHA_THRESHOLD <= 1:
		return alpha.getbbox()

	mask = alpha.point(lambda a: 255 if a >= ALPHA_THRESHOLD else 0)
	return mask.getbbox()


def pad_bbox(
	bbox: tuple[int, int, int, int],
	width: int,
	height: int,
	padding: int,
) -> tuple[int, int, int, int]:
	left, top, right, bottom = bbox

	left = max(0, left - padding)
	top = max(0, top - padding)
	right = min(width, right + padding)
	bottom = min(height, bottom + padding)

	return left, top, right, bottom


def cropped_output_path(source: Path, image_path: Path, output_root: Path) -> Path:
	if source.is_file():
		relative = image_path.name
	else:
		relative = image_path.relative_to(source)

	output_path = output_root / relative

	if FORCE_PNG_OUTPUT:
		output_path = output_path.with_suffix(".png")

	return output_path


def crop_sprite(image_path: Path) -> dict:
	with Image.open(image_path) as img:
		rgba = image_to_rgba(img)
		w, h = rgba.size

		bbox = alpha_bbox(rgba)

		if bbox is None:
			return {
				"path": str(image_path),
				"status": "skipped_empty_alpha",
				"original_size": [w, h],
				"cropped_size": [w, h],
				"bbox": None,
				"offset": [0, 0],
			}

		padded_bbox = pad_bbox(bbox, w, h, PADDING_PX)
		left, top, right, bottom = padded_bbox

		cropped = rgba.crop(padded_bbox)

	return {
		"path": str(image_path),
		"status": "cropped",
		"original_size": [w, h],
		"cropped_size": [right - left, bottom - top],
		"bbox": [left, top, right, bottom],
		"offset": [left, top],
		"image": cropped,
	}


def save_sprite(result: dict, output_path: Path) -> None:
	if result["status"] != "cropped":
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	result["image"].save(output_path)


def clean_metadata(result: dict, output_path: Path | None = None) -> dict:
	data = dict(result)
	data.pop("image", None)

	if output_path is not None:
		data["output_path"] = str(output_path)

	return data


def print_result(data: dict) -> None:
	print(data["path"])
	print(f"  status: {data['status']}")
	print(f"  original_size: {tuple(data['original_size'])}")
	print(f"  cropped_size: {tuple(data['cropped_size'])}")
	print(f"  offset: {tuple(data['offset'])}")

	if data["bbox"] is not None:
		print(f"  bbox: {tuple(data['bbox'])}")


#%% ----------------------------
# MAIN
# ----------------------------
def main() -> None:
	source = Path(SOURCE_PATH).resolve()
	output_root = Path(OUTPUT_PATH).resolve()

	images = collect_images(source, RECURSIVE)
	if not images:
		raise RuntimeError(f"No images found in {SOURCE_PATH} with extensions {sorted(INCLUDE_EXTS)}")

	print(f"ALPHA_THRESHOLD: {ALPHA_THRESHOLD}")
	print(f"PADDING_PX: {PADDING_PX}")
	print(f"RECURSIVE: {RECURSIVE}")
	print(f"OVERWRITE: {OVERWRITE}")
	print("")

	metadata = []

	for image_path in images:
		result = crop_sprite(image_path)

		if OVERWRITE:
			output_path = image_path
		else:
			output_path = cropped_output_path(source, image_path, output_root)

		save_sprite(result, output_path)

		data = clean_metadata(result, output_path)
		metadata.append(data)
		print_result(data)

		print("")


if __name__ == "__main__":
	main()