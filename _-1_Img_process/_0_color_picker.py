from collections import Counter
from pathlib import Path

from PIL import Image


#%% ----------------------------
# CONFIG
# ----------------------------
SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/_0_ui_box/_0_used/_0_color_picker/1.png"
SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/_0_ui_box/_0_used/_0_color_picker/txt.png"

COLOR_MODE = "median"       # "median" | "mean" | "dominant" | "center"
ALPHA_THRESHOLD = 1         # pixels below this alpha are ignored
INCLUDE_ALPHA = True
RECURSIVE = False

INCLUDE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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


def median_channel(values: list[int]) -> int:
	values = sorted(values)
	count = len(values)
	mid = count // 2
	if count % 2:
		return values[mid]
	return int(round((values[mid - 1] + values[mid]) / 2.0))


def rgba_to_hex(rgba: tuple[int, int, int, int]) -> str:
	r, g, b, a = rgba
	if INCLUDE_ALPHA:
		return f"#{r:02X}{g:02X}{b:02X}{a:02X}"
	return f"#{r:02X}{g:02X}{b:02X}"


def rgba_to_float_triplet(rgba: tuple[int, int, int, int]) -> tuple[float, ...]:
	values = rgba if INCLUDE_ALPHA else rgba[:3]
	return tuple(round(v / 255.0, 6) for v in values)


def rgba_to_int_triplet(rgba: tuple[int, int, int, int]) -> tuple[int, ...]:
	return rgba if INCLUDE_ALPHA else rgba[:3]


def valid_pixels(img: Image.Image) -> list[tuple[int, int, int, int]]:
	rgba = image_to_rgba(img)
	pixels = []
	for pixel in rgba.getdata():
		if pixel[3] >= ALPHA_THRESHOLD:
			pixels.append(pixel)
	return pixels


def pick_center_color(img: Image.Image) -> tuple[int, int, int, int]:
	rgba = image_to_rgba(img)
	w, h = rgba.size
	return rgba.getpixel((w // 2, h // 2))


def pick_color(img: Image.Image) -> tuple[int, int, int, int]:
	if COLOR_MODE == "center":
		return pick_center_color(img)

	pixels = valid_pixels(img)
	if not pixels:
		raise RuntimeError("Image has no pixels at or above ALPHA_THRESHOLD")

	if COLOR_MODE == "dominant":
		return Counter(pixels).most_common(1)[0][0]

	if COLOR_MODE == "mean":
		count = len(pixels)
		return tuple(
			int(round(sum(pixel[channel] for pixel in pixels) / float(count)))
			for channel in range(4)
		)

	if COLOR_MODE == "median":
		return tuple(
			median_channel([pixel[channel] for pixel in pixels])
			for channel in range(4)
		)

	raise ValueError(f"Unknown COLOR_MODE: {COLOR_MODE}")


def print_color(image_path: Path) -> None:
	with Image.open(image_path) as img:
		rgba = pick_color(img)

	print(image_path)
	print(f"  hex: {rgba_to_hex(rgba)}")
	print(f"  triplet_255: {rgba_to_int_triplet(rgba)}")
	print(f"  triplet_01: {rgba_to_float_triplet(rgba)}")


#%% ----------------------------
# MAIN
# ----------------------------
def main() -> None:
	images = collect_images(Path(SOURCE_PATH).resolve(), RECURSIVE)
	if not images:
		raise RuntimeError(f"No images found in {SOURCE_PATH} with extensions {sorted(INCLUDE_EXTS)}")

	print(f"COLOR_MODE: {COLOR_MODE}")
	print(f"ALPHA_THRESHOLD: {ALPHA_THRESHOLD}")
	for image_path in images:
		print_color(image_path)


if __name__ == "__main__":
	main()
