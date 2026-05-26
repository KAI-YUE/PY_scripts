from pathlib import Path

from PIL import Image


#%% ----------------------------
# CONFIG
# ----------------------------
SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/disk/"

OUTPUT_SUFFIX = "_subdisk"
OVERWRITE = False

ALPHA_THRESHOLD = 1         # pixels >= this alpha are treated as disk pixels
RECURSIVE = False

SUBDISK_RATIO = 0.97        # 1.0 = full disk, 0.5 = center half-radius disk
FULL_DISK_RADIUS_RATIO = 1.0

CENTER_MODE = "alpha_bbox"  # "alpha_bbox" | "canvas"
RADIUS_MODE = "alpha_bbox"  # "alpha_bbox" | "canvas"

CROP_TO_SUBDISK = True      # True = export only the subdisk square region
PAD_PIXELS = 0              # extra transparent padding after crop

CLEAR_RGB_WHEN_TRANSPARENT = True

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


def image_to_rgba(img: Image.Image) -> Image.Image:
	if img.mode == "RGBA":
		return img.copy()
	return img.convert("RGBA")


def get_output_path(image_path: Path) -> Path:
	if OVERWRITE:
		return image_path

	output_path = image_path.with_name(f"{image_path.stem}{OUTPUT_SUFFIX}{image_path.suffix}")

	if output_path == image_path:
		raise ValueError("Output path matches input path. Use OVERWRITE=True or set OUTPUT_SUFFIX.")

	return output_path


def get_alpha_bbox(rgba: Image.Image) -> tuple[int, int, int, int]:
	w, h = rgba.size
	pixels = rgba.load()

	min_x = w
	min_y = h
	max_x = -1
	max_y = -1

	for y in range(h):
		for x in range(w):
			a = pixels[x, y][3]

			if a >= ALPHA_THRESHOLD:
				min_x = min(min_x, x)
				min_y = min(min_y, y)
				max_x = max(max_x, x)
				max_y = max(max_y, y)

	if max_x < min_x or max_y < min_y:
		raise RuntimeError("No visible alpha pixels found.")

	return min_x, min_y, max_x, max_y


def get_disk_center_and_radius(rgba: Image.Image) -> tuple[float, float, float]:
	w, h = rgba.size

	if CENTER_MODE == "canvas":
		cx = w * 0.5
		cy = h * 0.5
	elif CENTER_MODE == "alpha_bbox":
		min_x, min_y, max_x, max_y = get_alpha_bbox(rgba)
		cx = (min_x + max_x + 1) * 0.5
		cy = (min_y + max_y + 1) * 0.5
	else:
		raise ValueError(f"Unsupported CENTER_MODE: {CENTER_MODE}")

	if RADIUS_MODE == "canvas":
		radius = min(w, h) * 0.5
	elif RADIUS_MODE == "alpha_bbox":
		min_x, min_y, max_x, max_y = get_alpha_bbox(rgba)
		bw = max_x - min_x + 1
		bh = max_y - min_y + 1
		radius = min(bw, bh) * 0.5
	else:
		raise ValueError(f"Unsupported RADIUS_MODE: {RADIUS_MODE}")

	radius *= FULL_DISK_RADIUS_RATIO

	return cx, cy, radius


def clamp_to_disk(rgba: Image.Image, cx: float, cy: float, radius: float) -> Image.Image:
	w, h = rgba.size
	pixels = list(rgba.getdata())

	r2 = radius * radius
	out_pixels = []

	for i, (r, g, b, a) in enumerate(pixels):
		x = i % w
		y = i // w

		px = x + 0.5
		py = y + 0.5

		dx = px - cx
		dy = py - cy

		if dx * dx + dy * dy <= r2:
			out_pixels.append((r, g, b, a))
		else:
			if CLEAR_RGB_WHEN_TRANSPARENT:
				out_pixels.append((0, 0, 0, 0))
			else:
				out_pixels.append((r, g, b, 0))

	out = Image.new("RGBA", rgba.size)
	out.putdata(out_pixels)

	return out


def crop_around_disk(rgba: Image.Image, cx: float, cy: float, radius: float) -> Image.Image:
	w, h = rgba.size

	left = int(cx - radius) - PAD_PIXELS
	top = int(cy - radius) - PAD_PIXELS
	right = int(cx + radius) + PAD_PIXELS
	bottom = int(cy + radius) + PAD_PIXELS

	left = max(0, left)
	top = max(0, top)
	right = min(w, right)
	bottom = min(h, bottom)

	return rgba.crop((left, top, right, bottom))


def make_subdisk(img: Image.Image) -> Image.Image:
	rgba = image_to_rgba(img)

	cx, cy, full_radius = get_disk_center_and_radius(rgba)

	# 1. Clean/clamp the original drawing outside the full disk.
	clean_disk = clamp_to_disk(rgba, cx, cy, full_radius)

	# 2. Clamp again to fetch only the center sub-disk.
	sub_radius = full_radius * SUBDISK_RATIO
	subdisk = clamp_to_disk(clean_disk, cx, cy, sub_radius)

	if CROP_TO_SUBDISK:
		subdisk = crop_around_disk(subdisk, cx, cy, sub_radius)

	return subdisk


def process_image(image_path: Path) -> None:
	with Image.open(image_path) as img:
		out = make_subdisk(img)

	output_path = get_output_path(image_path)
	out.save(output_path)

	print(image_path)
	print(f"  saved: {output_path}")


#%% ----------------------------
# MAIN
# ----------------------------
def main() -> None:
	images = collect_images(Path(SOURCE_PATH).resolve(), RECURSIVE)

	if not images:
		raise RuntimeError(f"No images found in {SOURCE_PATH} with extensions {sorted(INCLUDE_EXTS)}")

	print("MAKE SUB-DISK")
	print(f"SUBDISK_RATIO: {SUBDISK_RATIO}")
	print(f"FULL_DISK_RADIUS_RATIO: {FULL_DISK_RADIUS_RATIO}")
	print(f"CENTER_MODE: {CENTER_MODE}")
	print(f"RADIUS_MODE: {RADIUS_MODE}")
	print(f"CROP_TO_SUBDISK: {CROP_TO_SUBDISK}")
	print(f"OVERWRITE: {OVERWRITE}")

	for image_path in images:
		process_image(image_path)


if __name__ == "__main__":
	main()