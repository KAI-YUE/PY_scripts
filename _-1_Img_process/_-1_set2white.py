from pathlib import Path

from PIL import Image


#%% ----------------------------
# CONFIG
# ----------------------------
# SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/-1_gemini_uibox/_2_export/tmp/"
SOURCE_PATH = "/mnt/ssd/HMeshi/_2_UI_Uten/_4_gampad_btns/_-1_prev/prev/_01_non_mono/"

OUTPUT_SUFFIX = "_mask"
OVERWRITE = False

ALPHA_THRESHOLD = 1         # pixels >= this alpha are treated as sprite pixels
PRESERVE_ALPHA = True       # keep original opacity shape
RECURSIVE = False

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

	return image_path.with_name(f"{image_path.stem}{OUTPUT_SUFFIX}{image_path.suffix}")


def hard_set_sprite_to_white(img: Image.Image) -> Image.Image:
	rgba = image_to_rgba(img)
	pixels = list(rgba.getdata())

	out_pixels = []

	for r, g, b, a in pixels:
		if a >= ALPHA_THRESHOLD:
			if PRESERVE_ALPHA:
				out_pixels.append((255, 255, 255, a))
			else:
				out_pixels.append((255, 255, 255, 255))
		else:
			out_pixels.append((r, g, b, a))

	out = Image.new("RGBA", rgba.size)
	out.putdata(out_pixels)

	return out


def process_image(image_path: Path) -> None:
	with Image.open(image_path) as img:
		out = hard_set_sprite_to_white(img)

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

	print("HARD SET SPRITE TO WHITE")
	print(f"ALPHA_THRESHOLD: {ALPHA_THRESHOLD}")
	print(f"PRESERVE_ALPHA: {PRESERVE_ALPHA}")
	print(f"OVERWRITE: {OVERWRITE}")

	for image_path in images:
		process_image(image_path)


if __name__ == "__main__":
	main()