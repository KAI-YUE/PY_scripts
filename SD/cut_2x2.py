# Split 2x2 flipbook-style images into 4 separate images (TL/TR/BL/BR)
# Spyder-friendly: edit INPUT_DIR / OUTPUT_DIR / flags, then run.
#
# pip install pillow

from pathlib import Path
from PIL import Image

# =========================
# EDIT THESE PARAMETERS
# =========================
INPUT_DIR = r"/home/kyue/Downloads/nanobanana/"
OUTPUT_DIR = r"/home/kyue/Downloads/outputs/"

ALLOW_ODD = False      # If True, allows odd width/height splits
OVERWRITE = False      # If False, skips files that already exist
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# =========================
# IMPLEMENTATION
# =========================
def split_2x2(img: Image.Image, allow_odd: bool):
	w, h = img.size
	if (w % 2 != 0) or (h % 2 != 0):
		if not allow_odd:
			raise ValueError(f"Image size must be even for exact 2x2 split. Got {w}x{h}. Set ALLOW_ODD=True to proceed.")
	mx = w // 2
	my = h // 2

	return {
		"tl": img.crop((0, 0, mx, my)),
		"tr": img.crop((mx, 0, w, my)),
		"bl": img.crop((0, my, mx, h)),
		"br": img.crop((mx, my, w, h)),
	}

def safe_save(im: Image.Image, out_path: Path):
	ext = out_path.suffix.lower()
	if ext in {".jpg", ".jpeg"} and im.mode in {"RGBA", "LA", "P"}:
		im = im.convert("RGB")
	im.save(out_path)

def main():
	in_dir = Path(INPUT_DIR).expanduser().resolve()
	out_dir = Path(OUTPUT_DIR).expanduser().resolve()

	if not in_dir.exists() or not in_dir.is_dir():
		raise RuntimeError(f"Input folder not found or not a folder: {in_dir}")

	out_dir.mkdir(parents=True, exist_ok=True)

	paths = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
	if not paths:
		print(f"No supported images found in: {in_dir}")
		return

	for p in paths:
		try:
			with Image.open(p) as img:
				img.load()
				crops = split_2x2(img, allow_odd=ALLOW_ODD)

			for label, crop in crops.items():
				out_path = out_dir / f"{p.stem}_{label}{p.suffix}"
				if out_path.exists() and not OVERWRITE:
					print(f"Skip (exists): {out_path.name}")
					continue
				safe_save(crop, out_path)

			print(f"OK: {p.name} -> 4 images")
		except Exception as e:
			print(f"FAIL: {p.name} ({e})")

if __name__ == "__main__":
	main()
