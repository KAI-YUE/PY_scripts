import os
from PIL import Image

# =========================
# EDIT THESE
# =========================
INPUT_DIR = r"/mnt/ssd/HMeshi/card_design/bg_brush/ref/"
OUTPUT_DIR = r"/mnt/ssd/HMeshi/card_design/bg_brush/output/"

INPUT_DIR = r"/home/kyue/Desktop/inspiration/ui/ref/"
OUTPUT_DIR = r"/home/kyue/Desktop/inspiration/ui/output/"

# Target rectangle size
TARGET_WIDTH  = 1024
TARGET_HEIGHT = 1024

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FORCE_PNG_OUTPUT = True  # recommended

def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def sample_top_right_5_percent_color(img_rgba):
	w, h = img_rgba.size

	# (95%, 5%) from top-left coordinates -> near top-right
	x = int(round(w * 0.95))
	y = int(round(h * 0.05))

	# Clamp
	if x < 0:
		x = 0
	if x > w - 1:
		x = w - 1
	if y < 0:
		y = 0
	if y > h - 1:
		y = h - 1

	r, g, b, a = img_rgba.getpixel((x, y))

	# If sampled point is transparent, force opaque using its RGB anyway
	# (Better than writing transparent background)
	return (r, g, b, 255)


def pad_to_square_centered(img):
	rgba = img.convert("RGBA")
	ori_w, ori_h = rgba.size
	S = max(ori_w, ori_h)

	fill_color = sample_top_right_5_percent_color(rgba)
	bg = Image.new("RGBA", (S, S), fill_color)

	offset_x = (S - ori_w) // 2
	offset_y = (S - ori_h) // 2

	# Paste with alpha mask so transparent parts stay transparent over bg fill
	bg.paste(rgba, (offset_x, offset_y), mask=rgba.split()[-1])

	return bg



ensure_dir(OUTPUT_DIR)

for name in os.listdir(INPUT_DIR):
    in_path = os.path.join(INPUT_DIR, name)
    if not os.path.isfile(in_path):
        continue

    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTS:
        continue

    base = os.path.splitext(name)[0]
    out_name = base + (".png" if FORCE_PNG_OUTPUT else ext)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        img = Image.open(in_path)
        out = pad_to_square_centered(img)

        # If not forcing PNG and output is JPG, drop alpha
        if (not FORCE_PNG_OUTPUT) and ext in {".jpg", ".jpeg"}:
            out = out.convert("RGB")

        out.save(out_path)
        print(f"DONE: {name} -> {out_name}")

    except Exception as e:
        print(f"FAIL: {name} -> {e}")