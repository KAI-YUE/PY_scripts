# atlas_unpack.py

import os
import json
from pathlib import Path
from PIL import Image

_export_atlas_name = "ui_pack"
_export_atlas_name = "title_pack"
_export_atlas_name = "cards"

# ----------------------------
# CONFIG (edit these)
# ----------------------------
root_dir = os.environ.get(
	"ATLAS_ROOT_DIR",
	# "/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/ui",
	"/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/card/cards/500_aa/"
)
name = os.environ.get("ATLAS_NAME", _export_atlas_name)

SOURCE_ATLAS_PNG = os.path.join(root_dir, "./{:s}.png".format(name))
SOURCE_ATLAS_JSON = os.path.join(root_dir, "./{:s}.json".format(name))
OUTPUT_DIR = os.environ.get("ATLAS_OUTPUT_DIR", os.path.join(root_dir, "./{:s}_pieces".format(name)))

OVERWRITE = True
RESTORE_SOURCE_CANVAS = True			# uses sourceSize/spriteSourceSize when json has them
FORCE_PNG_OUTPUT = True


# ----------------------------
# HELPERS
# ----------------------------
def _read_json(path: Path) -> dict:
	if not path.exists():
		raise FileNotFoundError(f"Atlas json not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _frame_rect(frame: dict) -> tuple[int, int, int, int]:
	rect = frame.get("frame", frame)
	if isinstance(rect, list):
		return int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
	return int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])


def _source_canvas(frame: dict, crop_size: tuple[int, int]) -> tuple[int, int]:
	source = frame.get("sourceSize")
	if not source:
		return crop_size
	if isinstance(source, list):
		return int(source[0]), int(source[1])
	return int(source["w"]), int(source["h"])


def _source_offset(frame: dict) -> tuple[int, int]:
	source = frame.get("spriteSourceSize")
	if not source:
		return 0, 0
	if isinstance(source, list):
		return int(source[0]), int(source[1])
	return int(source.get("x", 0)), int(source.get("y", 0))


def _output_path(output_dir: Path, frame_name: str) -> Path:
	relative = Path(frame_name)
	if relative.is_absolute():
		relative = Path(relative.name)
	if relative.suffix == "" or FORCE_PNG_OUTPUT:
		relative = relative.with_suffix(".png")
	return output_dir / relative


def _unpack_sprite(atlas: Image.Image, frame: dict) -> Image.Image:
	x, y, w, h = _frame_rect(frame)
	sprite = atlas.crop((x, y, x + w, y + h))
	if not RESTORE_SOURCE_CANVAS or "sourceSize" not in frame:
		return sprite

	source_w, source_h = _source_canvas(frame, sprite.size)
	offset_x, offset_y = _source_offset(frame)
	out = Image.new("RGBA", (source_w, source_h), (0, 0, 0, 0))
	out.paste(sprite, (offset_x, offset_y))
	return out


def _write_sprite(img: Image.Image, path: Path) -> None:
	if path.exists() and not OVERWRITE:
		print(f"Skipped existing: {path}")
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	img.save(path, "PNG")


# ----------------------------
# MAIN
# ----------------------------
def dismantle_atlas() -> None:
	atlas_path = Path(SOURCE_ATLAS_PNG)
	json_path = Path(SOURCE_ATLAS_JSON)
	output_dir = Path(OUTPUT_DIR)
	if not atlas_path.exists():
		raise FileNotFoundError(f"Atlas png not found: {atlas_path}")

	meta = _read_json(json_path)
	frames = meta.get("frames", {})
	if not frames:
		raise RuntimeError(f"No frames found in atlas json: {json_path}")

	atlas = Image.open(atlas_path).convert("RGBA")
	count = 0

	for frame_name, frame in frames.items():
		sprite = _unpack_sprite(atlas, frame)
		out_path = _output_path(output_dir, frame_name)
		_write_sprite(sprite, out_path)
		count += 1
		print(f"Wrote: {out_path}")

	print(f"Unpacked {count} sprites from {atlas_path}")
	print(f"Output: {output_dir}")


if __name__ == "__main__":
	dismantle_atlas()
