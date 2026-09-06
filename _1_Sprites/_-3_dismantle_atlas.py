# atlas_unpack.py

import os
import json
from pathlib import Path
from PIL import Image

_export_atlas_name = "ui_pack"
_export_atlas_name = "hud_pack"
_export_atlas_name = "icon_pack"
_export_atlas_name = "title_pack"
_export_atlas_name = "colorful_bg"
# _export_atlas_name = "inter_btn_pack"
# _export_atlas_name = "card_pawn_icon_pack"

_export_atlas_name = "right_bubbles"
_export_atlas_name = "theme_pack"

dir = "/mnt/ssd/HMeshi/_6_Lua/HM/resources/textures/ui/buttons/"

# ----------------------------
# CONFIG (edit these)
# ----------------------------
root_dir = os.environ.get("ATLAS_ROOT_DIR", dir)
name = os.environ.get("ATLAS_NAME", _export_atlas_name)

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


def _find_atlas_pairs(root: Path) -> list[tuple[str, Path, Path]]:
	pairs = []
	for json_path in root.glob("*.json"):
		png_path = json_path.with_suffix(".png")
		if png_path.exists():
			pairs.append((json_path.stem, png_path, json_path))
	return pairs


def _resolve_atlas(root: Path, preferred_name: str) -> tuple[str, Path, Path]:
	atlas_path = root / f"{preferred_name}.png"
	json_path = root / f"{preferred_name}.json"
	if atlas_path.exists() and json_path.exists():
		return preferred_name, atlas_path, json_path

	pairs = _find_atlas_pairs(root)
	if not pairs:
		raise FileNotFoundError(
			f"No atlas png/json pair found in {root}. Tried: {atlas_path} and {json_path}"
		)

	def newest_mtime(pair: tuple[str, Path, Path]) -> float:
		_, png_path, json_path = pair
		return max(png_path.stat().st_mtime, json_path.stat().st_mtime)

	resolved_name, resolved_png, resolved_json = max(pairs, key=newest_mtime)
	print(
		f"Atlas '{preferred_name}' not found; using detected atlas '{resolved_name}' "
		f"from {root}"
	)
	return resolved_name, resolved_png, resolved_json


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
	root_path = Path(root_dir)
	resolved_name, atlas_path, json_path = _resolve_atlas(root_path, name)
	output_dir = Path(
		os.environ.get("ATLAS_OUTPUT_DIR", root_path / f"{resolved_name}_pieces")
	)

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
