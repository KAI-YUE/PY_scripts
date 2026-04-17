"""
Patch quilting texture synthesis.

Spyder usage:
1. Edit the CONFIG block below.
2. Press Run.
3. Inspect DST_DIR for synthesized tiles and tiled preview sheets.
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


# =============================================================================
# CONFIG - edit these in Spyder
# =============================================================================

src_dir = "/mnt/ssd/HMeshi/-1_field_landscape/grass/tiles/"

SOURCE_PATH = Path(src_dir, "tt3.png")
DST_DIR = Path(src_dir, "patch_quilting_output")

OUTPUT_W = 256
OUTPUT_H = 256
NUM_OUTPUTS = 4
RANDOM_SEED = 1234
OVERWRITE = True

# Quilting parameters.
PATCH_SIZE = 80
OVERLAP = 28
CANDIDATES_PER_PATCH = 120
TOP_K_RANDOM = 8

# Optional transforms applied to candidate patches before scoring/blending.
# Add "rot" to sample a random angle from ROT_ANGLE_MIN..ROT_ANGLE_MAX.
USE_PATCH_TRANSFORMS = True
PATCH_TRANSFORMS = [
    "orig",
    # "flip_x",
    # "flip_y",
    # "flip_xy",
    # "rot180",
    "rot",
]
ROT_ANGLE_MIN = 0.0
ROT_ANGLE_MAX = 360.0

# Patch paste mode: "hard", "feather", or "poisson".
# "feather" is usually the best first choice for reducing cut-edge artifacts.
BLEND_MODE = "feather"
SEAM_FEATHER_RADIUS = 2

# Optional limited color correction before patch paste.
USE_OVERLAP_COLOR_CORRECTION = True
MAX_COLOR_SHIFT = 0.08               # RGB shift cap in 0..1 space
COLOR_CORRECTION_STRENGTH = 0.75

# Optional Poisson/gradient-domain blending. Used only when BLEND_MODE="poisson".
POISSON_MODE = "normal"              # "normal" or "mixed"

# Larger synthesis gives the script room to search for a naturally better
# seamless crop. Set to 0 to directly synthesize OUTPUT_W x OUTPUT_H.
SEAMLESS_CROP_MARGIN = 96
SEAM_SCORE_EDGE = 10
SEAM_CROP_STRIDE = 4

# Optional source preparation.
CROP_SOURCE_TO_SQUARE = False
SOURCE_RESIZE_LONG_EDGE = None       # example: 512; set None to keep source size
PIXEL_ART = False

# Preview output.
SAVE_TILED_PREVIEW = True
PREVIEW_REPEAT = 3
PREVIEW_DIR_NAME = "previews"

# Optional transformed variants from each synthesized tile.
SAVE_TRANSFORM_VARIANTS = False
TRANSFORMS = [
    # "flip_x",
    # "flip_y",
    # "flip_xy",
    # "rot180",
    "rot",
]


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def to_np(img):
    return np.array(img).astype(np.float32) / 255.0


def to_img(arr):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def prepare_source(path):
    img = Image.open(path).convert("RGB")

    if CROP_SOURCE_TO_SQUARE:
        img = center_crop_square(img)

    if SOURCE_RESIZE_LONG_EDGE is not None:
        w, h = img.size
        scale = float(SOURCE_RESIZE_LONG_EDGE) / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resample = Image.Resampling.NEAREST if PIXEL_ART else Image.Resampling.BICUBIC
        img = img.resize((nw, nh), resample=resample)

    return to_np(img)


def make_tiled_preview(img, repeat=3):
    w, h = img.size
    preview = Image.new(img.mode, (w * repeat, h * repeat))
    for y in range(repeat):
        for x in range(repeat):
            preview.paste(img, (x * w, y * h))
    return preview


def transform_image(img, transform, rng=None):
    if transform == "orig":
        return img.copy()
    if transform == "flip_x":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform == "flip_y":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform == "flip_xy":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform == "rot90":
        return img.transpose(Image.Transpose.ROTATE_90)
    if transform == "rot180":
        return img.transpose(Image.Transpose.ROTATE_180)
    if transform == "rot270":
        return img.transpose(Image.Transpose.ROTATE_270)
    if transform == "rot":
        if rng is None:
            rng = np.random.default_rng()
        angle = rng.uniform(ROT_ANGLE_MIN, ROT_ANGLE_MAX)
        return rotate_image_center_crop(img, angle)

    raise ValueError(f"Unknown transform: {transform}")


def rotate_image_center_crop(img, angle):
    if cv2 is not None:
        arr = np.array(img.convert("RGB"))
        h, w, _ = arr.shape
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            arr,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return Image.fromarray(rotated)

    return img.rotate(
        float(angle),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=tuple(int(v) for v in np.array(img).mean(axis=(0, 1))),
    )


def transform_patch(patch, transform, rng):
    if transform == "orig":
        return patch
    if transform == "flip_x":
        return np.flip(patch, axis=1)
    if transform == "flip_y":
        return np.flip(patch, axis=0)
    if transform == "flip_xy":
        return np.flip(np.flip(patch, axis=1), axis=0)
    if transform == "rot90":
        return np.rot90(patch, k=1)
    if transform == "rot180":
        return np.rot90(patch, k=2)
    if transform == "rot270":
        return np.rot90(patch, k=3)
    if transform == "rot":
        angle = rng.uniform(ROT_ANGLE_MIN, ROT_ANGLE_MAX)
        img = to_img(patch)
        return to_np(rotate_image_center_crop(img, angle))

    raise ValueError(f"Unknown patch transform: {transform}")


def random_patch_transform(rng):
    if not USE_PATCH_TRANSFORMS or not PATCH_TRANSFORMS:
        return "orig"

    return PATCH_TRANSFORMS[int(rng.integers(0, len(PATCH_TRANSFORMS)))]


def required_source_sample_size(patch_size):
    if USE_PATCH_TRANSFORMS and "rot" in PATCH_TRANSFORMS:
        return int(np.ceil(patch_size * np.sqrt(2.0)))
    return patch_size


def crop_center_np(arr, size):
    h, w, _ = arr.shape
    y = (h - size) // 2
    x = (w - size) // 2
    return arr[y:y + size, x:x + size, :]


def save_image_and_preview(img, out_path):
    img.save(out_path)
    print(f"Wrote: {out_path}")

    if SAVE_TILED_PREVIEW:
        preview_dir = out_path.parent / PREVIEW_DIR_NAME
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{out_path.stem}_preview{out_path.suffix}"
        preview = make_tiled_preview(img, repeat=PREVIEW_REPEAT)
        preview.save(preview_path)
        print(f"Wrote: {preview_path}")


# =============================================================================
# MINIMUM-ERROR CUTS
# =============================================================================

def vertical_min_cut(cost):
    """Return one seam x-coordinate for each row in a vertical overlap."""
    h, w = cost.shape
    dp = cost.copy()
    back = np.zeros((h, w), dtype=np.int32)

    for y in range(1, h):
        for x in range(w):
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            prev = dp[y - 1, x0:x1]
            best = int(np.argmin(prev)) + x0
            dp[y, x] += dp[y - 1, best]
            back[y, x] = best

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = back[y + 1, seam[y + 1]]

    return seam


def apply_left_cut(use_patch, patch, target_region, overlap):
    cost = ((patch[:, :overlap, :] - target_region[:, :overlap, :]) ** 2).sum(axis=2)
    seam = vertical_min_cut(cost)

    for y, sx in enumerate(seam):
        use_patch[y, :sx] = False

    return use_patch


def apply_top_cut(use_patch, patch, target_region, overlap):
    cost = ((patch[:overlap, :, :] - target_region[:overlap, :, :]) ** 2).sum(axis=2)
    seam = vertical_min_cut(cost.T)

    for x, sy in enumerate(seam):
        use_patch[:sy, x] = False

    return use_patch


# =============================================================================
# POISSON BLENDING
# =============================================================================

def poisson_blend_patch(target_region, filled_region, patch, use_patch):
    if cv2 is None:
        raise RuntimeError('BLEND_MODE="poisson" requires OpenCV/cv2')

    if use_patch.sum() < 16:
        return patch

    # The quilting canvas is only partially filled while patches are placed.
    # Prefill empty destination pixels with the source patch so Poisson blending
    # does not solve against black pixels outside the already-filled overlap.
    clone_dst = target_region.copy()
    clone_dst[~filled_region] = patch[~filled_region]

    src = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    dst = np.clip(clone_dst * 255.0, 0, 255).astype(np.uint8)
    mask = np.where(use_patch, 255, 0).astype(np.uint8)

    # OpenCV expects BGR input. The patch and destination are the same local size.
    src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    h, w = mask.shape
    center = (w // 2, h // 2)
    mode = cv2.MIXED_CLONE if POISSON_MODE == "mixed" else cv2.NORMAL_CLONE
    blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask, center, mode)
    blended = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    return blended.astype(np.float32) / 255.0


def feather_alpha(use_patch, filled_region, radius):
    if radius <= 0:
        return use_patch.astype(np.float32)

    # Empty pixels are new patch territory. Keep them fully opaque so only the
    # already-filled overlap gets softened around the min-cut seam.
    overlap = filled_region
    if not overlap.any():
        return use_patch.astype(np.float32)

    if cv2 is not None:
        patch_side = use_patch.astype(np.uint8)
        old_side = (~use_patch).astype(np.uint8)
        dist_patch = cv2.distanceTransform(patch_side, cv2.DIST_L2, 3)
        dist_old = cv2.distanceTransform(old_side, cv2.DIST_L2, 3)
    else:
        dist_patch = slow_distance_to_false(use_patch)
        dist_old = slow_distance_to_false(~use_patch)

    denom = dist_patch + dist_old
    alpha = np.divide(
        dist_patch,
        denom,
        out=use_patch.astype(np.float32),
        where=denom > 0,
    )

    seam_band = np.logical_and(overlap, denom <= radius * 2.0 + 1.0)
    hard_alpha = use_patch.astype(np.float32)
    alpha = np.where(seam_band, alpha, hard_alpha)
    alpha = np.where(~filled_region, 1.0, alpha)

    return np.clip(alpha, 0.0, 1.0)


def slow_distance_to_false(mask):
    h, w = mask.shape
    false_points = np.argwhere(~mask)
    if len(false_points) == 0:
        return np.full((h, w), max(h, w), dtype=np.float32)

    dist = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dy = false_points[:, 0] - y
            dx = false_points[:, 1] - x
            dist[y, x] = np.sqrt(np.min(dx * dx + dy * dy))

    return dist


# =============================================================================
# COLOR CORRECTION
# =============================================================================

def correct_patch_color(patch, target_region, overlap_mask):
    if not USE_OVERLAP_COLOR_CORRECTION or not overlap_mask.any():
        return patch

    patch_mean = patch[overlap_mask].mean(axis=0)
    target_mean = target_region[overlap_mask].mean(axis=0)
    shift = (target_mean - patch_mean) * COLOR_CORRECTION_STRENGTH
    shift = np.clip(shift, -MAX_COLOR_SHIFT, MAX_COLOR_SHIFT)

    return np.clip(patch + shift.reshape(1, 1, 3), 0.0, 1.0)


# =============================================================================
# PATCH QUILTING
# =============================================================================

def grid_positions(total, patch_size, step):
    positions = [0]
    pos = step
    while pos + patch_size < total:
        positions.append(pos)
        pos += step

    last = total - patch_size
    if last > positions[-1]:
        positions.append(last)

    return positions


def random_source_patch(source, patch_size, rng):
    src_h, src_w, _ = source.shape
    transform = random_patch_transform(rng)
    sample_size = patch_size
    if transform == "rot":
        sample_size = int(np.ceil(patch_size * np.sqrt(2.0)))

    if src_w < sample_size or src_h < sample_size:
        raise ValueError(
            f"Source image is too small for transform='{transform}' with "
            f"sample_size={sample_size}: source is {src_w}x{src_h}"
        )

    y = int(rng.integers(0, src_h - sample_size + 1))
    x = int(rng.integers(0, src_w - sample_size + 1))
    patch = source[y:y + sample_size, x:x + sample_size, :]
    patch = transform_patch(patch, transform, rng)

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = crop_center_np(patch, patch_size)

    return patch.copy()


def score_patch(patch, target_region, filled_region):
    if not filled_region.any():
        return 0.0

    diff = patch - target_region
    diff = diff[filled_region]
    return float((diff * diff).mean())


def choose_patch(source, target_region, filled_region, patch_size, rng):
    best = []
    for _ in range(CANDIDATES_PER_PATCH):
        patch = random_source_patch(source, patch_size, rng)
        score = score_patch(patch, target_region, filled_region)
        best.append((score, patch))

    best.sort(key=lambda item: item[0])
    top_n = min(TOP_K_RANDOM, len(best))
    _, chosen = best[int(rng.integers(0, top_n))]
    return chosen.copy()


def paste_patch(output, filled, patch, x, y, overlap):
    patch_h, patch_w, _ = patch.shape
    target_region = output[y:y + patch_h, x:x + patch_w, :]
    filled_region = filled[y:y + patch_h, x:x + patch_w]

    use_patch = np.ones((patch_h, patch_w), dtype=bool)

    if x > 0:
        col_filled = filled_region.any(axis=0)
        left_overlap = 0
        while left_overlap < patch_w and col_filled[left_overlap]:
            left_overlap += 1
        left_overlap = max(1, min(left_overlap, patch_w))
        use_patch = apply_left_cut(use_patch, patch, target_region, left_overlap)

    if y > 0:
        row_filled = filled_region.any(axis=1)
        top_overlap = 0
        while top_overlap < patch_h and row_filled[top_overlap]:
            top_overlap += 1
        top_overlap = max(1, min(top_overlap, patch_h))
        use_patch = apply_top_cut(use_patch, patch, target_region, top_overlap)

    patch = correct_patch_color(patch, target_region, filled_region)

    # Keep existing pixels wherever this patch has no overlap-based authority.
    use_patch = np.logical_or(use_patch, ~filled_region)

    if BLEND_MODE == "poisson" and filled_region.any():
        blended = poisson_blend_patch(target_region, filled_region, patch, use_patch)
        target_region[use_patch] = blended[use_patch]
    elif BLEND_MODE == "feather" and filled_region.any():
        alpha = feather_alpha(use_patch, filled_region, SEAM_FEATHER_RADIUS)
        target_region[:, :, :] = target_region * (1.0 - alpha[..., None]) + patch * alpha[..., None]
    elif BLEND_MODE == "hard" or not filled_region.any():
        target_region[use_patch] = patch[use_patch]
    else:
        raise ValueError(f"Unknown BLEND_MODE: {BLEND_MODE}")

    filled_region[:, :] = True


def quilt_texture(source, out_w, out_h, patch_size, overlap, rng):
    src_h, src_w, _ = source.shape
    sample_size = required_source_sample_size(patch_size)
    if src_w < sample_size or src_h < sample_size:
        raise ValueError(
            f"Source image is too small for PATCH_SIZE={patch_size} "
            f"and required sample size {sample_size}: "
            f"source is {src_w}x{src_h}"
        )
    if overlap <= 0 or overlap >= patch_size:
        raise ValueError("OVERLAP must be greater than 0 and smaller than PATCH_SIZE")

    step = patch_size - overlap
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    filled = np.zeros((out_h, out_w), dtype=bool)

    xs = grid_positions(out_w, patch_size, step)
    ys = grid_positions(out_h, patch_size, step)

    for y in ys:
        for x in xs:
            target_region = output[y:y + patch_size, x:x + patch_size, :]
            filled_region = filled[y:y + patch_size, x:x + patch_size]
            patch = choose_patch(source, target_region, filled_region, patch_size, rng)
            paste_patch(output, filled, patch, x, y, overlap)

    return output


# =============================================================================
# SEAMLESS CROP
# =============================================================================

def edge_score(tile, edge):
    edge = min(edge, tile.shape[0] // 2, tile.shape[1] // 2)
    left = tile[:, :edge, :]
    right = tile[:, -edge:, :]
    top = tile[:edge, :, :]
    bottom = tile[-edge:, :, :]

    lr = ((left - right) ** 2).mean()
    tb = ((top - bottom) ** 2).mean()
    return float(lr + tb)


def best_seamless_crop(canvas, crop_w, crop_h, edge, stride):
    canvas_h, canvas_w, _ = canvas.shape
    if canvas_w == crop_w and canvas_h == crop_h:
        return canvas

    best_score = None
    best_crop = None

    for y in range(0, canvas_h - crop_h + 1, stride):
        for x in range(0, canvas_w - crop_w + 1, stride):
            crop = canvas[y:y + crop_h, x:x + crop_w, :]
            score = edge_score(crop, edge)
            if best_score is None or score < best_score:
                best_score = score
                best_crop = crop

    return best_crop.copy()


# =============================================================================
# RUNNERS
# =============================================================================

def synthesize_one(source_path, dst_dir, index, rng):
    dst_dir.mkdir(parents=True, exist_ok=True)

    out_path = dst_dir / f"{index:02d}.png"

    if out_path.exists() and not OVERWRITE:
        print(f"Skipped existing: {out_path}")
        return out_path

    source = prepare_source(source_path)
    canvas_w = OUTPUT_W + 2 * SEAMLESS_CROP_MARGIN
    canvas_h = OUTPUT_H + 2 * SEAMLESS_CROP_MARGIN

    canvas = quilt_texture(source, canvas_w, canvas_h, PATCH_SIZE, OVERLAP, rng)
    tile = best_seamless_crop(
        canvas,
        OUTPUT_W,
        OUTPUT_H,
        edge=SEAM_SCORE_EDGE,
        stride=SEAM_CROP_STRIDE,
    )

    img = to_img(tile)
    save_image_and_preview(img, out_path)

    if SAVE_TRANSFORM_VARIANTS:
        for transform in TRANSFORMS:
            variant = transform_image(img, transform, rng)
            variant_path = Path(dst_dir) / f"{index:02d}_{transform}.png"
            if variant_path.exists() and not OVERWRITE:
                print(f"Skipped existing: {variant_path}")
                continue
            save_image_and_preview(variant, variant_path)

    return out_path


def synthesize_many(source_path, dst_dir, count):
    rng = np.random.default_rng(RANDOM_SEED)
    return [synthesize_one(source_path, dst_dir, i + 1, rng) for i in range(count)]


if __name__ == "__main__":
    synthesize_many(SOURCE_PATH, DST_DIR, NUM_OUTPUTS)
