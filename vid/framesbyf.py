# -*- coding: utf-8 -*-
"""
Extract frames from an MP4 and save them as JPGs.
Works well in Spyder: edit variables, then run the file.
"""

import os
import cv2


def extract_frames_to_jpg(video_path, output_dir, prefix="frame", start_frame=0, step=1, max_frames=None, jpeg_quality=95):
	"""
	video_path: path to input .mp4
	output_dir: folder where jpg frames will be saved
	prefix: filename prefix (e.g., frame_000001.jpg)
	start_frame: first frame index to start from (0-based)
	step: save every Nth frame (1 = save all frames)
	max_frames: limit number of saved frames (None = no limit)
	jpeg_quality: 0-100 (higher = better quality / larger file)
	"""
	if step < 1:
		raise ValueError("step must be >= 1")

	if not os.path.isfile(video_path):
		raise FileNotFoundError(f"Video not found: {video_path}")

	os.makedirs(output_dir, exist_ok=True)

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
	fps = cap.get(cv2.CAP_PROP_FPS)

	print(f"Video: {video_path}")
	print(f"Output folder: {output_dir}")
	print(f"FPS: {fps}")
	print(f"Total frames (may be None/0 for some files): {total_frames}")
	print(f"Start frame: {start_frame}, Step: {step}, Max saved: {max_frames}, JPG quality: {jpeg_quality}")

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	saved_count = 0
	current_frame_index = start_frame

	encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		if (current_frame_index - start_frame) % step == 0:
			out_name = f"{prefix}_{current_frame_index:06d}.jpg"
			out_path = os.path.join(output_dir, out_name)

			wrote = cv2.imwrite(out_path, frame, encode_params)
			if not wrote:
				print(f"Warning: failed to write {out_path}")
			else:
				saved_count += 1

				if saved_count % 100 == 0:
					print(f"Saved {saved_count} frames... (latest: {out_name})")

				if max_frames is not None and saved_count >= max_frames:
					break

		current_frame_index += 1

	cap.release()
	print(f"Done. Saved {saved_count} frame(s).")



# ====== EDIT THESE ======
root_path = r"D:\HMeshi\-1_field_landscape\birds"
video_path = os.path.join(root_path, "1.mp4")
output_dir = os.path.join(root_path, "fbf")
# ========================

extract_frames_to_jpg(
    video_path=video_path,
    output_dir=output_dir,
    prefix="frame",
    start_frame=0,
    step=1,				# 1 = every frame, 2 = every other frame, etc.
    max_frames=None,	# e.g., 500 to stop after saving 500 frames
    jpeg_quality=95
)
