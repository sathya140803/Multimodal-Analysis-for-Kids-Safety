import cv2
import os
import math

# SAVE LOCATION
BASE_PATH = r"D:\CollectedImagesNew"
IMG_SIZE  = 260   # Resize all frames to 260x260 for EfficientNetB2


# GPU SETUP
# OpenCV can use CUDA for resizing if available
# Falls back to CPU automatically if not
USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_GPU:
    print(f" OpenCV CUDA detected — using GPU for resizing")
else:
    print(f" No OpenCV CUDA — using CPU for resizing (still fast)")

# LOCAL VIDEO FILES
SAFE_VIDEOS = {
    "peppa_pig": [
        r"D:\CollectedVideos\safe\peppa_pig\peppapig1.mp4",
        r"D:\CollectedVideos\safe\peppa_pig\peppapig2.mp4",
        r"D:\CollectedVideos\safe\peppa_pig\peppapig3.mp4",
        r"D:\CollectedVideos\safe\peppa_pig\peppapig4.mp4",
        r"D:\CollectedVideos\safe\peppa_pig\peppapig5.mp4",
        r"D:\CollectedVideos\safe\peppa_pig\peppapig6.mp4",
    ],
    "mickey_mouse": [
        r"D:\CollectedVideos\safe\mickeymouse\m1.mp4",
    ],
    "masha_and_bear": [
        r"D:\CollectedVideos\safe\masha_and_bear\mb1.mp4",
        r"D:\CollectedVideos\safe\masha_and_bear\mb2.mp4",
        r"D:\CollectedVideos\safe\masha_and_bear\mb3.mp4",
        r"D:\CollectedVideos\safe\masha_and_bear\mb4.mp4",
        r"D:\CollectedVideos\safe\masha_and_bear\mb5.mp4",
        r"D:\CollectedVideos\safe\masha_and_bear\mb6.mp4",
    ],
    "pj_masks": [
        r"D:\CollectedVideos\safe\pj_masks\pj1.mp4",
        r"D:\CollectedVideos\safe\pj_masks\pj2.mp4",
        r"D:\CollectedVideos\safe\pj_masks\pj3.mp4",
        r"D:\CollectedVideos\safe\pj_masks\pj4.mp4",
        r"D:\CollectedVideos\safe\pj_masks\pj5.mp4",
        r"D:\CollectedVideos\safe\pj_masks\pj6.mp4",
    ],
    "randomSafe": [
        r"D:\CollectedVideos\safe\randomSafe\Rs1.mp4",
        r"D:\CollectedVideos\safe\randomSafe\Rs2.mp4",
        r"D:\CollectedVideos\safe\randomSafe\Rs3.mp4",
        r"D:\CollectedVideos\safe\randomSafe\Rs4.mp4",
    ],
}

UNSAFE_VIDEOS = {
    "south_park": [
        r"D:\CollectedVideos\unsafe\south_park\s1.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s2.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s3.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s4.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s5.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s6.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s7.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s8.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s9.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s10.mp4",
        r"D:\CollectedVideos\unsafe\south_park\s11.mp4",
    ],
    "family_guy": [
        r"D:\CollectedVideos\unsafe\family_guy\f1.mp4",
        r"D:\CollectedVideos\unsafe\family_guy\f2.mp4",
        r"D:\CollectedVideos\unsafe\family_guy\f3.mp4",
        r"D:\CollectedVideos\unsafe\family_guy\f4.mp4",
        r"D:\CollectedVideos\unsafe\family_guy\f5.mp4",
        r"D:\CollectedVideos\unsafe\family_guy\f6.mp4",
    ],
    "simpsons": [
        r"D:\CollectedVideos\unsafe\simpsons\s1.mp4",
        r"D:\CollectedVideos\unsafe\simpsons\s2.mp4",
        r"D:\CollectedVideos\unsafe\simpsons\s3.mp4",
        r"D:\CollectedVideos\unsafe\simpsons\s4.mp4",
        r"D:\CollectedVideos\unsafe\simpsons\s5.mp4",
        r"D:\CollectedVideos\unsafe\simpsons\s6.mp4",
    ],
    "rick_and_monty": [
        r"D:\CollectedVideos\unsafe\rick_and_monty\r1.mp4",
        r"D:\CollectedVideos\unsafe\rick_and_monty\r2.mp4",
        r"D:\CollectedVideos\unsafe\rick_and_monty\r3.mp4",
        r"D:\CollectedVideos\unsafe\rick_and_monty\r4.mp4",
        r"D:\CollectedVideos\unsafe\rick_and_monty\r5.mp4",
        r"D:\CollectedVideos\unsafe\rick_and_monty\r6.mp4",
    ],
    "bojack_horseman": [
        r"D:\CollectedVideos\unsafe\bojack_horseman\b1.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b2.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b3.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b4.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b5.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b6.mp4",
        r"D:\CollectedVideos\unsafe\bojack_horseman\b7.mp4",
    ],
    "RandomUnsafe": [
        r"D:\CollectedVideos\unsafe\RandomUnsafe\r1.mp4",
        r"D:\CollectedVideos\unsafe\RandomUnsafe\r2.mp4",
        r"D:\CollectedVideos\unsafe\RandomUnsafe\r3.mp4",
        r"D:\CollectedVideos\unsafe\RandomUnsafe\r4.mp4",
        r"D:\CollectedVideos\unsafe\RandomUnsafe\r5.mp4",
    ],
}

# HOW MANY TOTAL FRAMES PER SHOW
FRAME_LIMITS = {
    "peppa_pig":      2000,
    "mickey_mouse":   2000,
    "masha_and_bear": 2000,
    "pj_masks":       2000,
    "randomSafe":     1000,
    "south_park":     2000,
    "family_guy":     2000,
    "simpsons":       2000,
    "rick_and_monty": 2000,
    "bojack_horseman":2000,
    "RandomUnsafe":   1000,
}

VAL_RATIO = 0.2
# RESIZE FRAME — GPU or CPU
def resize_frame(frame):
    if USE_GPU:
        # Upload frame to GPU, resize, download back
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_resized = cv2.cuda.resize(
            gpu_frame, (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        return gpu_resized.download()
    else:
        # CPU resize
        return cv2.resize(frame, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_LINEAR)


# EXTRACT FRAMES FROM A VIDEO
def extract_frames(video_path, output_folder,
                   prefix="frame", frame_rate=1,
                   max_frames=999999,
                   start_frame=0, end_frame=None):

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"   File not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = (end_frame - start_frame) / fps
    print(f"  Original: {width}x{height} → Saved: {IMG_SIZE}x{IMG_SIZE} "
          f"| FPS: {fps:.1f} | Section: {duration_sec:.0f}s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    saved_count = 0
    existing    = len(os.listdir(output_folder))
    interval    = max(1, int(fps / frame_rate))

    while cap.isOpened():
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_pos >= end_frame:
            break
        if saved_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            # Resize frame before saving
            frame_resized = resize_frame(frame)
            filename  = f"{prefix}_{existing + saved_count:06d}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, frame_resized)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


# PROCESS SHOW WITH MULTIPLE VIDEOS
def process_show_multi_video(show_name, video_paths, label):
    train_folder = os.path.join(BASE_PATH, "train", label)
    val_folder   = os.path.join(BASE_PATH, "val",   label)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder,   exist_ok=True)

    target_frames = FRAME_LIMITS.get(show_name, 1000)
    valid_paths   = [p for p in video_paths if p and os.path.exists(p)]

    if not valid_paths:
        print(f"  ⚠️  No valid files for [{show_name}] — skipping")
        return 0

    val_count   = max(1, math.ceil(len(valid_paths) * VAL_RATIO))
    train_paths = valid_paths[:-val_count]
    val_paths   = valid_paths[-val_count:]

    print(f"\n  [{show_name}]")
    print(f"  Total videos : {len(valid_paths)}")
    print(f"  Train videos : {len(train_paths)} → {[os.path.basename(p) for p in train_paths]}")
    print(f"  Val videos   : {len(val_paths)}   → {[os.path.basename(p) for p in val_paths]}")
    print(f"  Target frames: {target_frames}")
    print(f"  {'-'*40}")

    # ── TRAIN ──
    train_target     = int(target_frames * (1 - VAL_RATIO))
    frames_per_train = train_target // len(train_paths) if train_paths else 0
    remainder        = train_target % len(train_paths)  if train_paths else 0
    train_saved      = 0

    print(f"  TRAIN ({len(train_paths)} videos, ~{frames_per_train} frames each)")
    for i, path in enumerate(train_paths):
        limit = frames_per_train + (remainder if i == len(train_paths) - 1 else 0)
        print(f"    [{i+1}/{len(train_paths)}] {os.path.basename(path)} — up to {limit} frames")
        saved = extract_frames(
            video_path=path, output_folder=train_folder,
            prefix=show_name, frame_rate=1, max_frames=limit
        )
        train_saved += saved
        print(f"  {saved} frames saved")

    # ── VAL ──
    val_target     = int(target_frames * VAL_RATIO)
    frames_per_val = val_target // len(val_paths) if val_paths else 0
    val_remainder  = val_target % len(val_paths)  if val_paths else 0
    val_saved      = 0

    print(f"  VAL ({len(val_paths)} videos, ~{frames_per_val} frames each)")
    for i, path in enumerate(val_paths):
        limit = frames_per_val + (val_remainder if i == len(val_paths) - 1 else 0)
        print(f"    [{i+1}/{len(val_paths)}] {os.path.basename(path)} — up to {limit} frames")
        saved = extract_frames(
            video_path=path, output_folder=val_folder,
            prefix=show_name, frame_rate=1, max_frames=limit
        )
        val_saved += saved
        print(f"  {saved} frames saved")

    total = train_saved + val_saved
    print(f"\n  {show_name} COMPLETE — "
          f"Train: {train_saved} | Val: {val_saved} | Total: {total}")
    return total

# PROCESS SHOW WITH SINGLE LONG VIDEO
def process_show_single_video(show_name, video_path, label):
    train_folder = os.path.join(BASE_PATH, "train", label)
    val_folder   = os.path.join(BASE_PATH, "val",   label)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder,   exist_ok=True)

    target_frames = FRAME_LIMITS.get(show_name, 1000)

    if not os.path.exists(video_path):
        print(f"   File not found: {video_path}")
        return 0

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    split_frame  = int(total_frames * (1 - VAL_RATIO))
    duration_min = total_frames / fps / 60

    print(f"\n  [{show_name}] — Single long video")
    print(f"  Duration     : {duration_min:.1f} minutes")
    print(f"  Total frames : {total_frames}")
    print(f"  Train section: frame 0 → {split_frame} ({split_frame/fps/60:.1f} min)")
    print(f"  Val section  : frame {split_frame} → {total_frames} "
          f"({(total_frames-split_frame)/fps/60:.1f} min)")
    print(f"  {'-'*40}")

    train_target = int(target_frames * (1 - VAL_RATIO))
    print(f"  TRAIN — extracting up to {train_target} frames")
    train_saved = extract_frames(
        video_path=video_path, output_folder=train_folder,
        prefix=show_name, frame_rate=1, max_frames=train_target,
        start_frame=0, end_frame=split_frame
    )
    print(f"  Train: {train_saved} frames saved")

    val_target = int(target_frames * VAL_RATIO)
    print(f"  VAL — extracting up to {val_target} frames")
    val_saved = extract_frames(
        video_path=video_path, output_folder=val_folder,
        prefix=show_name, frame_rate=1, max_frames=val_target,
        start_frame=split_frame, end_frame=total_frames
    )
    print(f"  Val: {val_saved} frames saved")

    total = train_saved + val_saved
    print(f"\n  {show_name} COMPLETE — "
          f"Train: {train_saved} | Val: {val_saved} | Total: {total}")
    return total

# MAIN
if __name__ == "__main__":

    print("=" * 55)
    print("   FRAME EXTRACTION — VIDEO-LEVEL SPLIT")
    print(f"   Saving to  : {BASE_PATH}")
    print(f"   Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Train/Val  : {int((1-VAL_RATIO)*100)}/{int(VAL_RATIO*100)}")
    print(f"   Resize GPU : {' Yes' if USE_GPU else '  No (CPU)'}")
    print("=" * 55)

    print("\n SAFE VIDEOS")
    print("=" * 55)
    for show_name, paths in SAFE_VIDEOS.items():
        if show_name == "mickey_mouse":
            process_show_single_video(show_name, paths[0], label="safe")
        else:
            process_show_multi_video(show_name, paths, label="safe")

    print("\n  UNSAFE VIDEOS")
    print("=" * 55)
    for show_name, paths in UNSAFE_VIDEOS.items():
        process_show_multi_video(show_name, paths, label="unsafe")

    print("\n========== FINAL DATASET SUMMARY ==========")
    total = 0
    for split in ["train", "val"]:
        for label in ["safe", "unsafe"]:
            folder = os.path.join(BASE_PATH, split, label)
            if os.path.exists(folder):
                count = len(os.listdir(folder))
                total += count
                print(f"  {split}/{label}: {count} images")
    print(f"\n  GRAND TOTAL: {total} images")
    print("============================================")
    print(" Dataset ready! You can now run train.py")