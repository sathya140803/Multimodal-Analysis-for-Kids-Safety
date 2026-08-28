"""
services/video_analyser.py
===========================
Wraps the EfficientNet visual analysis from VideoPredicitions.py.
Downloads the video, processes frames, then deletes the temp file.

Returns a standardised result dict:
{
    "verdict":           "SAFE" | "UNSAFE" | "ERROR",
    "score":             float  (unsafe_ratio, 0.0 - 1.0),
    "unsafe_frames":     int,
    "total_frames":      int,
    "unsafe_ratio":      float,
    "unsafe_timestamps": list of {"timestamp_str": str, "confidence": float},
    "error":             str | None
}
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model

from config import (
    VIDEO_MODEL_PATH,
    VIDEO_IMG_SIZE,
    VIDEO_FRAME_RATE,
    VIDEO_THRESHOLD,
    VIDEO_UNSAFE_RATIO,
    TEMP_VIDEO_DIR,
)

# ── lazy-loaded globals ──────────────────────────────────────────────────────
_model  = None
_device = None

# ── Transform — must match training exactly ──────────────────────────────────
class _ScaleTo255:
    def __call__(self, x):
        return x * 255.0

_transform = transforms.Compose([
    transforms.Resize((VIDEO_IMG_SIZE, VIDEO_IMG_SIZE)),
    transforms.ToTensor(),
    _ScaleTo255(),
])


class _CartoonClassifier(nn.Module):
    def __init__(self, base_model, feature_size):
        super().__init__()
        self.base = base_model
        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)


def _load_model():
    global _model, _device
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VideoAnalyser] Loading EfficientNet on {_device}...")
    base         = create_model('efficientnet_b2', pretrained=False, num_classes=0, global_pool='avg')
    feature_size = base.num_features
    _model       = _CartoonClassifier(base, feature_size).to(_device)
    _model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=_device, weights_only=True))
    _model.eval()
    print("[VideoAnalyser] Model ready.")


def _download_video(video_id: str) -> str:
    """Download the video to TEMP_VIDEO_DIR and return the file path."""
    import yt_dlp

    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
    output_template = os.path.join(TEMP_VIDEO_DIR, f"{video_id}.%(ext)s")

    ydl_opts = {
        'format'     : 'bestvideo[ext=mp4][height<=480]/bestvideo[ext=mp4]/best[ext=mp4]/best',
        'outtmpl'    : output_template,
        'quiet'      : True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    # Find the downloaded file (extension may vary)
    for ext in ['mp4', 'webm', 'mkv']:
        path = os.path.join(TEMP_VIDEO_DIR, f"{video_id}.{ext}")
        if os.path.exists(path):
            print(f"[VideoAnalyser] Video downloaded: {path}")
            return path

    raise FileNotFoundError(f"Video download failed — no file found in {TEMP_VIDEO_DIR}")


def _extract_frames(video_path: str) -> list:
    """Extract 1 frame per second from the video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(1, int(fps / VIDEO_FRAME_RATE))

    frames      = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            timestamp = frame_count / fps
            frames.append((frame_count, timestamp, frame))
        frame_count += 1

    cap.release()
    print(f"[VideoAnalyser] Extracted {len(frames)} frames.")
    return frames


def _predict_frames(frames: list) -> list:
    results = []
    for frame_idx, timestamp, frame in frames:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(rgb)
        tensor = _transform(img).unsqueeze(0).to(_device)

        with torch.no_grad():
            prob = _model(tensor).item()

        label      = "UNSAFE" if prob > VIDEO_THRESHOLD else "SAFE"
        confidence = prob if prob > VIDEO_THRESHOLD else 1 - prob

        results.append({
            "frame_idx"    : frame_idx,
            "timestamp"    : round(timestamp, 2),
            "timestamp_str": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
            "label"        : label,
            "confidence"   : round(confidence * 100, 2),
            "prob"         : round(prob, 4),
        })

    return results


def analyse(video_id=None, is_local=False, local_path=None):
    """
    Run EfficientNet visual analysis on a video.

    For YouTube videos : pass video_id — video is downloaded then deleted
    For local videos   : pass is_local=True and local_path
    """
    video_path   = None
    temp_created = False

    try:
        _load_model()

        if is_local:
            if not local_path or not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            video_path = local_path
            print(f"[VideoAnalyser] Using local file: {local_path}")
        else:
            if not video_id:
                raise ValueError("video_id is required for YouTube video analysis.")
            print(f"[VideoAnalyser] Downloading video: {video_id}")
            video_path   = _download_video(video_id)
            temp_created = True

        # Extract frames
        print("[VideoAnalyser] Extracting frames...")
        frames = _extract_frames(video_path)

        if not frames:
            raise RuntimeError("No frames could be extracted from the video.")

        # Predict
        print(f"[VideoAnalyser] Running predictions on {len(frames)} frames...")
        results = _predict_frames(frames)

        # Aggregate
        total        = len(results)
        unsafe_list  = [r for r in results if r["label"] == "UNSAFE"]
        unsafe_ratio = len(unsafe_list) / total if total > 0 else 0.0
        verdict      = "UNSAFE" if unsafe_ratio >= VIDEO_UNSAFE_RATIO else "SAFE"

        unsafe_timestamps = [
            {"timestamp_str": r["timestamp_str"], "confidence": r["confidence"]}
            for r in unsafe_list[:20]
        ]

        print(f"[VideoAnalyser] Done. Verdict: {verdict} "
              f"({len(unsafe_list)}/{total} unsafe frames, ratio={unsafe_ratio:.3f})")

        return {
            "verdict"           : verdict,
            "score"             : round(unsafe_ratio, 4),
            "unsafe_frames"     : len(unsafe_list),
            "total_frames"      : total,
            "unsafe_ratio"      : round(unsafe_ratio, 4),
            "unsafe_timestamps" : unsafe_timestamps,
            "error"             : None,
        }

    except Exception as e:
        print(f"[VideoAnalyser] ERROR: {e}")
        return {
            "verdict"           : "ERROR",
            "score"             : None,
            "unsafe_frames"     : 0,
            "total_frames"      : 0,
            "unsafe_ratio"      : 0.0,
            "unsafe_timestamps" : [],
            "error"             : str(e),
        }

    finally:
        # Delete downloaded temp file — never delete user's local file
        if temp_created and video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"[VideoAnalyser] Temp file deleted: {video_path}")