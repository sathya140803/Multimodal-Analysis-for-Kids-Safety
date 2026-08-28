import os
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model
import matplotlib.pyplot as plt
import yt_dlp

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
MODEL_PATH = r"D:\PythonYouTube\Visual\model_output\best_model.pth"
IMG_SIZE = 260
THRESHOLD = 0.5
FRAME_RATE = 1  # 1 frame per second
UNSAFE_RATIO_THRESHOLD = 0.2  # 20% unsafe frames = video unsafe

# ─────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}\n")


# ─────────────────────────────────────────────────
# CUSTOM TRANSFORM
# ─────────────────────────────────────────────────
class ScaleTo255:
    def __call__(self, x):
        return x * 255.0


predict_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    ScaleTo255(),
])


# ─────────────────────────────────────────────────
# REBUILD MODEL — must match train.py exactly
# ─────────────────────────────────────────────────
class CartoonClassifier(nn.Module):
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


# ─────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────
def load_model():
    print("Loading model...")
    base = create_model(
        'efficientnet_b2',
        pretrained=False,
        num_classes=0,
        global_pool='avg'
    )
    feature_size = base.num_features
    model = CartoonClassifier(base, feature_size).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print("✅ Model loaded\n")
    return model


# ─────────────────────────────────────────────────
# GET YOUTUBE STREAM URL — no download, no temp file
# yt-dlp fetches a direct streamable URL that
# cv2.VideoCapture opens directly in memory
# ─────────────────────────────────────────────────
def get_youtube_stream_url(url):
    print(f"  🔗 Getting stream URL (no download)...")
    print(f"     URL: {url}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/bestvideo/best',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # download=False → only fetches metadata + stream URL, nothing saved
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            stream_url = info['url']
            print(f"  ✅ Stream ready : {title}")
            print(f"     Duration     : {duration // 60:.0f}m {duration % 60:.0f}s")
            print(f"     Nothing saved to disk\n")
            return stream_url, title
    except Exception as e:
        print(f"  ❌ Failed to get stream URL: {e}")
        return None, None


# ─────────────────────────────────────────────────
# VALIDATE LOCAL VIDEO PATH
# ─────────────────────────────────────────────────
def validate_video_path(path):
    """
    Validates a video file path with detailed error messages.
    Returns (is_valid, error_message)
    """
    # Remove invisible Unicode characters (LRM, RLM, etc.)
    import unicodedata
    path = ''.join(c for c in path if unicodedata.category(c) != 'Cf')

    # Remove surrounding quotes if present
    path = path.strip().strip('"').strip("'")

    # Expand user home directory (~)
    path = os.path.expanduser(path)

    # Convert to absolute path
    path = os.path.abspath(path)

    # Check if path exists
    if not os.path.exists(path):
        return False, f"Path does not exist:\n     {path}"

    # Check if it's a file (not a directory)
    if not os.path.isfile(path):
        return False, f"Path is a directory, not a file:\n     {path}"

    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    _, ext = os.path.splitext(path)
    if ext.lower() not in valid_extensions:
        return False, (f"Unsupported video format: {ext}\n"
                       f"     Supported: {', '.join(valid_extensions)}")

    # Check file size (warn if very small)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb < 1:
        return False, f"File is very small ({file_size_mb:.2f}MB) — may not be a valid video"

    return True, path


# ─────────────────────────────────────────────────
# GET VIDEO SOURCE FROM USER
# ─────────────────────────────────────────────────
def get_video_source():
    print("=" * 55)
    print("   VIDEO SAFETY ANALYSER")
    print("=" * 55)
    print("\nHow do you want to provide the video?")
    print("  1 → Local video file")
    print("  2 → YouTube URL  (streamed — nothing downloaded)")
    print()

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            break
        print("  ⚠️  Please enter 1 or 2")

    if choice == "1":
        while True:
            path = input("\nEnter full path to video file:\n> ").strip()

            is_valid, result = validate_video_path(path)

            if is_valid:
                print(f"  ✅ Valid video file: {os.path.basename(result)}\n")
                return result, os.path.basename(result)
            else:
                print(f"  ❌ {result}")
                print("     Please check the path and try again\n")

    elif choice == "2":
        while True:
            url = input("\nEnter YouTube URL:\n> ").strip()
            if "youtube.com" in url or "youtu.be" in url:
                break
            print("  ⚠️  That doesn't look like a YouTube URL. Try again.")

        stream_url, title = get_youtube_stream_url(url)
        if not stream_url:
            print("❌ Could not get stream URL. Exiting.")
            exit()

        return stream_url, title


# ─────────────────────────────────────────────────
# EXTRACT FRAMES FROM VIDEO INTO A LIST
# Works for both local files and stream URLs
# ─────────────────────────────────────────────────
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video")
        print(f"   Possible reasons:")
        print(f"   • File is corrupted or not a valid video")
        print(f"   • Required video codec is not installed")
        print(f"   • File permissions issue (try running as admin)")
        print(f"   • For YouTube: stream URL expired (try again)")
        print(f"\n   Path: {video_path}")
        return [], 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if total_frames > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(1, int(fps / frame_rate))

        print(f"  📹 Video info:")
        print(f"     Resolution: {width}x{height}")
        print(f"     FPS       : {fps:.1f}")
        if duration_sec > 0:
            print(f"     Duration  : {duration_sec:.0f}s ({duration_sec / 60:.1f} min)")
        print(f"  Extracting 1 frame per second...")

        frames = []
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
        print(f"  ✅ Extracted {len(frames)} frames\n")
        return frames, fps

    except Exception as e:
        cap.release()
        print(f"❌ Error processing video: {e}")
        return [], 0


# ─────────────────────────────────────────────────
# PREDICT EACH FRAME IN THE LIST
# ─────────────────────────────────────────────────
def predict_frames(model, frames):
    print(f"  Running predictions on {len(frames)} frames...")

    results = []

    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = predict_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(tensor).item()

        label = "UNSAFE" if prob > THRESHOLD else "SAFE"
        confidence = prob if prob > THRESHOLD else 1 - prob
        confidence = round(confidence * 100, 2)

        results.append({
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 2),
            "timestamp_str": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
            "label": label,
            "confidence": confidence,
            "prob": round(prob, 4),
            "frame": frame
        })

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(frames)} frames...")

    print(f"  ✅ Predictions complete\n")
    return results


# ─────────────────────────────────────────────────
# ANALYSE RESULTS
# ─────────────────────────────────────────────────
def analyse_results(results, video_title):
    total = len(results)
    unsafe_frames = [r for r in results if r["label"] == "UNSAFE"]
    safe_frames = [r for r in results if r["label"] == "SAFE"]
    unsafe_ratio = len(unsafe_frames) / total if total > 0 else 0
    verdict = "UNSAFE" if unsafe_ratio >= UNSAFE_RATIO_THRESHOLD else "SAFE"

    print("=" * 55)
    print("   VIDEO ANALYSIS RESULTS")
    print("=" * 55)
    print(f"  Video    : {video_title}")
    print(f"  Total    : {total} frames analysed")
    print(f"  Safe     : {len(safe_frames)} ({(1 - unsafe_ratio) * 100:.1f}%)")
    print(f"  Unsafe   : {len(unsafe_frames)} ({unsafe_ratio * 100:.1f}%)")
    print(f"  Threshold: {UNSAFE_RATIO_THRESHOLD * 100:.0f}% unsafe = video unsafe")
    print(f"\n  🎬 VIDEO VERDICT: {verdict}")
    print("=" * 55)

    if unsafe_frames:
        print(f"\n  ⚠️  Unsafe content detected at:")
        for r in unsafe_frames[:20]:
            print(f"     [{r['timestamp_str']}] "
                  f"Frame {r['frame_idx']} — "
                  f"{r['confidence']}% confidence")
        if len(unsafe_frames) > 20:
            print(f"     ... and {len(unsafe_frames) - 20} more")

    return verdict, unsafe_frames, safe_frames


# ─────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────
def display_results(results, verdict, unsafe_frames, video_title):
    timestamps = [r["timestamp"] for r in results]
    probs = [r["prob"] for r in results]
    colors = ["#e74c3c" if r["label"] == "UNSAFE"
              else "#2ecc71" for r in results]

    # ── Timeline graph ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(timestamps, probs, c=colors, s=20, zorder=3)
    ax.plot(timestamps, probs, color='gray', alpha=0.4, linewidth=1)
    ax.axhline(y=THRESHOLD, color='orange', linestyle='--',
               linewidth=1.5, label=f'Threshold ({THRESHOLD})')
    ax.fill_between(timestamps, THRESHOLD, 1,
                    alpha=0.1, color='red', label='Unsafe zone')
    ax.fill_between(timestamps, 0, THRESHOLD,
                    alpha=0.1, color='green', label='Safe zone')
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Unsafe Probability", fontsize=11)
    ax.set_title(
        f"{video_title}\nVerdict: {verdict} — "
        f"{len(unsafe_frames)}/{len(results)} unsafe frames",
        fontsize=12, fontweight='bold',
        color="#e74c3c" if verdict == "UNSAFE" else "#2ecc71"
    )
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Sample unsafe frames grid ──
    if unsafe_frames:
        sample = unsafe_frames[:8]
        cols = 4
        rows = math.ceil(len(sample) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten()

        for i, r in enumerate(sample):
            ax = axes[i]
            rgb = cv2.cvtColor(r["frame"], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(
                f"⚠️  UNSAFE\n[{r['timestamp_str']}] {r['confidence']}%",
                fontsize=9, fontweight='bold', color="#e74c3c"
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            f"Sample Unsafe Frames (showing {len(sample)} of {len(unsafe_frames)})",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()
    else:
        print("\n  ✅ No unsafe frames to display")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 0 — ask user for video source
    video_path, video_title = get_video_source()

    # Step 1 — load model
    model = load_model()

    # Step 2 — extract frames into list
    frames, fps = extract_frames(video_path, frame_rate=FRAME_RATE)
    if not frames:
        print("❌ No frames extracted — check video path or URL")
        exit()

    # Step 3 — predict each frame
    results = predict_frames(model, frames)

    # Step 4 — analyse results
    verdict, unsafe_frames, safe_frames = analyse_results(results, video_title)

    # Step 5 — display timeline + sample frames
    display_results(results, verdict, unsafe_frames, video_title)

    print("\n✅ Analysis complete!")