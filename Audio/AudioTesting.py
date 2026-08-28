"""
YAMNet Inference — Child Safety Audio Classifier
=================================================
Thesis: Detecting Harmful Content on YouTube Aimed at Children
Module: Unspoken Audio — Inference Script

This script:
    1. Asks the user: upload a local audio file OR enter a YouTube URL
    2. If YouTube URL: streams audio DIRECTLY into memory (no file saved)
    3. Loads and preprocesses the audio
    4. Extracts YAMNet embeddings (frozen)
    5. Runs the trained classifier
    6. Prints the result: SAFE or UNSAFE with confidence score

Install extras if needed:
    pip install yt-dlp
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# =============================================================================
# CONFIGURATION  — must match yamnet_classifier.py
# =============================================================================

class Config:
    SAMPLE_RATE   = 16000
    CLIP_DURATION = 10
    YAMNET_URL    = "https://tfhub.dev/google/yamnet/1"
    MODEL_PATH    = r"D:\PythonYouTube\Audio\models\yamnet_transfer.keras"
    THRESHOLD     = 0.4        # tuned threshold from training
    TEMP_DIR      = r"D:\PythonYouTube\Audio\temp"   # temp folder for downloads

cfg = Config()


# =============================================================================
# AUDIO LOADING — local file
# =============================================================================

def load_audio_file(file_path: str) -> np.ndarray:
    """
    Load a local audio file.
    Resamples to 16kHz mono, pads/trims to fixed length, normalises to [-1, 1].
    """
    print(f"  Loading file: {file_path}")
    waveform, _ = librosa.load(file_path, sr=cfg.SAMPLE_RATE, mono=True)
    return preprocess(waveform)


# =============================================================================
# AUDIO LOADING — YouTube download
# =============================================================================

def download_youtube_audio(url: str):
    """
    Download audio from a YouTube URL using yt-dlp.
    Saves as MP3 to the temp folder, loads it, then deletes the file.

    Returns:
        waveform : preprocessed float32 numpy array
        title    : video title string
    """
    try:
        import yt_dlp
    except ImportError:
        print("\n[ERROR] yt-dlp is not installed.")
        print("  Run:  pip install yt-dlp\n")
        sys.exit(1)

    os.makedirs(cfg.TEMP_DIR, exist_ok=True)
    output_template = os.path.join(cfg.TEMP_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format":         "bestaudio/best",
        "outtmpl":        output_template,
        "quiet":          False,
        "no_warnings":    False,
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "mp3",
            "preferredquality": "192",
        }],
    }

    print(f"  Downloading audio from YouTube...")
    print(f"  URL: {url}\n")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info     = ydl.extract_info(url, download=True)
            video_id = info.get("id",       "download")
            title    = info.get("title",    "Unknown")
            duration = info.get("duration", 0)

        mp3_path = os.path.join(cfg.TEMP_DIR, f"{video_id}.mp3")

        print(f"\n  Title     : {title}")
        print(f"  Duration  : {duration}s")
        print(f"  Saved to  : {mp3_path}\n")

        waveform = load_audio_file(mp3_path)

        # Delete the temp file after loading
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
            print(f"  Temp file deleted.\n")

        return waveform, title

    except Exception as e:
        print(f"\n[ERROR] Failed to download: {e}")
        print("  Make sure the URL is a valid public YouTube video.")
        print("  Also make sure ffmpeg is installed: https://ffmpeg.org/download.html\n")
        sys.exit(1)


# =============================================================================
# PREPROCESSING  (shared by both input methods)
# =============================================================================

def preprocess(waveform: np.ndarray) -> np.ndarray:
    """
    Pad or trim to fixed length, normalise amplitude.
    Must match exactly what was done during training.
    """
    waveform = waveform.astype(np.float32)

    target = cfg.SAMPLE_RATE * cfg.CLIP_DURATION
    if len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)))
    else:
        waveform = waveform[:target]

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    print(f"  Audio ready — {cfg.CLIP_DURATION}s @ {cfg.SAMPLE_RATE} Hz\n")
    return waveform


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embedding(yamnet_model, waveform: np.ndarray) -> np.ndarray:
    """
    Run waveform through frozen YAMNet.
    Returns mean-pooled embedding shape (1, 1024).
    """
    _, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)


# =============================================================================
# INFERENCE
# =============================================================================

def predict(model, embedding: np.ndarray) -> tuple:
    """
    Run the classification head.

    Returns:
        label       : "SAFE" or "UNSAFE"
        confidence  : confidence in the predicted label (0-1)
        prob_unsafe : raw P(unsafe) score
    """
    prob_unsafe = float(model.predict(embedding, verbose=0).flatten()[0])
    label       = "UNSAFE" if prob_unsafe >= cfg.THRESHOLD else "SAFE"
    confidence  = prob_unsafe if label == "UNSAFE" else (1.0 - prob_unsafe)
    return label, confidence, prob_unsafe


def print_result(label: str, confidence: float, prob_unsafe: float, source: str):
    """Print a clear formatted result."""
    bar_len = 40
    filled  = int(prob_unsafe * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "=" * 55)
    print("  CLASSIFICATION RESULT")
    print("=" * 55)
    print(f"  Source      : {source}")
    print(f"  Threshold   : {cfg.THRESHOLD}")
    print(f"  P(unsafe)   : {prob_unsafe:.4f}  [{bar}]")
    print(f"  P(safe)     : {1 - prob_unsafe:.4f}")
    print()

    if label == "UNSAFE":
        print(f"  ⚠  RESULT    : UNSAFE  (confidence: {confidence*100:.1f}%)")
        print(f"     This audio contains sounds that may be")
        print(f"     harmful or inappropriate for children.")
    else:
        print(f"  ✓  RESULT    : SAFE  (confidence: {confidence*100:.1f}%)")
        print(f"     This audio does not appear to contain")
        print(f"     harmful sounds for children.")

    print("=" * 55 + "\n")


# =============================================================================
# USER INPUT
# =============================================================================

def get_user_choice() -> str:
    """Ask user how they want to provide audio."""
    print("\n" + "=" * 55)
    print("  YAMNet — Child Safety Audio Classifier")
    print("=" * 55)
    print()
    print("  How would you like to provide the audio?")
    print()
    print("  [1]  Upload a local audio file")
    print("       (.wav  .mp3  .ogg  .flac  .m4a)")
    print()
    print("  [2]  Enter a YouTube URL")
    print("       (audio streamed directly — nothing saved)")
    print()

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return choice
        print("  Please enter 1 or 2.")


def get_file_path() -> str:
    """Prompt for a local file path and validate it."""
    print()
    while True:
        path = input("  Enter the full path to your audio file:\n  > ").strip().strip('"')
        if not path:
            print("  Path cannot be empty. Please try again.")
            continue
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            print("  Please check the path and try again.")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
            print(f"  Unsupported format: {ext}")
            print("  Supported: .wav  .mp3  .ogg  .flac  .m4a")
            continue
        return path


def get_youtube_url() -> str:
    """Prompt for a YouTube URL with basic validation."""
    print()
    while True:
        url = input("  Enter the YouTube URL:\n  > ").strip()
        if not url:
            print("  URL cannot be empty. Please try again.")
            continue
        if "youtube.com" not in url and "youtu.be" not in url:
            print("  That does not look like a YouTube URL.")
            print("  Example: https://www.youtube.com/watch?v=XXXXXXXXXXX")
            confirm = input("  Continue anyway? (y/n): ").strip().lower()
            if confirm == "y":
                return url
            continue
        return url


# =============================================================================
# MAIN
# =============================================================================

def main():

    # ── Step 1: Get user input ────────────────────────────────────────────────
    choice = get_user_choice()
    title  = None

    if choice == "1":
        audio_path = get_file_path()
        source     = os.path.basename(audio_path)
        print("\nPreprocessing audio...")
        waveform   = load_audio_file(audio_path)

    else:
        url = get_youtube_url()
        print("\nStreaming audio from YouTube...")
        waveform, title = download_youtube_audio(url)
        source = title or url

    # ── Step 2: Load YAMNet ───────────────────────────────────────────────────
    print("Loading YAMNet feature extractor...")
    yamnet = hub.load(cfg.YAMNET_URL)
    print("  YAMNet ready.\n")

    # ── Step 3: Extract embedding ─────────────────────────────────────────────
    print("Extracting audio features...")
    embedding = extract_embedding(yamnet, waveform)
    print("  Features extracted.\n")

    # ── Step 4: Load trained classifier ──────────────────────────────────────
    print("Loading trained classifier...")
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"\n[ERROR] Model not found: {cfg.MODEL_PATH}")
        print("  Run yamnet_classifier.py first to train the model.\n")
        sys.exit(1)
    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    print("  Classifier ready.\n")

    # ── Step 5: Predict and show result ──────────────────────────────────────
    print("Running classification...")
    label, confidence, prob_unsafe = predict(model, embedding)
    print_result(label, confidence, prob_unsafe, source)


if __name__ == "__main__":
    main()