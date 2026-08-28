import os
import sys
import numpy as np
import tensorflow as tf

from config import (
    AUDIO_MODEL_PATH,
    YAMNET_URL,
    AUDIO_SAMPLE_RATE,
    AUDIO_THRESHOLD,
    TEMP_AUDIO_DIR,
)

# lazy-loaded globals
_yamnet_model      = None
_classifier_model  = None

def _load_models():
    global _yamnet_model, _classifier_model
    if _yamnet_model is not None:
        return

    import tensorflow_hub as hub

    print("[AudioAnalyser] Loading YAMNet feature extractor...")
    _yamnet_model = hub.load(YAMNET_URL)
    print("[AudioAnalyser] Loading trained classifier...")
    _classifier_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    print("[AudioAnalyser] Models ready.")

def _load_audio(file_path: str) -> np.ndarray:
    import librosa
    # Load audio at the specific sample rate YAMNet expects
    waveform, _ = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32)

def _download_audio_from_youtube(video_id: str) -> str:
    import yt_dlp
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    # Use a specific filename to avoid confusion
    output_path = os.path.join(TEMP_AUDIO_DIR, f"{video_id}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(TEMP_AUDIO_DIR, f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    return output_path

def analyse(video_id=None, is_local=False, local_path=None):
    audio_path = None
    temp_created = False

    try:
        _load_models()

        if is_local:
            audio_path = local_path
        else:
            audio_path = _download_audio_from_youtube(video_id)
            temp_created = True

        waveform = _load_audio(audio_path)

        # 1. Extract raw embeddings from YAMNet
        # YAMNet returns embeddings for every ~0.96 seconds of audio
        _, embeddings, _ = _yamnet_model(waveform)

        # 2. Run classifier on EVERY embedding frame instead of averaging them
        predictions = _classifier_model.predict(embeddings, verbose=0)

        unsafe_timestamps = []
        total_frames = len(predictions)
        unsafe_count = 0

        for i, prob_vec in enumerate(predictions):
            prob_unsafe = float(prob_vec[0])

            if prob_unsafe >= AUDIO_THRESHOLD:
                unsafe_count += 1

                # Calculate time (YAMNet frames are spaced exactly 0.96s apart)
                seconds = i * 0.96
                mins = int(seconds // 60)
                secs = int(seconds % 60)

                unsafe_timestamps.append({
                    "timestamp_str": f"{mins:02d}:{secs:02d}",
                    "confidence": round(prob_unsafe * 100, 1)
                })

        # Calculate ratio-based score (percentage of audio that is unsafe)
        unsafe_ratio = unsafe_count / total_frames if total_frames > 0 else 0.0

        # Determine verdict
        # If more than 5% of the audio contains harmful sounds, mark as UNSAFE
        label = "UNSAFE" if unsafe_ratio > 0.05 else "SAFE"

        # For the global score, we return the ratio to feed the Fusion model
        # For confidence, we take the highest probability found
        max_prob = float(np.max(predictions)) if total_frames > 0 else 0.0

        return {
            "verdict": label,
            "score": round(unsafe_ratio, 4),
            "confidence": round(max_prob, 4),
            "unsafe_timestamps": unsafe_timestamps[:15], # Return first 15 detections
            "label": label,
            "error": None,
        }

    except Exception as e:
        print(f"[AudioAnalyser] ERROR: {e}")
        return {
            "verdict": "ERROR",
            "score": None,
            "confidence": None,
            "unsafe_timestamps": [],
            "error": str(e),
        }

    finally:
        if temp_created and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)