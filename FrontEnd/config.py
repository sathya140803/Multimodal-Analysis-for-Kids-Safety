import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# BASE DIRECTORIES
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────────────────────
VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "..", "Visual", "model_output", "best_model.pth")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "..", "Audio", "models", "yamnet_transfer.keras")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "..", "Textual", "best_roberta_model")

# ─────────────────────────────────────────────────────────────────
# YAMNET HUB URL  (downloaded automatically on first run)
# ─────────────────────────────────────────────────────────────────
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# ─────────────────────────────────────────────────────────────────
# TEMPORARY DIRECTORIES  (created automatically if missing)
# ─────────────────────────────────────────────────────────────────
TEMP_DIR         = os.path.join(BASE_DIR, "temp")
TEMP_AUDIO_DIR   = os.path.join(TEMP_DIR, "audio")
TEMP_VIDEO_DIR   = os.path.join(TEMP_DIR, "video")
UPLOAD_DIR       = os.path.join(BASE_DIR, "uploads")

for _dir in (TEMP_DIR, TEMP_AUDIO_DIR, TEMP_VIDEO_DIR, UPLOAD_DIR):
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# YOUTUBE API
# ─────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL   = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_SEARCH_LIMIT = 10   # number of results to return per search

# ─────────────────────────────────────────────────────────────────
# VIDEO ANALYSER  (EfficientNet)
# ─────────────────────────────────────────────────────────────────
VIDEO_IMG_SIZE           = 260
VIDEO_FRAME_RATE         = 1      # frames per second to sample
VIDEO_THRESHOLD          = 0.5    # per-frame unsafe probability cutoff
VIDEO_UNSAFE_RATIO       = 0.20   # >= 20% unsafe frames → video UNSAFE

# ─────────────────────────────────────────────────────────────────
# AUDIO ANALYSER  (YAMNet)
# ─────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE    = 16000
AUDIO_CLIP_DURATION  = 10     # seconds per audio clip
AUDIO_THRESHOLD      = 0.4    # unsafe probability cutoff

# ─────────────────────────────────────────────────────────────────
# TEXT ANALYSER  (RoBERTa)
# ─────────────────────────────────────────────────────────────────
TEXT_MAX_LEN                     = 128
TEXT_CONFIDENCE_THRESHOLD        = 0.55
TEXT_TRANSCRIPT_THRESHOLD        = 0.85   # stricter for transcripts
TEXT_COMMENT_HARM_THRESHOLD      = 0.15   # > 15% harmful comments → flag
TEXT_MAX_COMMENTS                = 250    # max comments to fetch per video

# ─────────────────────────────────────────────────────────────────
# WHISPER  (Spoken audio transcription)
# ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"   # options: tiny, base, small, medium, large

# ─────────────────────────────────────────────────────────────────
# SCORE FUSION  (Multimodal combined score)
# ─────────────────────────────────────────────────────────────────
# Weights must sum to 1.0
FUSION_WEIGHTS = {
    "video":   0.40,
    "audio":   0.10,
    "text":    0.35,
    "spoken":  0.15,
}

# A combined score >= this threshold → overall verdict UNSAFE
FUSION_UNSAFE_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────
DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'guardian.db')}"

# ─────────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────────
SECRET_KEY         = "change-this-in-production"
MAX_UPLOAD_MB      = 500
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}