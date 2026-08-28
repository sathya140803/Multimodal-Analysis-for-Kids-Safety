
import os
import re
import html
import json
import datetime
import warnings
import torch
import numpy as np
warnings.filterwarnings("ignore")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import contractions


# CONFIGURATION — must match RoBERTa inference script


class Config:
    # ── RoBERTa model
    MODEL_PATH   = r"D:\PythonYouTube\Textual\best_roberta_model"
    MAX_LEN      = 128      # must match training

    # ── Classification thresholds
    CONFIDENCE_THRESHOLD = 0.70   # minimum confidence to accept a prediction
    CHUNK_HARM_THRESHOLD = 0.20   # flag as HARMFUL if >20% of chunks are harmful

    # ── Whisper
    WHISPER_MODEL = "base"        # tiny | base | small | medium | large
                                  # base = good balance of speed and accuracy
                                  # use "small" or "medium" for better accuracy

    # ── Paths
    TEMP_DIR     = r"D:\PythonYouTube\Audio\temp"
    OUTPUT_DIR   = r"D:\PythonYouTube\Audio\results"

cfg = Config()

os.makedirs(cfg.TEMP_DIR,   exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 60)
print("  Spoken Audio Classifier — Child Safety")
print("=" * 60)
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
else:
    print("  GPU : Not detected — running on CPU")
print(f"  Device : {device}\n")

# LOAD ROBERTA MODEL
print("Loading RoBERTa classifier...")
if not os.path.exists(cfg.MODEL_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] RoBERTa model not found at: {cfg.MODEL_PATH}\n"
        f"Please train the model first using the training script.\n"
    )

tokenizer  = RobertaTokenizer.from_pretrained(cfg.MODEL_PATH)
rob_model  = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_PATH)
rob_model  = rob_model.to(device)
rob_model.eval()
print(f"  RoBERTa loaded from: {cfg.MODEL_PATH}\n")


# LOAD WHISPER MODEL

print(f"Loading Whisper ({cfg.WHISPER_MODEL}) for speech-to-text...")
try:
    import whisper
    whisper_model = whisper.load_model(cfg.WHISPER_MODEL)
    print(f"  Whisper '{cfg.WHISPER_MODEL}' loaded.\n")
except ImportError:
    print("\n[ERROR] Whisper is not installed.")
    print("  Run: pip install openai-whisper\n")
    raise


# LANGUAGE NAMES MAP
LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan",   "cs": "Czech",  "cy": "Welsh",     "da": "Danish",
    "de": "German",    "el": "Greek",  "en": "English",   "es": "Spanish",
    "et": "Estonian",  "fa": "Persian","fi": "Finnish",   "fr": "French",
    "gu": "Gujarati",  "he": "Hebrew", "hi": "Hindi",     "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian","it":"Italian", "ja": "Japanese",
    "ko": "Korean",    "ml": "Malayalam","mr":"Marathi",  "ms": "Malay",
    "nl": "Dutch",     "no": "Norwegian","pl":"Polish",   "pt": "Portuguese",
    "ro": "Romanian",  "ru": "Russian","sk":"Slovak",     "sl": "Slovenian",
    "sq": "Albanian",  "sr": "Serbian","sv":"Swedish",    "sw": "Swahili",
    "ta": "Tamil",     "te": "Telugu", "th": "Thai",      "tl": "Filipino",
    "tr": "Turkish",   "uk": "Ukrainian","ur":"Urdu",     "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",   "zh-tw": "Chinese (Traditional)",
}

def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, f"Unknown ({code})")


# TEXT HELPERS  (same logic as RoBERTa inference script)
def clean_text(text: str) -> str:
    """Clean text before classification — mirrors your inference script."""
    if not text:
        return ""
    try:
        text = contractions.fix(str(text))
    except IndexError:
        text = str(text)
    text = html.unescape(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_and_translate(text: str):
    """
    Detect language. If not English, translate to English.
    Returns (translated_text, lang_code, lang_name, was_translated)
    """
    if not text or len(text.strip()) < 3:
        return text, "en", "English", False

    try:
        lang_code = detect(text)
    except LangDetectException:
        lang_code = "en"

    lang_name = get_language_name(lang_code)

    if lang_code == "en":
        return text, "en", "English", False

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        if not translated:
            translated = text
    except Exception:
        translated = text
        lang_code  = "en"
        lang_name  = "English"

    return translated, lang_code, lang_name, True


def classify_text(text: str) -> tuple:
    """
    Run a single cleaned English text through RoBERTa.
    Returns (label, confidence, safe_prob, harm_prob)
    Mirrors classify_text() from your inference script exactly.
    """
    if not text or len(text.strip()) < 3:
        return 0, 1.0, 1.0, 0.0

    encoding = tokenizer(
        text,
        max_length=cfg.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model_forward(input_ids, attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze()

    safe_prob  = probs[0].item()
    harm_prob  = probs[1].item()
    label      = 1 if harm_prob > safe_prob else 0
    confidence = max(safe_prob, harm_prob)

    return label, confidence, safe_prob, harm_prob


def model_forward(input_ids, attention_mask):
    """Wrapper so rob_model is used clearly."""
    return rob_model(input_ids=input_ids, attention_mask=attention_mask)


# TRANSCRIPTION

def transcribe(audio_path: str) -> dict:
    """
    Transcribe audio file using Whisper.

    Whisper automatically detects the language.
    Returns a dict with:
        text          : full transcription string
        language      : detected language code (e.g. "en", "fr")
        language_name : full language name
        segments      : list of timed segments
    """
    print(f"\n  Transcribing: {os.path.basename(audio_path)}")
    print(f"  Whisper model: {cfg.WHISPER_MODEL}")
    print("  Please wait...\n")

    result = whisper_model.transcribe(
        audio_path,
        fp16=torch.cuda.is_available()   # use FP16 on GPU for speed
    )

    lang_code = result.get("language", "en")
    lang_name = get_language_name(lang_code)
    text      = result.get("text", "").strip()

    print(f"  Detected language : {lang_name} ({lang_code})")
    print(f"  Transcription     : {text[:200]}{'...' if len(text) > 200 else ''}\n")

    return {
        "text"         : text,
        "language"     : lang_code,
        "language_name": lang_name,
        "segments"     : result.get("segments", [])
    }


# AUDIO INPUT — LOCAL FILE


def get_audio_from_file(file_path: str) -> str:
    """Validate and return the local audio file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mp4", ".webm"}:
        raise ValueError(f"Unsupported format: {ext}")
    return file_path


# AUDIO INPUT — YOUTUBE URL

def get_audio_from_youtube(url: str) -> tuple:
    """
    Download audio from YouTube URL using yt-dlp.
    Returns (audio_path, video_title)
    """
    try:
        import yt_dlp
    except ImportError:
        print("\n[ERROR] yt-dlp is not installed.")
        print("  Run: pip install yt-dlp\n")
        raise

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

    print(f"  Downloading audio from: {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        video_id = info.get("id",    "download")
        title    = info.get("title", "Unknown")

    audio_path = os.path.join(cfg.TEMP_DIR, f"{video_id}.mp3")
    print(f"  Title  : {title}")
    print(f"  Saved  : {audio_path}\n")

    return audio_path, title


# CLASSIFICATION PIPELINE

def classify_transcription(transcription: dict, source_name: str) -> dict:
    """
    Takes the full Whisper transcription, splits into chunks,
    translates if needed, then classifies each chunk with RoBERTa.

    Chunking is necessary because RoBERTa has a max token limit (128).
    We split the transcription into ~200-character chunks — same approach
    as the transcript analysis in your RoBERTa inference script.

    Returns a full result dict.
    """
    full_text = transcription["text"]
    lang_code = transcription["language"]
    lang_name = transcription["language_name"]

    if not full_text.strip():
        print("  [WARNING] Transcription is empty — nothing to classify.")
        return None

    # ── Translation if not English
    was_translated = False
    translated_text = full_text

    if lang_code != "en":
        print(f"  Detected language: {lang_name} — translating to English...")
        translated_text, _, _, was_translated = detect_and_translate(full_text)
        print(f"  Translation complete.\n")

        print("  Translated text (English):")
        print(f"  {translated_text[:300]}{'...' if len(translated_text) > 300 else ''}\n")

    # ── Split into chunks
    words      = translated_text.split()
    chunks     = []
    chunk      = []
    char_count = 0

    for word in words:
        chunk.append(word)
        char_count += len(word)
        if char_count >= 200:
            chunks.append(" ".join(chunk))
            chunk      = []
            char_count = 0
    if chunk:
        chunks.append(" ".join(chunk))

    print(f"  Classifying {len(chunks)} text chunks with RoBERTa...")

    # ── Classify each chunk
    chunk_results  = []
    harmful_chunks = 0

    for i, chunk_text in enumerate(chunks):
        cleaned = clean_text(chunk_text)
        if not cleaned:
            continue

        label, confidence, safe_prob, harm_prob = classify_text(cleaned)

        # Apply confidence threshold
        if confidence < cfg.CONFIDENCE_THRESHOLD:
            label = 0

        verdict = "HARMFUL" if label == 1 else "SAFE"
        if verdict == "HARMFUL":
            harmful_chunks += 1

        chunk_results.append({
            "chunk_index"  : i + 1,
            "text"         : chunk_text[:200],
            "verdict"      : verdict,
            "confidence"   : round(confidence, 4),
            "safe_prob"    : round(safe_prob,  4),
            "harmful_prob" : round(harm_prob,  4),
        })

    # ── Overall verdict
    total_chunks   = len(chunk_results)
    harm_pct       = (harmful_chunks / total_chunks * 100) if total_chunks > 0 else 0
    overall_verdict = (
        "HARMFUL"
        if harm_pct > (cfg.CHUNK_HARM_THRESHOLD * 100)
        else "SAFE"
    )

    # Aggregate confidence
    harm_probs = [r["harmful_prob"] for r in chunk_results]
    avg_harm_prob = float(np.mean(harm_probs)) if harm_probs else 0.0
    max_harm_prob = float(np.max(harm_probs))  if harm_probs else 0.0

    harmful_texts = [r for r in chunk_results if r["verdict"] == "HARMFUL"]

    return {
        "source"            : source_name,
        "original_language" : lang_name,
        "was_translated"    : was_translated,
        "original_text"     : full_text[:500],
        "translated_text"   : translated_text[:500] if was_translated else None,
        "total_chunks"      : total_chunks,
        "harmful_chunks"    : harmful_chunks,
        "harm_percentage"   : round(harm_pct, 2),
        "avg_harmful_prob"  : round(avg_harm_prob, 4),
        "max_harmful_prob"  : round(max_harm_prob, 4),
        "overall_verdict"   : overall_verdict,
        "chunk_results"     : chunk_results,
        "harmful_texts"     : harmful_texts,
    }


# PRINT RESULT

def print_result(result: dict):
    """Print a clear formatted result to console."""
    if not result:
        print("\n[No result to display]")
        return

    verdict = result["overall_verdict"]
    icon    = "HARMFUL" if verdict == "HARMFUL" else "SAFE"

    print("\n" + "=" * 60)
    print("  SPOKEN AUDIO CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"  Source            : {result['source']}")
    print(f"  Language detected : {result['original_language']}")
    print(f"  Was translated    : {'Yes → English' if result['was_translated'] else 'No (already English)'}")
    print(f"  Total chunks      : {result['total_chunks']}")
    print(f"  Harmful chunks    : {result['harmful_chunks']} / {result['total_chunks']} ({result['harm_percentage']:.1f}%)")
    print(f"  Avg harmful prob  : {result['avg_harmful_prob']*100:.1f}%")
    print(f"  Max harmful prob  : {result['max_harmful_prob']*100:.1f}%")
    print()

    if verdict == "HARMFUL":
        print(f"  ⚠  VERDICT : {icon}  — spoken content contains potentially")
        print(f"     harmful language inappropriate for children.")
    else:
        print(f"  ✓  VERDICT : {icon}  — spoken content appears safe for children.")

    print("=" * 60)

    if result["harmful_texts"]:
        print("\n  HARMFUL CHUNKS DETECTED:")
        print("  " + "-" * 56)
        for r in result["harmful_texts"]:
            print(f"\n  Chunk {r['chunk_index']:>3} | Confidence: {r['confidence']*100:.1f}%"
                  f" | Harm prob: {r['harmful_prob']*100:.1f}%")
            print(f"  Text: {r['text']}")
    else:
        print("\n  ✓ No harmful chunks detected in the spoken audio.\n")


# SAVE RESULT

def save_result(result: dict):
    """Save result as JSON and plain text to OUTPUT_DIR."""
    if not result:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^\w\-]', '_', result["source"])[:40]

    # JSON
    json_path = os.path.join(cfg.OUTPUT_DIR, f"spoken_{safe_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    # Plain text
    txt_path = os.path.join(cfg.OUTPUT_DIR, f"spoken_{safe_name}_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SPOKEN AUDIO CLASSIFICATION REPORT\n")
        f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source    : {result['source']}\n")
        f.write("=" * 60 + "\n\n")

        verdict_icon = "HARMFUL" if result["overall_verdict"] == "HARMFUL" else "SAFE"
        f.write(f"VERDICT           : {verdict_icon}\n")
        f.write(f"Language          : {result['original_language']}\n")
        f.write(f"Translated        : {'Yes' if result['was_translated'] else 'No'}\n")
        f.write(f"Total chunks      : {result['total_chunks']}\n")
        f.write(f"Harmful chunks    : {result['harmful_chunks']} ({result['harm_percentage']:.1f}%)\n")
        f.write(f"Avg harmful prob  : {result['avg_harmful_prob']*100:.1f}%\n\n")

        f.write("FULL TRANSCRIPTION:\n")
        f.write("-" * 40 + "\n")
        f.write(result["original_text"] + "\n\n")

        if result["was_translated"]:
            f.write("TRANSLATED TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write((result["translated_text"] or "") + "\n\n")

        if result["harmful_texts"]:
            f.write("HARMFUL CHUNKS:\n")
            f.write("-" * 40 + "\n")
            for r in result["harmful_texts"]:
                f.write(f"  Chunk {r['chunk_index']} | Confidence: {r['confidence']*100:.1f}%\n")
                f.write(f"  {r['text']}\n\n")
        else:
            f.write("No harmful chunks detected.\n")

    print(f"\n  JSON saved → {json_path}")
    print(f"  TXT  saved → {txt_path}\n")


# USER INPUT

def get_user_choice() -> str:
    print("\n" + "=" * 60)
    print("  How would you like to provide the audio?")
    print("=" * 60)
    print()
    print("  [1]  Local audio file")
    print("       (.wav  .mp3  .ogg  .flac  .m4a)")
    print()
    print("  [2]  YouTube URL")
    print("       (audio downloaded, transcribed, then deleted)")
    print()
    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return choice
        print("  Please enter 1 or 2.")


def get_file_path() -> str:
    print()
    while True:
        path = input("  Enter the full path to your audio file:\n  > ").strip().strip('"')
        if not path:
            print("  Path cannot be empty.")
            continue
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue
        return path


def get_youtube_url() -> str:
    print()
    while True:
        url = input("  Enter the YouTube URL:\n  > ").strip()
        if not url:
            print("  URL cannot be empty.")
            continue
        if "youtube.com" not in url and "youtu.be" not in url:
            print("  That does not look like a YouTube URL.")
            confirm = input("  Continue anyway? (y/n): ").strip().lower()
            if confirm == "y":
                return url
            continue
        return url


# MAIN

def main():
    temp_file = None

    choice = get_user_choice()

    if choice == "1":
        file_path   = get_file_path()
        audio_path  = get_audio_from_file(file_path)
        source_name = os.path.basename(file_path)

    else:
        url = get_youtube_url()
        audio_path, title = get_audio_from_youtube(url)
        source_name = title
        temp_file   = audio_path   # mark for cleanup

    # ── Transcribe
    print("\nStep 1 — Transcribing spoken audio with Whisper...")
    transcription = transcribe(audio_path)

    # ── Classify
    print("Step 2 — Classifying transcription with RoBERTa...")
    result = classify_transcription(transcription, source_name)

    # ── Print & save
    print_result(result)
    save_result(result)

    # ── Clean up YouTube temp file
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"  Temp file deleted: {temp_file}")  


if __name__ == "__main__":
    main()