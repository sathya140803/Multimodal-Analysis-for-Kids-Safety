"""
services/spoken_analyser.py
============================
Wraps the Whisper + RoBERTa spoken audio analysis from SpokenAudio.py.
Exposes a single function: analyse(video_id, is_local, local_path)

Returns a standardised result dict:
{
    "verdict":          "SAFE" | "UNSAFE" | "ERROR",
    "score":            float  (proportion of harmful chunks, 0.0 - 1.0),
    "avg_harmful_prob": float,
    "max_harmful_prob": float,
    "total_chunks":     int,
    "harmful_chunks":   int,
    "harm_percentage":  float,
    "transcription":    str,
    "language":         str,
    "was_translated":   bool,
    "harmful_texts":    list,
    "error":            str | None
}
"""

import os
import sys
import numpy as np

from config import (
    TEXT_MODEL_PATH,
    WHISPER_MODEL_SIZE,
    TEMP_AUDIO_DIR,
    TEXT_MAX_LEN,
    TEXT_CONFIDENCE_THRESHOLD,
)

# Spoken audio uses a lower harm threshold than text
# because a single harmful spoken chunk in a children's video matters
SPOKEN_CHUNK_HARM_THRESHOLD = 0.20   # > 20% harmful chunks = UNSAFE

# lazy-loaded globals
_whisper_model    = None
_rob_tokenizer    = None
_rob_model        = None
_device           = None


def _load_models():
    global _whisper_model, _rob_tokenizer, _rob_model, _device

    if _whisper_model is not None:
        return

    import torch
    import whisper
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SpokenAnalyser] Loading Whisper ({WHISPER_MODEL_SIZE})...")
    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    print(f"[SpokenAnalyser] Loading RoBERTa on {_device}...")
    _rob_tokenizer = RobertaTokenizer.from_pretrained(TEXT_MODEL_PATH)
    _rob_model     = RobertaForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    _rob_model     = _rob_model.to(_device)
    _rob_model.eval()

    print("[SpokenAnalyser] Models ready.")


def _transcribe(audio_path: str) -> dict:
    import torch
    print(f"[SpokenAnalyser] Transcribing: {os.path.basename(audio_path)}")
    result    = _whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
    lang_code = result.get("language", "en")
    text      = result.get("text", "").strip()
    print(f"[SpokenAnalyser] Detected language: {lang_code} | Length: {len(text)} chars")
    return {"text": text, "language": lang_code}


def _translate_if_needed(text: str, lang_code: str):
    if lang_code == "en":
        return text, False
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return (translated or text), True
    except Exception:
        return text, False


def _classify_chunk(text: str):
    import torch
    import torch.nn.functional as F

    if not text or len(text.strip()) < 3:
        return 0, 1.0, 1.0, 0.0

    encoding = _rob_tokenizer(
        text,
        max_length=TEXT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        outputs = _rob_model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).squeeze()

    safe_prob  = probs[0].item()
    harm_prob  = probs[1].item()
    label      = 1 if harm_prob > safe_prob else 0
    confidence = max(safe_prob, harm_prob)
    return label, confidence, safe_prob, harm_prob


def _download_audio_from_youtube(video_id: str) -> str:
    import yt_dlp

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    output_template = os.path.join(TEMP_AUDIO_DIR, f"spoken_{video_id}.%(ext)s")

    ydl_opts = {
        "format":         "bestaudio/best",
        "outtmpl":        output_template,
        "quiet":          True,
        "no_warnings":    True,
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    audio_path = os.path.join(TEMP_AUDIO_DIR, f"spoken_{video_id}.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio download failed: {audio_path}")
    return audio_path


def analyse(video_id=None, is_local=False, local_path=None):
    """
    Transcribe audio with Whisper then classify with RoBERTa.

    For YouTube videos : pass video_id
    For local videos   : pass is_local=True and local_path
    """
    audio_path   = None
    temp_created = False

    try:
        _load_models()

        if is_local:
            if not local_path or not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            audio_path = local_path
        else:
            if not video_id:
                raise ValueError("video_id is required for YouTube spoken analysis.")
            print(f"[SpokenAnalyser] Downloading audio for video: {video_id}")
            audio_path   = _download_audio_from_youtube(video_id)
            temp_created = True

        # Transcribe
        transcription = _transcribe(audio_path)
        full_text     = transcription["text"]
        lang_code     = transcription["language"]

        if not full_text.strip():
            return {
                "verdict"         : "SAFE",
                "score"           : 0.0,
                "avg_harmful_prob": 0.0,
                "max_harmful_prob": 0.0,
                "total_chunks"    : 0,
                "harmful_chunks"  : 0,
                "harm_percentage" : 0.0,
                "transcription"   : "",
                "language"        : lang_code,
                "was_translated"  : False,
                "harmful_texts"   : [],
                "error"           : None,
            }

        # Translate if needed
        translated_text, was_translated = _translate_if_needed(full_text, lang_code)

        # Split into ~200-char chunks
        words, chunk, char_count, chunks = translated_text.split(), [], 0, []
        for word in words:
            chunk.append(word)
            char_count += len(word)
            if char_count >= 200:
                chunks.append(" ".join(chunk))
                chunk, char_count = [], 0
        if chunk:
            chunks.append(" ".join(chunk))

        print(f"[SpokenAnalyser] Classifying {len(chunks)} chunks with RoBERTa...")

        # Classify each chunk
        chunk_results  = []
        harmful_chunks = 0

        for i, chunk_text in enumerate(chunks):
            label, confidence, safe_prob, harm_prob = _classify_chunk(chunk_text)
            if confidence < TEXT_CONFIDENCE_THRESHOLD:
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

        total_chunks    = len(chunk_results)
        harm_pct        = (harmful_chunks / total_chunks * 100) if total_chunks > 0 else 0
        overall_verdict = "UNSAFE" if harm_pct > (SPOKEN_CHUNK_HARM_THRESHOLD * 100) else "SAFE"

        harm_probs      = [r["harmful_prob"] for r in chunk_results]
        avg_harm_prob   = float(np.mean(harm_probs)) if harm_probs else 0.0
        max_harm_prob   = float(np.max(harm_probs))  if harm_probs else 0.0
        harmful_texts   = [r for r in chunk_results if r["verdict"] == "HARMFUL"]
        score           = harmful_chunks / total_chunks if total_chunks > 0 else 0.0

        print(f"[SpokenAnalyser] Done. Verdict: {overall_verdict} ({harmful_chunks}/{total_chunks} harmful chunks)")

        return {
            "verdict"         : overall_verdict,
            "score"           : round(score, 4),
            "avg_harmful_prob": round(avg_harm_prob, 4),
            "max_harmful_prob": round(max_harm_prob, 4),
            "total_chunks"    : total_chunks,
            "harmful_chunks"  : harmful_chunks,
            "harm_percentage" : round(harm_pct, 2),
            "transcription"   : full_text[:1000],
            "language"        : lang_code,
            "was_translated"  : was_translated,
            "harmful_texts"   : harmful_texts,
            "error"           : None,
        }

    except Exception as e:
        print(f"[SpokenAnalyser] ERROR: {e}")
        return {
            "verdict"         : "ERROR",
            "score"           : None,
            "avg_harmful_prob": None,
            "max_harmful_prob": None,
            "total_chunks"    : 0,
            "harmful_chunks"  : 0,
            "harm_percentage" : 0.0,
            "transcription"   : "",
            "language"        : "unknown",
            "was_translated"  : False,
            "harmful_texts"   : [],
            "error"           : str(e),
        }

    finally:
        if temp_created and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[SpokenAnalyser] Temp file deleted: {audio_path}")