import re
import os
import sys

# TextDetection.py lives in D:\PythonYouTube\Textual\
_TEXTUAL_DIR = r"D:\PythonYouTube\Textual"
if _TEXTUAL_DIR not in sys.path:
    sys.path.insert(0, _TEXTUAL_DIR)

from config import (
    TEXT_MODEL_PATH,
    YOUTUBE_API_KEY,
    TEXT_CONFIDENCE_THRESHOLD,
    TEXT_TRANSCRIPT_THRESHOLD,
    TEXT_COMMENT_HARM_THRESHOLD,
    TEXT_MAX_LEN,
    TEXT_MAX_COMMENTS,
)

# lazy-loaded globals (loaded once on first call)
_tokenizer = None
_model     = None
_device    = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = RobertaTokenizer.from_pretrained(TEXT_MODEL_PATH)
    _model     = RobertaForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    _model     = _model.to(_device)
    _model.eval()
    print(f"[TextAnalyser] RoBERTa loaded on {_device}")


def _clean_text(text):
    import html as html_lib
    import contractions
    if not text:
        return ""
    try:
        text = contractions.fix(str(text))
    except Exception:
        text = str(text)
    text = html_lib.unescape(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_transcript(text):
    if not text:
        return ""
    text = re.sub(r'>+',      '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+',    ' ', text).strip()
    return text


def _detect_and_translate(text):
    from langdetect import detect, LangDetectException
    from deep_translator import GoogleTranslator
    if not text or len(text.strip()) < 3:
        return text, "en", "English", False
    try:
        lang_code = detect(text)
    except LangDetectException:
        lang_code = "en"
    if lang_code == "en":
        return text, "en", "English", False
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        if not translated:
            translated = text
    except Exception:
        translated = text
        lang_code  = "en"
    return translated, lang_code, lang_code, True


def _classify_text(text):
    import torch
    import torch.nn.functional as F
    if not text or len(text.strip()) < 3:
        return 0, 1.0, 1.0, 0.0
    encoding = _tokenizer(
        text,
        max_length=TEXT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)
    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).squeeze()
    safe_prob  = probs[0].item()
    harm_prob  = probs[1].item()
    label      = 1 if harm_prob > safe_prob else 0
    confidence = max(safe_prob, harm_prob)
    return label, confidence, safe_prob, harm_prob


def _process_text(raw_text, source_label, is_transcript=False):
    if not raw_text or len(raw_text.strip()) < 3:
        return None
    translated, lang_code, lang_name, was_translated = _detect_and_translate(raw_text)
    cleaned = _clean_text(translated)
    if is_transcript:
        cleaned = _clean_transcript(cleaned)
    if not cleaned:
        return None
    label, confidence, safe_prob, harm_prob = _classify_text(cleaned)
    threshold = TEXT_TRANSCRIPT_THRESHOLD if is_transcript else TEXT_CONFIDENCE_THRESHOLD
    if confidence < threshold:
        label = 0
    return {
        "source"           : source_label,
        "original_text"    : raw_text[:300],
        "was_translated"   : was_translated,
        "original_language": lang_name if was_translated else "English",
        "translated_text"  : translated[:300] if was_translated else None,
        "verdict"          : "HARMFUL" if label == 1 else "SAFE",
        "confidence"       : round(confidence, 4),
        "safe_prob"        : round(safe_prob, 4),
        "harmful_prob"     : round(harm_prob, 4),
    }


def _fetch_youtube_data(video_id):
    from googleapiclient.discovery import build
    from youtube_transcript_api import YouTubeTranscriptApi

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # Metadata
    video_resp = youtube.videos().list(part="snippet", id=video_id).execute()
    if not video_resp.get("items"):
        raise ValueError(f"No YouTube video found for ID: {video_id}")
    snippet     = video_resp["items"][0]["snippet"]
    title       = snippet.get("title", "")
    description = snippet.get("description", "")

    # Comments — capped at TEXT_MAX_COMMENTS
    comments  = []
    next_page = None
    while True:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                textFormat="plainText"
            ).execute()
            for item in resp.get("items", []):
                comments.append(
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                )
                if len(comments) >= TEXT_MAX_COMMENTS:
                    break
            next_page = resp.get("nextPageToken")
            if not next_page or len(comments) >= TEXT_MAX_COMMENTS:
                break
        except Exception:
            break

    print(f"[TextAnalyser] Comments fetched: {len(comments)} (cap: {TEXT_MAX_COMMENTS})")

    # Transcript
    transcript_text = ""
    try:
        ytt_api         = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        transcript      = transcript_list.find_transcript(['en'])
        fetched         = transcript.fetch()
        transcript_text = " ".join([s.text for s in fetched])
    except Exception:
        try:
            ytt_api         = YouTubeTranscriptApi()
            fetched         = ytt_api.fetch(video_id)
            transcript_text = " ".join([s.text for s in fetched])
        except Exception:
            pass

    return {
        "title"      : title,
        "description": description,
        "comments"   : comments,
        "transcript" : transcript_text,
    }


def _aggregate(all_results):
    sources = {"title": [], "description": [], "transcript": [], "comments": []}
    for r in all_results:
        if r and r["source"] in sources:
            sources[r["source"]].append(r)

    core_unsafe = any(
        r["verdict"] == "HARMFUL"
        for src in ["title", "description", "transcript"]
        for r in sources[src]
    )

    total_comments   = len(sources["comments"])
    harmful_comments = sum(1 for r in sources["comments"] if r["verdict"] == "HARMFUL")
    comment_harm_pct = (harmful_comments / total_comments * 100) if total_comments > 0 else 0
    comments_unsafe  = comment_harm_pct > (TEXT_COMMENT_HARM_THRESHOLD * 100)

    verdict      = "UNSAFE" if (core_unsafe or comments_unsafe) else "SAFE"
    harmful_list = [r for r in all_results if r and r["verdict"] == "HARMFUL"]
    total_valid  = len([r for r in all_results if r])
    score        = (len(harmful_list) / total_valid) if total_valid > 0 else 0.0

    return {
        "verdict"          : verdict,
        "score"            : round(score, 4),
        "total_analysed"   : total_valid,
        "total_harmful"    : len(harmful_list),
        "comment_harm_pct" : round(comment_harm_pct, 2),
        "per_source": {
            src: {
                "analysed": len(sources[src]),
                "harmful" : sum(1 for r in sources[src] if r["verdict"] == "HARMFUL")
            }
            for src in sources
        },
        "harmful_texts": harmful_list,
        "error"        : None,
    }


def analyse(video_id=None, is_local=False, local_path=None):
    """
    Run text analysis on a video.
    For YouTube videos : pass video_id (11-char string)
    For local videos   : returns N/A — no metadata to analyse
    """
    try:
        _load_model()

        if is_local:
            return {
                "verdict"          : "N/A",
                "score"            : None,
                "total_analysed"   : 0,
                "total_harmful"    : 0,
                "comment_harm_pct" : 0,
                "per_source"       : {},
                "harmful_texts"    : [],
                "error"            : "Text analysis not available for local files.",
            }

        if not video_id:
            raise ValueError("video_id is required for YouTube text analysis.")

        print(f"[TextAnalyser] Fetching data for video: {video_id}")
        data = _fetch_youtube_data(video_id)

        all_results = []

        # Title
        if data["title"]:
            r = _process_text(data["title"], "title", is_transcript=False)
            if r:
                all_results.append(r)

        # Description
        if data["description"]:
            sentences = re.split(r'(?<=[.!?])\s+', data["description"])
            for sent in sentences:
                if len(sent.strip()) > 10:
                    r = _process_text(sent.strip(), "description", is_transcript=False)
                    if r:
                        all_results.append(r)

        # Transcript
        if data["transcript"]:
            words, chunk, char_count = data["transcript"].split(), [], 0
            for word in words:
                chunk.append(word)
                char_count += len(word)
                if char_count >= 200:
                    r = _process_text(" ".join(chunk), "transcript", is_transcript=True)
                    if r:
                        all_results.append(r)
                    chunk, char_count = [], 0
            if chunk:
                r = _process_text(" ".join(chunk), "transcript", is_transcript=True)
                if r:
                    all_results.append(r)

        # Comments
        for comment in data["comments"]:
            r = _process_text(comment, "comments", is_transcript=False)
            if r:
                all_results.append(r)

        print(f"[TextAnalyser] Done. {len(all_results)} texts analysed.")
        return _aggregate(all_results)

    except Exception as e:
        print(f"[TextAnalyser] ERROR: {e}")
        return {
            "verdict"          : "ERROR",
            "score"            : None,
            "total_analysed"   : 0,
            "total_harmful"    : 0,
            "comment_harm_pct" : 0,
            "per_source"       : {},
            "harmful_texts"    : [],
            "error"            : str(e),
        }