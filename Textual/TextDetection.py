import os
import re
import json
import html
import datetime
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import contractions

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH             = r"D:\PythonYouTube\Textual\best_roberta_model"
OUTPUT_DIR             = r"D:\PythonYouTube\Textual"
YOUTUBE_API_KEY        = "AIzaSyDKtW_i6k43D4rgNdAjoZTPH1kBUiyiesc"

MAX_LEN                     = 128    # must match training
CONFIDENCE_THRESHOLD        = 0.70   # minimum confidence for comments/title/description
TRANSCRIPT_CONFIDENCE_THRESHOLD = 0.95   # stricter threshold for transcripts (reduces false positives)
COMMENT_HARM_THRESHOLD      = 0.10   # flag video if >10% of comments are harmful

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LANGUAGE CODE → FULL NAME MAP
# ─────────────────────────────────────────────
LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "nl": "Dutch", "no": "Norwegian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "sq": "Albanian", "sr": "Serbian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

def get_language_name(code):
    return LANGUAGE_NAMES.get(code, f"Unknown ({code})")

# ─────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────
print("=" * 60)
print("INFERENCE PIPELINE — HARMFUL CONTENT DETECTOR")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ GPU detected : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU — running on CPU")
print(f"   Device       : {device}")

# ─────────────────────────────────────────────
# LOAD MODEL & TOKENIZER
# ─────────────────────────────────────────────
print(f"\n   Loading model from: {MODEL_PATH}")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(device)
model.eval()
print("✅ Model and tokenizer loaded\n")

# ─────────────────────────────────────────────
# HELPER: clean_text
# ─────────────────────────────────────────────
def clean_text(text):
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

# ─────────────────────────────────────────────
# HELPER: clean_transcript
# Removes transcript-specific noise before classification
# ─────────────────────────────────────────────
def clean_transcript(text):
    """
    Removes auto-generated transcript formatting noise only.
    Does NOT lowercase — RoBERTa was trained on mixed-case text.
      - Speaker markers like >> or >
      - Sound/action tags like [laughter], [music], [screaming]
      - Parenthetical notes like (applause)
      - Extra whitespace
    """
    if not text:
        return ""
    text = re.sub(r'>+',      '',  text)   # remove >> speaker markers
    text = re.sub(r'\[.*?\]', '',  text)   # remove [laughter], [music] etc
    text = re.sub(r'\(.*?\)', '',  text)   # remove (applause) etc
    text = re.sub(r'\s+',    ' ', text).strip()
    return text

# ─────────────────────────────────────────────
# HELPER: detect & translate
# ─────────────────────────────────────────────
def detect_and_translate(text):
    """
    Detects language of text.
    If not English, translates to English.
    Returns:
        translated_text  : English text (or original if already English)
        original_lang    : language code e.g. "ar"
        lang_name        : full language name e.g. "Arabic"
        was_translated   : True if translation occurred
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

    # Translate to English
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        if not translated:
            translated = text
    except Exception:
        translated = text
        lang_code  = "en"
        lang_name  = "English"

    return translated, lang_code, lang_name, True

# ─────────────────────────────────────────────
# HELPER: RoBERTa inference on single text
# ─────────────────────────────────────────────
def classify_text(text):
    """
    Runs a single cleaned English text through RoBERTa.
    Returns:
        label      : 0 (safe) or 1 (harmful)
        confidence : float probability of predicted class
        safe_prob  : probability of being safe
        harm_prob  : probability of being harmful
    """
    if not text or len(text.strip()) < 3:
        return 0, 1.0, 1.0, 0.0

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze()

    safe_prob  = probs[0].item()
    harm_prob  = probs[1].item()
    label      = 1 if harm_prob > safe_prob else 0
    confidence = max(safe_prob, harm_prob)

    return label, confidence, safe_prob, harm_prob

# ─────────────────────────────────────────────
# HELPER: Process a single piece of text
# (detect lang → translate → clean → classify)
# ─────────────────────────────────────────────
def process_text(raw_text, source_label="text", is_transcript=False):
    """
    Full pipeline for one piece of text.
    is_transcript=True applies extra transcript noise cleaning.
    Returns a result dict.
    """
    if not raw_text or len(raw_text.strip()) < 3:
        return None

    # Step 1 — language detection & translation
    translated, lang_code, lang_name, was_translated = detect_and_translate(raw_text)

    # Step 2 — clean
    cleaned = clean_text(translated)

    # Step 2b — extra transcript cleaning if applicable
    if is_transcript:
        cleaned = clean_transcript(cleaned)

    if not cleaned:
        return None

    # Step 3 — classify
    label, confidence, safe_prob, harm_prob = classify_text(cleaned)

    # Step 4 — apply confidence threshold
    # Transcripts use a stricter threshold to reduce false positives
    threshold = TRANSCRIPT_CONFIDENCE_THRESHOLD if is_transcript else CONFIDENCE_THRESHOLD
    if confidence < threshold:
        label = 0

    result = {
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

    return result

# ─────────────────────────────────────────────
# MODE 1: ONLINE YOUTUBE VIDEO
# ─────────────────────────────────────────────
def extract_youtube_data(url):
    """
    Extracts title, description, comments, and transcript
    from a YouTube video URL.
    """
    from googleapiclient.discovery import build
    from youtube_transcript_api import YouTubeTranscriptApi

    # Extract video ID from URL
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    video_id = match.group(1)
    print(f"   Video ID : {video_id}")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # ── Metadata (title + description) ──
    print("   Fetching metadata...")
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    snippet     = video_response["items"][0]["snippet"]
    title       = snippet.get("title", "")
    description = snippet.get("description", "")
    print(f"   Title       : {title[:60]}...")

    # ── Comments (all pages) ──
    print("   Fetching all comments...")
    comments     = []
    next_page    = None
    comment_count= 0

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment_text)
                comment_count += 1

            next_page = response.get("nextPageToken")
            if not next_page:
                break

        except Exception as e:
            print(f"   ⚠️  Could not fetch comments: {e}")
            break

    print(f"   Comments fetched: {comment_count:,}")

    # ── Transcript ──
    print("   Fetching transcript...")
    transcript_text = ""
    try:
        ytt_api         = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        transcript      = transcript_list.find_transcript(['en'])
        fetched         = transcript.fetch()
        transcript_text = " ".join([snippet.text for snippet in fetched])
        print(f"   Transcript length: {len(transcript_text):,} characters")
    except Exception:
        try:
            ytt_api         = YouTubeTranscriptApi()
            fetched         = ytt_api.fetch(video_id)
            transcript_text = " ".join([snippet.text for snippet in fetched])
            print(f"   Transcript length: {len(transcript_text):,} characters")
        except Exception as e:
            print(f"   ⚠️  No transcript available: {e}")

    return {
        "video_id"   : video_id,
        "title"      : title,
        "description": description,
        "comments"   : comments,
        "transcript" : transcript_text
    }

# ─────────────────────────────────────────────
# MODE 2: OFFLINE SUBTITLE FILE
# ─────────────────────────────────────────────
def extract_subtitle_data(subtitle_path):
    """
    Parses a .srt or .vtt subtitle file and extracts all text.
    """
    ext = os.path.splitext(subtitle_path)[1].lower()
    print(f"   Subtitle file : {subtitle_path}")
    print(f"   Format        : {ext}")

    text_lines = []

    if ext == ".srt":
        import pysrt
        subs = pysrt.open(subtitle_path, encoding="utf-8")
        text_lines = [sub.text for sub in subs]

    elif ext == ".vtt":
        import webvtt
        for caption in webvtt.read(subtitle_path):
            text_lines.append(caption.text)

    else:
        raise ValueError(f"Unsupported subtitle format: {ext}. Use .srt or .vtt")

    # Clean HTML tags from subtitle text (e.g. <i>, <b>)
    clean_lines = [re.sub(r'<[^>]+>', '', line).strip() for line in text_lines]
    clean_lines = [l for l in clean_lines if l]

    full_text = " ".join(clean_lines)
    print(f"   Subtitle lines  : {len(clean_lines):,}")
    print(f"   Total characters: {len(full_text):,}")

    return {
        "video_id"   : os.path.basename(subtitle_path),
        "title"      : "",
        "description": "",
        "comments"   : [],
        "transcript" : full_text
    }

# ─────────────────────────────────────────────
# AGGREGATION & FINAL VERDICT
# ─────────────────────────────────────────────
def aggregate_results(all_results, video_id):
    sources = {
        "title"      : [],
        "description": [],
        "transcript" : [],
        "comments"   : []
    }

    for r in all_results:
        if r and r["source"] in sources:
            sources[r["source"]].append(r)

    core_unsafe = False
    for src in ["title", "description", "transcript"]:
        for r in sources[src]:
            if r["verdict"] == "HARMFUL":
                core_unsafe = True

    total_comments   = len(sources["comments"])
    harmful_comments = sum(1 for r in sources["comments"] if r["verdict"] == "HARMFUL")
    comment_harm_pct = (harmful_comments / total_comments * 100) if total_comments > 0 else 0
    comments_unsafe  = comment_harm_pct > (COMMENT_HARM_THRESHOLD * 100)

    final_verdict = "UNSAFE" if (core_unsafe or comments_unsafe) else "SAFE"
    harmful_texts = [r for r in all_results if r and r["verdict"] == "HARMFUL"]

    summary = {
        "video_id"            : video_id,
        "final_verdict"       : final_verdict,
        "total_texts_analysed": len([r for r in all_results if r]),
        "total_harmful"       : len(harmful_texts),
        "comment_harm_pct"    : round(comment_harm_pct, 2),
        "translated_texts"    : sum(1 for r in all_results if r and r["was_translated"]),
        "per_source": {
            "title"      : {"analysed": len(sources["title"]),       "harmful": sum(1 for r in sources["title"]       if r["verdict"] == "HARMFUL")},
            "description": {"analysed": len(sources["description"]), "harmful": sum(1 for r in sources["description"] if r["verdict"] == "HARMFUL")},
            "transcript" : {"analysed": len(sources["transcript"]),  "harmful": sum(1 for r in sources["transcript"]  if r["verdict"] == "HARMFUL")},
            "comments"   : {"analysed": total_comments,              "harmful": harmful_comments}
        },
        "harmful_texts": harmful_texts,
        "all_results"  : [r for r in all_results if r]
    }

    return summary

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(summary, video_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id   = re.sub(r'[^\w\-]', '_', video_id)

    json_path = os.path.join(OUTPUT_DIR, f"results_{safe_id}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    txt_path = os.path.join(OUTPUT_DIR, f"results_{safe_id}_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HARMFUL CONTENT DETECTION REPORT\n")
        f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video ID  : {video_id}\n")
        f.write("=" * 60 + "\n\n")

        verdict_icon = "🔴 UNSAFE" if summary["final_verdict"] == "UNSAFE" else "🟢 SAFE"
        f.write(f"FINAL VERDICT : {verdict_icon}\n\n")
        f.write(f"Total texts analysed : {summary['total_texts_analysed']}\n")
        f.write(f"Total harmful found  : {summary['total_harmful']}\n")
        f.write(f"Texts translated     : {summary['translated_texts']}\n")
        f.write(f"Comment harm rate    : {summary['comment_harm_pct']:.1f}%\n\n")

        f.write("PER SOURCE BREAKDOWN:\n")
        f.write("-" * 40 + "\n")
        for src, data in summary["per_source"].items():
            if data["analysed"] > 0:
                f.write(f"  {src.capitalize():<15}: {data['harmful']} harmful / {data['analysed']} analysed\n")

        if summary["harmful_texts"]:
            f.write("\n" + "=" * 60 + "\n")
            f.write("HARMFUL TEXTS DETECTED:\n")
            f.write("=" * 60 + "\n\n")
            for i, r in enumerate(summary["harmful_texts"], 1):
                f.write(f"[{i}] Source     : {r['source'].upper()}\n")
                f.write(f"    Verdict    : {r['verdict']}  (confidence: {r['confidence']*100:.1f}%)\n")
                f.write(f"    Safe prob  : {r['safe_prob']*100:.1f}%\n")
                f.write(f"    Harm prob  : {r['harmful_prob']*100:.1f}%\n")
                if r["was_translated"]:
                    f.write(f"    Text       : {r['translated_text']}  (Original - {r['original_language']})\n")
                    f.write(f"    Original   : {r['original_text']}\n")
                else:
                    f.write(f"    Text       : {r['original_text']}\n")
                f.write("\n")
        else:
            f.write("\n✅ No harmful texts detected.\n")

    print(f"\n✅ JSON report saved  → {json_path}")
    print(f"✅ Text summary saved → {txt_path}")
    return json_path, txt_path

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(mode, url=None, subtitle_path=None):
    print("\n" + "=" * 60)
    print(f"MODE : {'Online YouTube Video' if mode == 'online' else 'Offline Subtitle File'}")
    print("=" * 60)

    if mode == "online":
        if not url:
            raise ValueError("URL is required for online mode")
        data = extract_youtube_data(url)
    else:
        if not subtitle_path:
            raise ValueError("Subtitle file path is required for offline mode")
        data = extract_subtitle_data(subtitle_path)

    video_id    = data["video_id"]
    all_results = []

    print("\n" + "─" * 60)
    print("ANALYSING TEXT SOURCES")
    print("─" * 60)

    # Title
    if data["title"]:
        print(f"\n[1/4] Analysing title...")
        result = process_text(data["title"], source_label="title", is_transcript=False)
        if result:
            all_results.append(result)
            icon      = "❌ HARMFUL" if result["verdict"] == "HARMFUL" else "✅ SAFE"
            lang_note = f" [Translated from {result['original_language']}]" if result["was_translated"] else ""
            print(f"      → {icon}  (confidence: {result['confidence']*100:.1f}%){lang_note}")

    # Description
    if data["description"]:
        print(f"\n[2/4] Analysing description...")
        sentences    = re.split(r'(?<=[.!?])\s+', data["description"])
        sentences    = [s.strip() for s in sentences if len(s.strip()) > 10]
        desc_harmful = 0
        for sent in sentences:
            result = process_text(sent, source_label="description", is_transcript=False)
            if result:
                all_results.append(result)
                if result["verdict"] == "HARMFUL":
                    desc_harmful += 1
        print(f"      → {desc_harmful} harmful sentences found in {len(sentences)} sentences")

    # Transcript — is_transcript=True applies noise cleaning
    if data["transcript"]:
        print(f"\n[3/4] Analysing transcript...")
        print(f"      (noise cleaning: speaker markers, sound tags, caps normalised)")
        words      = data["transcript"].split()
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

        trans_harmful = 0
        for ch in chunks:
            result = process_text(ch, source_label="transcript", is_transcript=True)
            if result:
                all_results.append(result)
                if result["verdict"] == "HARMFUL":
                    trans_harmful += 1
        print(f"      → {trans_harmful} harmful chunks found in {len(chunks)} transcript chunks")

    # Comments
    if data["comments"]:
        print(f"\n[4/4] Analysing {len(data['comments']):,} comments...")
        harmful_count    = 0
        translated_count = 0
        from tqdm import tqdm
        for comment in tqdm(data["comments"], desc="  Comments", unit="comment", ncols=80):
            result = process_text(comment, source_label="comments", is_transcript=False)
            if result:
                all_results.append(result)
                if result["verdict"] == "HARMFUL":
                    harmful_count += 1
                if result["was_translated"]:
                    translated_count += 1
        print(f"      → {harmful_count} harmful / {len(data['comments']):,} total comments")
        print(f"      → {translated_count} comments were translated")

    # Aggregate
    print("\n" + "─" * 60)
    print("AGGREGATING RESULTS")
    print("─" * 60)
    summary = aggregate_results(all_results, video_id)

    # Print verdict
    print("\n" + "=" * 60)
    verdict_icon = "🔴 UNSAFE" if summary["final_verdict"] == "UNSAFE" else "🟢 SAFE"
    print(f"  FINAL VERDICT : {verdict_icon}")
    print(f"  Total analysed: {summary['total_texts_analysed']:,}")
    print(f"  Total harmful : {summary['total_harmful']:,}")
    print(f"  Translated    : {summary['translated_texts']:,}")
    print(f"  Comment harm% : {summary['comment_harm_pct']:.1f}%")
    print("=" * 60)

    # Print all harmful texts
    if summary["harmful_texts"]:
        print("\n" + "=" * 60)
        print("HARMFUL TEXTS DETECTED:")
        print("=" * 60)
        for i, r in enumerate(summary["harmful_texts"], 1):
            print(f"\n[{i}] Source  : {r['source'].upper()}")
            print(f"    Verdict : {r['verdict']}  (confidence: {r['confidence']*100:.1f}%)")
            if r["was_translated"]:
                print(f"    Text    : {r['translated_text']}  (Original - {r['original_language']})")
                print(f"    Original: {r['original_text']}")
            else:
                print(f"    Text    : {r['original_text']}")
    else:
        print("\n✅ No harmful texts detected.")

    # Save
    save_results(summary, video_id)

    return summary

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SELECT MODE")
    print("=" * 60)
    print("  1 — Online YouTube Video (URL)")
    print("  2 — Offline Subtitle File (.srt or .vtt)")
    print()

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        url = input("Enter YouTube URL: ").strip()
        run_pipeline(mode="online", url=url)

    elif choice == "2":
        subtitle_path = input("Enter subtitle file path (.srt or .vtt): ").strip()
        run_pipeline(mode="offline", subtitle_path=subtitle_path)

    else:
        print("❌ Invalid choice. Please enter 1 or 2.")