"""
evaluate_multimodal_kfold.py
============================
K-Fold Cross-Validation for multimodal evaluation

Splits your dataset into K folds, trains/tests on each fold,
then averages metrics across all folds for reliable performance estimates.

HOW TO USE:
  1. Run from FrontEnd/ directory:
         python evaluate_multimodal_kfold.py
  2. Results printed to console and saved to evaluation_results_kfold.json

GROUND TRUTH LABELS:
  0 = SAFE
  1 = UNSAFE
"""

import sys
import os
import json
import time
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    try:
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            return parse_qs(parsed.query).get('v', [None])[0]
        elif "youtu.be" in parsed.netloc:
            return parsed.path.lstrip('/')
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────────
# COMBINED DATASET (Text + Local Videos)
# All your test videos in one list for cross-validation
# ─────────────────────────────────────────────────────────────────
ALL_VIDEOS = [
    # ── TEXT (YouTube URLs) ──
    {"type": "text", "url": "https://youtu.be/vM-kt5sGJjk?si=eOHfL-Vba4mF5ial", "label": 0,
     "title": "Peppa Pig Episode"},
    {"type": "text", "url": "https://youtu.be/mWXrM-OKBNQ?si=2VAzw1je8FLVRBMm", "label": 0,
     "title": "Masha and the Bear Episode"},
    {"type": "text", "url": "https://youtu.be/yQBtn_XaKL4?si=t1NPGJetSKSeOQrY", "label": 0,
     "title": "PJ Masks Episode"},
    {"type": "text", "url": "https://youtu.be/tR9Gj_WGyqU?si=LhCslHPkPEpsvmjl", "label": 0,
     "title": "Mickey Mouse Clubhouse"},
    {"type": "text", "url": "https://youtu.be/YpI0jgqNJGc?si=p4vnRBdmYfgjiUpF", "label": 0, "title": "Bluey Episode"},
    {"type": "text", "url": "https://youtu.be/-un6q8_74Rw?si=V48_uM9UuvR42uI6", "label": 1,
     "title": "BoJack Horseman Clip"},
    {"type": "text", "url": "https://youtu.be/nZXd0TMEJfo?si=-PIJElCI8_YOtATM", "label": 1, "title": "Family Guy Clip"},
    {"type": "text", "url": "https://youtu.be/R1OwwEmF4h0?si=YeYZKrIYjWXlRei2", "label": 1,
     "title": "Rick and Morty Clip"},
    {"type": "text", "url": "https://youtu.be/4AaYwJPGZPM?si=SM8d6weHh42bsF-b", "label": 1, "title": "South Park Clip"},
    {"type": "text", "url": "https://youtu.be/rGQkLXIey4Y?si=ngoQRtO-SbpRV_OA", "label": 1,
     "title": "Fake Peppa Pig Horror"},

    # ── LOCAL (Video Files) ──
    {"type": "local", "path": r"D:\CollectedVideos\safe\peppa_pig\peppapig2.mp4", "label": 0, "title": "Peppa Pig 1"},
    {"type": "local", "path": r"D:\CollectedVideos\safe\masha_and_bear\mb3.mp4", "label": 0,
     "title": "Masha and Bear 1"},
    {"type": "local", "path": r"D:\CollectedVideos\safe\pj_masks\pj6.mp4", "label": 0, "title": "PJ Masks 1"},
    {"type": "local", "path": r"D:\CollectedVideos\safe\mickeymouse\m1.mp4", "label": 0, "title": "Mickey Mouse 1"},
    {"type": "local", "path": r"D:\CollectedVideos\safe\randomSafe\Rs3.mp4", "label": 0, "title": "Minecraft 1"},
    {"type": "local",
     "path": r"D:\PythonYouTube\FrontEnd\downloads\The Funniest BoJack Horseman Moments (That Still Hurt a Little).mp4",
     "label": 1, "title": "BoJack Horseman 1"},
    {"type": "local",
     "path": r"D:\PythonYouTube\FrontEnd\downloads\Family Guy Season 15 EP 15 Full Episodes ｜ Family Guy 2024 Full No Zoom NoCuts 1080P.mp4",
     "label": 1, "title": "Family Guy 1"},
    {"type": "local",
     "path": r"D:\PythonYouTube\FrontEnd\downloads\The Vindicators (Compilation) ｜ Rick and Morty ｜ adult swim.mp4",
     "label": 1, "title": "Rick and Morty 1"},
    {"type": "local",
     "path": r"D:\PythonYouTube\FrontEnd\downloads\There was a Ghost! This is ECTOPLASM! - SOUTH PARK.mp4", "label": 1,
     "title": "South Park 1"},
    {"type": "local", "path": r"D:\PythonYouTube\FrontEnd\downloads\The Simpsons Funniest Moments Part #2.mp4",
     "label": 1, "title": "Simpsons 1"},
]


# ─────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def run_text_inference(video_url: str) -> str:
    """Text only — uses YouTube API."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return "ERROR"

        from services.text_analyser import analyse as text_analyse
        from services.score_fusion import fuse

        text_result = text_analyse(video_id=video_id)
        fusion = fuse(
            selected_modes=["Text"],
            text_result=text_result,
        )
        return fusion["verdict"]
    except Exception as e:
        print(f"      TEXT ERROR: {e}")
        return "ERROR"


def run_local_inference(local_path: str, selected_modes: list) -> str:
    """Video/Audio/Spoken — uses local file."""
    try:
        video_result = None
        audio_result = None
        spoken_result = None

        if "Video" in selected_modes:
            from services.video_analyser import analyse as video_analyse
            video_result = video_analyse(is_local=True, local_path=local_path)

        if "Audio" in selected_modes:
            from services.audio_analyser import analyse as audio_analyse
            audio_result = audio_analyse(is_local=True, local_path=local_path)
            from services.spoken_analyser import analyse as spoken_analyse
            spoken_result = spoken_analyse(is_local=True, local_path=local_path)

        from services.score_fusion import fuse
        fusion_modes = list(selected_modes)
        if "Audio" in fusion_modes:
            fusion_modes.append("Spoken")

        fusion = fuse(
            selected_modes=fusion_modes,
            video_result=video_result,
            audio_result=audio_result,
            spoken_result=spoken_result,
        )
        return fusion["verdict"]
    except Exception as e:
        print(f"      LOCAL ERROR: {e}")
        return "ERROR"


def run_inference(video: dict, selected_modes: list) -> str:
    """Run inference based on video type."""
    if video["type"] == "text":
        return run_text_inference(video["url"])
    elif video["type"] == "local":
        return run_local_inference(video["path"], selected_modes)
    return "ERROR"


# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────

def verdict_to_int(verdict: str) -> int:
    """Convert verdict string to integer label."""
    if verdict == "UNSAFE":
        return 1
    if verdict == "SAFE":
        return 0
    return -1  # error


def compute_metrics(y_true: list, y_pred_raw: list) -> dict:
    """Compute metrics, filtering out ERROR predictions."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred_raw) if p != -1]
    if not pairs:
        return {"error": "All predictions errored"}

    yt = np.array([p[0] for p in pairs])
    yp = np.array([p[1] for p in pairs])

    cm = confusion_matrix(yt, yp)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics = {
        "accuracy": round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp, zero_division=0), 4),
        "recall": round(recall_score(yt, yp, zero_division=0), 4),
        "f1_score": round(f1_score(yt, yp, zero_division=0), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": len(yt),
        "errors": len(y_pred_raw) - len(yt),
    }

    # Add ROC-AUC if we have both classes
    if len(np.unique(yt)) == 2:
        try:
            metrics["roc_auc"] = round(roc_auc_score(yt, yp), 4)
        except:
            metrics["roc_auc"] = None

    return metrics


def print_metrics(name: str, metrics: dict):
    """Pretty-print metrics."""
    print(f"    [{name}]")
    if "error" in metrics:
        print(f"      {metrics['error']}")
        return
    print(f"      Accuracy  : {metrics['accuracy']}")
    print(f"      Precision : {metrics['precision']}")
    print(f"      Recall    : {metrics['recall']}")
    print(f"      F1 Score  : {metrics['f1_score']}")
    if metrics.get("roc_auc"):
        print(f"      ROC-AUC   : {metrics['roc_auc']}")
    print(f"      TP:{metrics['tp']}  TN:{metrics['tn']}  FP:{metrics['fp']}  FN:{metrics['fn']}  "
          f"Total:{metrics['total']}  Errors:{metrics['errors']}")


def aggregate_metrics(all_fold_metrics: list) -> dict:
    """Average metrics across all folds."""
    if not all_fold_metrics:
        return {}

    metrics_to_avg = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    aggregated = {}

    for metric in metrics_to_avg:
        values = [m[metric] for m in all_fold_metrics if metric in m and m[metric] is not None]
        if values:
            aggregated[f"{metric}_mean"] = round(np.mean(values), 4)
            aggregated[f"{metric}_std"] = round(np.std(values), 4)

    # Sum confusion matrix components
    aggregated["tp_total"] = sum(m.get("tp", 0) for m in all_fold_metrics)
    aggregated["tn_total"] = sum(m.get("tn", 0) for m in all_fold_metrics)
    aggregated["fp_total"] = sum(m.get("fp", 0) for m in all_fold_metrics)
    aggregated["fn_total"] = sum(m.get("fn", 0) for m in all_fold_metrics)
    aggregated["total_samples"] = sum(m.get("total", 0) for m in all_fold_metrics)
    aggregated["total_errors"] = sum(m.get("errors", 0) for m in all_fold_metrics)

    return aggregated


def print_aggregated_metrics(name: str, aggregated: dict, n_splits: int):
    """Pretty-print aggregated metrics across folds."""
    print(f"\n  ╔═══════════════════════════════════════════════════════╗")
    print(f"  ║ {name:<53} ║")
    print(f"  ║ ({n_splits}-Fold Cross-Validation Results)              ║")
    print(f"  ╚═══════════════════════════════════════════════════════╝")
    print(f"    Accuracy   : {aggregated.get('accuracy_mean', 'N/A')} (±{aggregated.get('accuracy_std', 'N/A')})")
    print(f"    Precision  : {aggregated.get('precision_mean', 'N/A')} (±{aggregated.get('precision_std', 'N/A')})")
    print(f"    Recall     : {aggregated.get('recall_mean', 'N/A')} (±{aggregated.get('recall_std', 'N/A')})")
    print(f"    F1 Score   : {aggregated.get('f1_score_mean', 'N/A')} (±{aggregated.get('f1_score_std', 'N/A')})")
    if aggregated.get('roc_auc_mean'):
        print(f"    ROC-AUC    : {aggregated.get('roc_auc_mean', 'N/A')} (±{aggregated.get('roc_auc_std', 'N/A')})")
    print(
        f"    TP:{aggregated['tp_total']}  TN:{aggregated['tn_total']}  FP:{aggregated['fp_total']}  FN:{aggregated['fn_total']}  "
        f"Total:{aggregated['total_samples']}  Errors:{aggregated['total_errors']}")


# ─────────────────────────────────────────────────────────────────
# MAIN K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────

def main():
    n_splits = 5
    random_state = 42

    # Filter out videos that don't exist
    valid_videos = []
    for video in ALL_VIDEOS:
        if video["type"] == "text":
            valid_videos.append(video)
        elif video["type"] == "local" and os.path.exists(video["path"]):
            valid_videos.append(video)
        else:
            print(f"  SKIP: {video['title']} (file not found or invalid)")

    if not valid_videos:
        print("ERROR: No valid videos found!")
        return

    print(f"\n{'=' * 60}")
    print(f"  K-FOLD CROSS-VALIDATION SETUP")
    print(f"{'=' * 60}")
    print(f"  Total videos: {len(valid_videos)}")
    print(f"  Safe videos: {sum(1 for v in valid_videos if v['label'] == 0)}")
    print(f"  Unsafe videos: {sum(1 for v in valid_videos if v['label'] == 1)}")
    print(f"  K-Folds: {n_splits}")
    print(f"  Strategy: Stratified (maintains class balance in each fold)")
    print()

    # Prepare data
    X = np.arange(len(valid_videos))
    y = np.array([v["label"] for v in valid_videos])

    # Use Stratified K-Fold to maintain class balance
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_results = {}

    # ── SECTION A: Text only ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION A — Text Modality (K-Fold)")
    print("=" * 60)

    text_fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n  [Fold {fold_idx + 1}/{n_splits}]")
        test_videos = [valid_videos[i] for i in test_idx]

        y_true, y_pred = [], []
        for video in test_videos:
            if video["type"] != "text":
                continue
            print(f"    Testing: {video['title']}")
            verdict = run_text_inference(video["url"])
            y_true.append(video["label"])
            y_pred.append(verdict_to_int(verdict))
            time.sleep(1)

        if y_true:
            metrics = compute_metrics(y_true, y_pred)
            text_fold_metrics.append(metrics)
            print_metrics(f"Fold {fold_idx + 1}", metrics)

    if text_fold_metrics:
        text_aggregated = aggregate_metrics(text_fold_metrics)
        all_results["Text"] = text_aggregated
        print_aggregated_metrics("TEXT MODALITY", text_aggregated, n_splits)

    # ── SECTION B: Local videos (Video, Audio, Video+Audio) ─────
    print("\n" + "=" * 60)
    print("  SECTION B — Local Videos (K-Fold)")
    print("=" * 60)

    local_combos = [
        ["Video"],
        ["Audio"],
        ["Video", "Audio"],
    ]

    for combo in local_combos:
        combo_name = " + ".join(combo)
        print(f"\n  --- {combo_name} ---")

        combo_fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n    [Fold {fold_idx + 1}/{n_splits}]")
            test_videos = [valid_videos[i] for i in test_idx]

            y_true, y_pred = [], []
            for video in test_videos:
                if video["type"] != "local":
                    continue
                if not os.path.exists(video["path"]):
                    continue
                print(f"      Testing: {video['title']}")
                verdict = run_inference(video, combo)
                y_true.append(video["label"])
                y_pred.append(verdict_to_int(verdict))

            if y_true:
                metrics = compute_metrics(y_true, y_pred)
                combo_fold_metrics.append(metrics)
                print_metrics(f"Fold {fold_idx + 1}", metrics)

        if combo_fold_metrics:
            combo_aggregated = aggregate_metrics(combo_fold_metrics)
            all_results[combo_name] = combo_aggregated
            print_aggregated_metrics(f"{combo_name.upper()}", combo_aggregated, n_splits)

    # ── SUMMARY TABLE ───────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("  FINAL SUMMARY (5-FOLD CROSS-VALIDATION)")
    print(f"{'=' * 60}")
    print(f"  {'Modality':<25} {'Accuracy':<15} {'F1 Score':<15}")
    print(f"  {'-' * 55}")
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"  {name:<25} {'N/A':<15} {'N/A':<15}")
        else:
            acc = metrics.get('accuracy_mean', 'N/A')
            f1 = metrics.get('f1_score_mean', 'N/A')
            acc_std = metrics.get('accuracy_std', 0)
            f1_std = metrics.get('f1_score_std', 0)
            print(f"  {name:<25} {str(acc)}±{acc_std:<6} {str(f1)}±{f1_std:<6}")

    # ── SAVE ─────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results_kfold.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()