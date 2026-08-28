
from config import FUSION_WEIGHTS, FUSION_UNSAFE_THRESHOLD


def fuse(selected_modes: list, video_result=None, audio_result=None,
         text_result=None, spoken_result=None) -> dict:
    """
    Combine modality scores into a final verdict.

    Parameters
    ----------
    selected_modes : list of str
        The modalities the user selected, e.g. ['Video', 'Audio', 'Text']
        Valid values: 'Video', 'Audio', 'Text'
        (Spoken is included automatically when Audio is selected — or
         can be passed explicitly if you add it as a separate checkbox later)

    video_result   : dict from video_analyser.analyse()
    audio_result   : dict from audio_analyser.analyse()
    text_result    : dict from text_analyser.analyse()
    spoken_result  : dict from spoken_analyser.analyse()

    Returns
    -------
    dict with:
        verdict         : "SAFE" | "UNSAFE"
        combined_score  : float (0.0 - 1.0)
        modality_scores : dict  per-modality score and verdict
        weights_used    : dict  actual weights after redistribution
    """

    # Map mode names to their result dicts and config weight keys
    modality_map = {
        "Video" : ("video",  video_result),
        "Audio" : ("audio",  audio_result),
        "Text"  : ("text",   text_result),
        "Spoken": ("spoken", spoken_result),
    }

    # Build score table for selected modalities
    available = {}   # key → score (only modalities with a valid score)
    skipped   = {}   # key → reason (not selected or errored)

    for mode in ["Video", "Audio", "Text", "Spoken"]:
        key, result = modality_map[mode]

        if mode not in selected_modes:
            skipped[key] = "not_selected"
            continue

        if result is None:
            skipped[key] = "no_result"
            continue

        score = result.get("score")
        if score is None or result.get("verdict") in ("ERROR", "N/A"):
            skipped[key] = result.get("error") or result.get("verdict", "error")
            continue

        available[key] = float(score)

    if not available:
        # Nothing could be analysed
        return {
            "verdict"        : "ERROR",
            "combined_score" : None,
            "modality_scores": _build_modality_scores(modality_map, selected_modes),
            "weights_used"   : {},
            "error"          : "No modality produced a valid score.",
        }

    # Redistribute weights: take only the weights for available modalities,
    # then normalise so they sum to 1.0
    raw_weights = {key: FUSION_WEIGHTS.get(key, 0.0) for key in available}
    total_weight = sum(raw_weights.values())

    if total_weight == 0:
        # Fallback: equal weights
        normalised = {key: 1.0 / len(available) for key in available}
    else:
        normalised = {key: w / total_weight for key, w in raw_weights.items()}

    # Weighted sum
    combined_score = sum(available[key] * normalised[key] for key in available)
    combined_score = round(combined_score, 4)

    verdict = "UNSAFE" if combined_score >= FUSION_UNSAFE_THRESHOLD else "SAFE"

    return {
        "verdict"        : verdict,
        "combined_score" : combined_score,
        "modality_scores": _build_modality_scores(modality_map, selected_modes),
        "weights_used"   : {k: round(v, 4) for k, v in normalised.items()},
        "error"          : None,
    }


def _build_modality_scores(modality_map: dict, selected_modes: list) -> dict:
    """Build a clean per-modality summary for the report page."""
    out = {}
    for mode, (key, result) in modality_map.items():
        if mode not in selected_modes:
            out[key] = {"selected": False, "verdict": None, "score": None, "error": None}
            continue

        if result is None:
            out[key] = {"selected": True, "verdict": "ERROR", "score": None, "error": "No result returned"}
            continue

        out[key] = {
            "selected": True,
            "verdict" : result.get("verdict"),
            "score"   : result.get("score"),
            "error"   : result.get("error"),
        }

    return out