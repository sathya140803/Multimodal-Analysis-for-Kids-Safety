"""
database/db.py
==============
SQLAlchemy models and helper functions.

Tables:
    videos          — video metadata (one row per unique video)
    analyses        — one row per analysis run
    modality_scores — one row per modality per analysis
"""

import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# ─────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────

class Video(db.Model):
    __tablename__ = "videos"

    id          = db.Column(db.Integer,     primary_key=True)
    youtube_id  = db.Column(db.String(20),  unique=True, nullable=True)
    title       = db.Column(db.String(500), nullable=False)
    channel     = db.Column(db.String(200), nullable=True)
    thumbnail   = db.Column(db.String(500), nullable=True)
    duration    = db.Column(db.String(20),  nullable=True)
    views       = db.Column(db.String(50),  nullable=True)
    is_local    = db.Column(db.Boolean,     default=False)
    is_blocked  = db.Column(db.Boolean,     default=False)
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)

    analyses    = db.relationship("Analysis", backref="video", lazy=True)

    def to_dict(self):
        return {
            "id"        : self.id,
            "youtube_id": self.youtube_id,
            "title"     : self.title,
            "channel"   : self.channel,
            "thumbnail" : self.thumbnail,
            "duration"  : self.duration,
            "views"     : self.views,
            "is_local"  : self.is_local,
            "created_at": self.created_at.isoformat(),
        }


class Analysis(db.Model):
    __tablename__ = "analyses"

    id              = db.Column(db.Integer,     primary_key=True)
    video_id        = db.Column(db.Integer,     db.ForeignKey("videos.id"), nullable=False)
    selected_modes  = db.Column(db.String(100), nullable=False)
    overall_verdict = db.Column(db.String(10),  nullable=False)
    combined_score  = db.Column(db.Float,       nullable=True)
    created_at      = db.Column(db.DateTime,    default=datetime.utcnow)

    modality_scores = db.relationship("ModalityScore", backref="analysis", lazy=True)

    def to_dict(self):
        return {
            "id"             : self.id,
            "video_id"       : self.video_id,
            "selected_modes" : self.selected_modes,
            "overall_verdict": self.overall_verdict,
            "combined_score" : self.combined_score,
            "created_at"     : self.created_at.isoformat(),
            "modality_scores": [m.to_dict() for m in self.modality_scores],
        }


class ModalityScore(db.Model):
    __tablename__ = "modality_scores"

    id          = db.Column(db.Integer,    primary_key=True)
    analysis_id = db.Column(db.Integer,    db.ForeignKey("analyses.id"), nullable=False)
    modality    = db.Column(db.String(20), nullable=False)
    verdict     = db.Column(db.String(10), nullable=True)
    score       = db.Column(db.Float,      nullable=True)
    details     = db.Column(db.Text,       nullable=True)

    def to_dict(self):
        return {
            "id"         : self.id,
            "analysis_id": self.analysis_id,
            "modality"   : self.modality,
            "verdict"    : self.verdict,
            "score"      : self.score,
            "details"    : json.loads(self.details) if self.details else None,
        }


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def get_or_create_video(video_dict: dict) -> Video:
    """
    Look up a video by youtube_id. If it does not exist, create it.
    For local files (no youtube_id), always create a new row.
    """
    youtube_id = video_dict.get("id")
    is_local   = video_dict.get("is_local", False)

    if youtube_id and not is_local:
        existing = Video.query.filter_by(youtube_id=youtube_id).first()
        if existing:
            return existing

    video = Video(
        youtube_id = youtube_id if not is_local else None,
        title      = video_dict.get("title", "Unknown"),
        channel    = video_dict.get("channel", ""),
        thumbnail  = video_dict.get("thumbnail", ""),
        duration   = video_dict.get("duration", ""),
        views      = video_dict.get("views", ""),
        is_local   = is_local,
    )
    db.session.add(video)
    db.session.flush()
    return video


def block_video(youtube_id: str) -> bool:
    """
    Mark a video as blocked in the database.
    Returns True if successful, False if video not found.
    """
    if not youtube_id:
        return False
    video = Video.query.filter_by(youtube_id=youtube_id).first()
    if not video:
        return False
    video.is_blocked = True
    db.session.commit()
    print(f"[DB] Video {youtube_id} marked as blocked.")
    return True


def get_cached_analysis(youtube_id: str, selected_modes: list) -> dict | None:
    """
    Check if this video has already been analysed with the same modalities.

    Matching logic:
        - Same youtube_id
        - Stored selected_modes contains ALL the requested modes
          (so if user previously ran Video+Text+Audio and now runs
           just Text, we still have that data — but we only return
           a cache hit if ALL requested modes were previously run)

    Returns a dict with keys:
        found          : True
        verdict        : overall verdict string
        combined_score : float
        results        : dict with video/audio/text/spoken result dicts
        created_at     : when it was analysed
    Or None if no matching cache entry exists.
    """
    if not youtube_id:
        return None  # never cache local files

    video = Video.query.filter_by(youtube_id=youtube_id).first()
    if not video:
        return None

    # Sort modes for consistent comparison
    requested = sorted([m.lower() for m in selected_modes])

    # Look through all analyses for this video, newest first
    analyses = Analysis.query.filter_by(video_id=video.id)\
                             .order_by(Analysis.created_at.desc()).all()

    for analysis in analyses:
        stored_modes = sorted([m.strip().lower()
                                for m in analysis.selected_modes.split(",")])
        # Check all requested modes are covered by this stored analysis
        if all(m in stored_modes for m in requested):
            # Reconstruct results dict from stored modality_scores
            results = _reconstruct_results(analysis)
            print(f"[DB Cache] Hit for video {youtube_id} — "
                  f"analysed on {analysis.created_at.strftime('%Y-%m-%d %H:%M')}")
            return {
                "found"         : True,
                "verdict"       : "UNSAFE" if video.is_blocked else analysis.overall_verdict,
                "combined_score": analysis.combined_score,
                "results"       : results,
                "created_at"    : analysis.created_at.strftime("%Y-%m-%d %H:%M"),
                "is_blocked"    : video.is_blocked,
            }

    return None


def _reconstruct_results(analysis: Analysis) -> dict:
    """
    Rebuild the results dict from stored modality_scores so it can be
    passed directly to report.html — same structure as a fresh analysis.
    """
    results = {"video": None, "audio": None, "text": None, "spoken": None}

    for ms in analysis.modality_scores:
        details = json.loads(ms.details) if ms.details else {}
        # Merge top-level verdict/score back in (they're also in details
        # but having them at the top level matches the live result shape)
        details["verdict"] = ms.verdict
        details["score"]   = ms.score
        results[ms.modality] = details

    return results


def save_analysis(video_dict: dict, selected_modes: list,
                  results: dict, fusion: dict) -> Analysis:
    """
    Save a full analysis run to the database.

    Parameters
    ----------
    video_dict     : video dict from session
    selected_modes : list of selected modality names e.g. ['Video', 'Audio', 'Text']
    results        : dict with keys video/audio/text/spoken
    fusion         : dict returned by score_fusion.fuse()
    """
    video_record = get_or_create_video(video_dict)

    analysis = Analysis(
        video_id        = video_record.id,
        selected_modes  = ",".join(selected_modes),
        overall_verdict = fusion.get("verdict", "ERROR"),
        combined_score  = fusion.get("combined_score"),
    )
    db.session.add(analysis)
    db.session.flush()

    modality_map = {
        "video" : results.get("video"),
        "audio" : results.get("audio"),
        "text"  : results.get("text"),
        "spoken": results.get("spoken"),
    }

    for modality, result in modality_map.items():
        if result is None:
            continue

        safe_result = _sanitise_result(modality, result)

        ms = ModalityScore(
            analysis_id = analysis.id,
            modality    = modality,
            verdict     = result.get("verdict"),
            score       = result.get("score"),
            details     = json.dumps(safe_result),
        )
        db.session.add(ms)

    db.session.commit()
    return analysis


def _sanitise_result(modality: str, result: dict) -> dict:
    """Strip non-serialisable fields before saving to DB."""
    if not result:
        return {}

    base = {
        "verdict": result.get("verdict"),
        "score"  : result.get("score"),
        "error"  : result.get("error"),
    }

    if modality == "video":
        base.update({
            "unsafe_frames"    : result.get("unsafe_frames"),
            "total_frames"     : result.get("total_frames"),
            "unsafe_ratio"     : result.get("unsafe_ratio"),
            "unsafe_timestamps": result.get("unsafe_timestamps", [])[:20],
        })
    elif modality == "audio":
        base.update({
            "confidence": result.get("confidence"),
        })
    elif modality == "text":
        base.update({
            "total_analysed"  : result.get("total_analysed"),
            "total_harmful"   : result.get("total_harmful"),
            "comment_harm_pct": result.get("comment_harm_pct"),
            "per_source"      : result.get("per_source", {}),
            "harmful_texts"   : result.get("harmful_texts", [])[:10],
        })
    elif modality == "spoken":
        base.update({
            "total_chunks"   : result.get("total_chunks"),
            "harmful_chunks" : result.get("harmful_chunks"),
            "harm_percentage": result.get("harm_percentage"),
            "transcription"  : result.get("transcription", "")[:500],
            "language"       : result.get("language"),
            "was_translated" : result.get("was_translated"),
            "harmful_texts"  : result.get("harmful_texts", [])[:10],
        })

    return base


def get_video_history(youtube_id: str) -> list:
    """Return all past analyses for a given YouTube video ID, newest first."""
    video = Video.query.filter_by(youtube_id=youtube_id).first()
    if not video:
        return []
    analyses = Analysis.query.filter_by(video_id=video.id)\
                             .order_by(Analysis.created_at.desc()).all()
    return [a.to_dict() for a in analyses]


def get_recent_analyses(limit: int = 20) -> list:
    """Return the most recent analyses across all videos, newest first."""
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(limit).all()
    return [a.to_dict() for a in analyses]