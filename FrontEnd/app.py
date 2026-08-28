import os
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session
)
from config import (
    SECRET_KEY, MAX_UPLOAD_MB, UPLOAD_DIR,
    ALLOWED_EXTENSIONS, DATABASE_URI
)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"]             = MAX_UPLOAD_MB * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"]        = DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

from database.db import db
db.init_app(app)

with app.app_context():
    db.create_all()


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────────────────────────
# 1. HOME PAGE
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


# ─────────────────────────────────────────────────────────────────
# 2. SEARCH
# ─────────────────────────────────────────────────────────────────

@app.route("/search", methods=["POST"])
def search():
    from services.youtube_service import search_videos, is_youtube_url, get_video_from_url

    query = request.form.get("search_query", "").strip()

    if not query:
        flash("Please enter a search term or YouTube URL.", "warning")
        return redirect(url_for("home"))

    if is_youtube_url(query):
        video, error = get_video_from_url(query)
        if error:
            flash(error, "danger")
            return redirect(url_for("home"))
        session["selected_video"] = video
        return redirect(url_for("analysis_setup"))

    videos, error = search_videos(query)
    if error:
        flash(error, "danger")
        return redirect(url_for("home"))

    return render_template("results.html", query=query, videos=videos)


# ─────────────────────────────────────────────────────────────────
# 3. SELECT VIDEO FROM RESULTS
# ─────────────────────────────────────────────────────────────────

@app.route("/select-video", methods=["POST"])
def select_video():
    from services.youtube_service import get_video_by_id

    video_id = request.form.get("video_id", "").strip()
    if not video_id:
        flash("No video selected.", "warning")
        return redirect(url_for("home"))

    video, error = get_video_by_id(video_id)
    if error:
        flash(error, "danger")
        return redirect(url_for("home"))

    session["selected_video"] = video
    return redirect(url_for("analysis_setup"))


# ─────────────────────────────────────────────────────────────────
# 4. LOCAL VIDEO UPLOAD
# ─────────────────────────────────────────────────────────────────

@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "video_file" not in request.files:
        flash("No file selected.", "warning")
        return redirect(url_for("home"))

    file = request.files["video_file"]

    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "danger")
        return redirect(url_for("home"))

    from werkzeug.utils import secure_filename
    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    local_video = {
        "id"         : None,
        "title"      : os.path.splitext(filename)[0],
        "channel"    : "Local file",
        "published"  : "",
        "description": "",
        "thumbnail"  : "",
        "views"      : "",
        "duration"   : "",
        "youtube_url": "",
        "embed_url"  : "",
        "local_path" : save_path,
        "is_local"   : True,
    }

    session["selected_video"] = local_video
    return redirect(url_for("analysis_setup"))


# ─────────────────────────────────────────────────────────────────
# 5. ANALYSIS SETUP PAGE
# ─────────────────────────────────────────────────────────────────

@app.route("/analysis")
def analysis_setup():
    video = session.get("selected_video")
    if not video:
        flash("No video selected. Please search or upload a video first.", "warning")
        return redirect(url_for("home"))
    return render_template("analysis.html", video=video)


# ─────────────────────────────────────────────────────────────────
# 6. RUN FILTER — cache check first, then models
# ─────────────────────────────────────────────────────────────────

@app.route("/run-filter", methods=["POST"])
def run_filter():
    selected_modes = request.form.getlist("filter_type")
    video          = session.get("selected_video")

    if not selected_modes:
        flash("Please select at least one filtering type.", "warning")
        return redirect(url_for("analysis_setup"))

    if not video:
        flash("Session expired. Please select a video again.", "warning")
        return redirect(url_for("home"))

    video_id   = video.get("id")
    is_local   = video.get("is_local", False)
    local_path = video.get("local_path")

    # ── Cache check — skip models if already analysed ─────────────
    from database.db import get_cached_analysis
    cache = get_cached_analysis(video_id, selected_modes)
    if cache:
        print(f"[Cache] Returning cached result from {cache['created_at']}")
        return render_template(
            "report.html",
            video=video,
            selected_modes=selected_modes,
            results=cache["results"],
            verdict=cache["verdict"],
            combined_score=cache["combined_score"],
            modality_scores={},
            from_cache=True,
            cached_at=cache["created_at"],
            cache_is_blocked=cache.get("is_blocked", False),
        )

    # ── Run selected analysers ────────────────────────────────────
    video_result  = None
    audio_result  = None
    text_result   = None
    spoken_result = None

    if "Video" in selected_modes:
        from services.video_analyser import analyse as video_analyse
        video_result = video_analyse(
            video_id=video_id, is_local=is_local, local_path=local_path
        )

    if "Audio" in selected_modes:
        from services.audio_analyser import analyse as audio_analyse
        audio_result = audio_analyse(
            video_id=video_id, is_local=is_local, local_path=local_path
        )

    if "Text" in selected_modes:
        from services.text_analyser import analyse as text_analyse
        text_result = text_analyse(
            video_id=video_id, is_local=is_local, local_path=local_path
        )

    # Spoken runs automatically when Audio is selected
    if "Audio" in selected_modes:
        from services.spoken_analyser import analyse as spoken_analyse
        spoken_result = spoken_analyse(
            video_id=video_id, is_local=is_local, local_path=local_path
        )

    # ── Fuse scores ───────────────────────────────────────────────
    from services.score_fusion import fuse

    fusion_modes = list(selected_modes)
    if "Audio" in fusion_modes and "Spoken" not in fusion_modes:
        fusion_modes.append("Spoken")

    fusion = fuse(
        selected_modes=fusion_modes,
        video_result=video_result,
        audio_result=audio_result,
        text_result=text_result,
        spoken_result=spoken_result,
    )

    results = {
        "video" : video_result,
        "audio" : audio_result,
        "text"  : text_result,
        "spoken": spoken_result,
    }

    # ── Save to database ──────────────────────────────────────────
    try:
        from database.db import save_analysis
        save_analysis(
            video_dict     = video,
            selected_modes = selected_modes,
            results        = results,
            fusion         = fusion,
        )
        print(f"[DB] Analysis saved. Verdict: {fusion['verdict']}")
    except Exception as e:
        print(f"[DB] WARNING — could not save analysis: {e}")

    return render_template(
        "report.html",
        video=video,
        selected_modes=selected_modes,
        results=results,
        verdict=fusion["verdict"],
        combined_score=fusion["combined_score"],
        modality_scores=fusion["modality_scores"],
        from_cache=False,
        cached_at=None,
        cache_is_blocked=False,
    )


# ─────────────────────────────────────────────────────────────────
# 7. BLOCK VIDEO
# ─────────────────────────────────────────────────────────────────

@app.route("/block-video", methods=["POST"])
def block_video():
    from database.db import block_video as db_block_video
    youtube_id = request.form.get("youtube_id", "").strip()

    if youtube_id:
        db_block_video(youtube_id)

    flash("Video has been blocked. It will be flagged as UNSAFE on future analyses.", "warning")
    return redirect(url_for("home"))


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)