"""
services/youtube_service.py
============================
Handles all YouTube-related operations:
  - Searching videos by query using the YouTube Data API v3
  - Fetching full video details (title, thumbnail, duration, channel, views)
  - Resolving a YouTube URL to a video ID
  - Validating whether a string is a YouTube URL
"""

import re
import requests
from config import (
    YOUTUBE_API_KEY,
    YOUTUBE_SEARCH_URL,
    YOUTUBE_VIDEOS_URL,
    YOUTUBE_SEARCH_LIMIT,
)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def is_youtube_url(text: str) -> bool:
    """Return True if the text looks like a YouTube URL."""
    return bool(re.search(r"(youtube\.com/watch|youtu\.be/)", text))


def extract_video_id(url: str) -> str | None:
    """
    Extract the 11-character video ID from a YouTube URL.
    Handles both formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
    Returns None if no ID is found.
    """
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",   # ?v= or &v=
        r"youtu\.be/([A-Za-z0-9_-]{11})" # youtu.be/
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def format_duration(iso_duration: str) -> str:
    """
    Convert ISO 8601 duration (e.g. 'PT4M13S') to a readable string ('4:13').
    Returns '--:--' if parsing fails.
    """
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        iso_duration or ""
    )
    if not match:
        return "--:--"

    hours   = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_view_count(count_str: str) -> str:
    """Format a raw view count string to a human-readable value (e.g. '1.2M views')."""
    try:
        count = int(count_str)
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M views"
        if count >= 1_000:
            return f"{count / 1_000:.0f}K views"
        return f"{count} views"
    except (ValueError, TypeError):
        return "N/A"


def _build_video_dict(item: dict, stats: dict = None) -> dict:
    """
    Build a normalised video dict from a YouTube Data API 'search' item.
    Optionally merges in stats (viewCount, etc.) from the videos endpoint.
    """
    snippet    = item.get("snippet", {})
    video_id   = item.get("id", {}).get("videoId", "")
    thumbnails = snippet.get("thumbnails", {})

    # Prefer high-res thumbnail, fall back through available sizes
    thumbnail = (
        thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )

    views    = ""
    duration = ""
    if stats:
        views    = format_view_count(stats.get("viewCount", ""))
        duration = format_duration(stats.get("duration", ""))

    return {
        "id":           video_id,
        "title":        snippet.get("title", "Untitled"),
        "channel":      snippet.get("channelTitle", "Unknown channel"),
        "published":    snippet.get("publishedAt", "")[:10],  # YYYY-MM-DD only
        "description":  snippet.get("description", ""),
        "thumbnail":    thumbnail,
        "views":        views,
        "duration":     duration,
        "youtube_url":  f"https://www.youtube.com/watch?v={video_id}",
        "embed_url":    f"https://www.youtube.com/embed/{video_id}",
    }


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def search_videos(query: str) -> tuple[list[dict], str | None]:
    """
    Search YouTube for the given query.

    Returns:
        (videos, error)
        videos : list of video dicts (empty list on failure)
        error  : error message string, or None on success
    """
    try:
        # Step 1 — search for video IDs
        search_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": YOUTUBE_SEARCH_LIMIT,
            "key": YOUTUBE_API_KEY,
            "eventType": "completed",  # excludes live and upcoming streams
            "videoDuration": "medium",  # excludes very short clips (under 4 min)
        }
        search_resp = requests.get(YOUTUBE_SEARCH_URL, params=search_params, timeout=10)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        items = search_data.get("items", [])
        if not items:
            return [], None   # valid response but no results

        # Collect video IDs so we can fetch stats in one call
        video_ids = [
            item["id"]["videoId"]
            for item in items
            if item.get("id", {}).get("videoId")
        ]

        # Step 2 — fetch duration + view count for all IDs at once
        stats_map = {}
        if video_ids:
            stats_params = {
                "part": "contentDetails,statistics",
                "id":   ",".join(video_ids),
                "key":  YOUTUBE_API_KEY,
            }
            stats_resp = requests.get(YOUTUBE_VIDEOS_URL, params=stats_params, timeout=10)
            if stats_resp.ok:
                for v in stats_resp.json().get("items", []):
                    stats_map[v["id"]] = {
                        **v.get("contentDetails", {}),
                        **v.get("statistics", {}),
                    }

        # Step 3 — build normalised video list
        videos = []
        for item in items:
            vid_id = item.get("id", {}).get("videoId", "")
            videos.append(_build_video_dict(item, stats_map.get(vid_id)))

        return videos, None

    except requests.exceptions.Timeout:
        return [], "YouTube search timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 403:
            return [], "YouTube API quota exceeded or API key is invalid."
        return [], f"YouTube API error (HTTP {status})."
    except Exception as e:
        return [], f"Unexpected error during search: {str(e)}"


def get_video_by_id(video_id: str) -> tuple[dict | None, str | None]:
    """
    Fetch full details for a single video by its ID.

    Returns:
        (video, error)
        video : video dict, or None on failure
        error : error message string, or None on success
    """
    try:
        params = {
            "part": "snippet,contentDetails,statistics",
            "id":   video_id,
            "key":  YOUTUBE_API_KEY,
        }
        resp = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return None, f"No video found with ID: {video_id}"

        item = items[0]
        stats = {
            **item.get("contentDetails", {}),
            **item.get("statistics", {}),
        }

        # Wrap as a search-style item so _build_video_dict works the same way
        wrapped = {
            "id":      {"videoId": video_id},
            "snippet": item.get("snippet", {}),
        }
        return _build_video_dict(wrapped, stats), None

    except requests.exceptions.Timeout:
        return None, "Request timed out while fetching video details."
    except Exception as e:
        return None, f"Error fetching video details: {str(e)}"


def get_video_from_url(url: str) -> tuple[dict | None, str | None]:
    """
    Convenience wrapper: resolve a YouTube URL → fetch full video details.

    Returns:
        (video, error)
    """
    video_id = extract_video_id(url)
    if not video_id:
        return None, "Could not extract a video ID from that URL."
    return get_video_by_id(video_id)