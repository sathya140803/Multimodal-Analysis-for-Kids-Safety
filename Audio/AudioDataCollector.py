

import os
import time
import requests
from pathlib import Path


# CONFIGURATION


FREESOUND_API_KEY = "qg1Rb8iUAo1tBmpb2ofgiq66Cxxz5adKhM0GF0cP"

BASE_DIR   = r"D:\PythonYouTube\Audio\data"
SAFE_DIR   = os.path.join(BASE_DIR, "safe")
UNSAFE_DIR = os.path.join(BASE_DIR, "unsafe")

CLIPS_PER_QUERY = 20      # increased to get more results per query
MIN_DURATION    = 2
MAX_DURATION    = 30
REQUEST_DELAY   = 0.5     # seconds between requests — avoids rate limiting

# SOUND CATEGORIES
# 50 safe queries + 50 unsafe queries


SAFE_QUERIES = [
    # ── Children's play and laughter ─────────────────────────────────────────
    "children laughing",
    "kids playing",
    "baby laughing",
    "children singing",
    "children playing outside",
    "toddler giggling",
    "kids cheering",
    "playground sounds",
    "children clapping",
    "baby cooing",

    # ── Gentle animals ────────────────────────────────────────────────────────
    "bird singing",
    "duck quacking",
    "cat purring",
    "dog barking friendly",
    "birds chirping forest",
    "rooster crowing",
    "cow mooing",
    "horse neighing",
    "sheep baa",
    "frog croaking",

    # ── Cartoon and toy sounds ────────────────────────────────────────────────
    "cartoon sound effect",
    "toy music box",
    "xylophone melody",
    "bubble pop sound",
    "cartoon boing",
    "squeaky toy",
    "wind up toy",
    "cartoon whistle",
    "playful bell",
    "cartoon sparkle",

    # ── Calm nature ───────────────────────────────────────────────────────────
    "rain gentle",
    "ocean waves calm",
    "wind chimes",
    "river stream flowing",
    "forest ambience birds",
    "light breeze leaves",
    "waterfall peaceful",
    "crickets night",
    "campfire crackling",
    "morning birds",

    # ── Music and educational ─────────────────────────────────────────────────
    "nursery rhyme music",
    "children music happy",
    "clapping audience",
    "cheerful bells",
    "happy music kids",
    "ukulele happy",
    "acoustic guitar gentle",
    "piano lullaby",
    "marimba playful",
    "flute happy melody",

    # ── Extra safe — to reach 750 ─────────────────────────────────────────────
    "ice cream truck music",
    "merry go round music",
    "kids party music",
    "school bell ringing",
    "kindergarten music",
    "lullaby music gentle",
    "happy whistle tune",
    "recorder flute children",
    "music box lullaby",
    "triangle instrument sound",
    "tambourine music",
    "harmonica happy",
    "banjo happy music",
    "cat meowing",
    "puppy sounds",
    "parrot sounds",
    "hamster sounds",
    "rabbit sounds",
    "dolphin sounds",
    "whale sounds",
    "raindrop sounds",
    "thunderstorm gentle rain",
    "beach waves seagulls",
    "forest stream birds",
    "autumn leaves wind",
]

UNSAFE_QUERIES = [
    # ── Violence and aggression ───────────────────────────────────────────────
    "gunshot",
    "explosion loud",
    "glass breaking violent",
    "fight sounds punching",
    "sword fight",
    "bomb explosion",
    "machine gun",
    "bullet impact",
    "grenade explosion",
    "violent crash",

    # ── Distress ──────────────────────────────────────────────────────────────
    "screaming terror",
    "crying distress",
    "person screaming fear",
    "baby crying distress",
    "woman screaming",
    "man yelling anger",
    "painful scream",
    "sobbing crying",
    "panic scream",
    "distressed crying",

    # ── Horror and jump scare ─────────────────────────────────────────────────
    "horror sound effect",
    "scary music stinger",
    "creepy ambience",
    "demon growl",
    "evil laugh",
    "horror atmosphere",
    "haunted house sound",
    "monster roar",
    "ghost sound",
    "zombie sound",

    # ── Disturbing and startling ──────────────────────────────────────────────
    "loud alarm sudden",
    "thunder crack loud",
    "metal screeching",
    "jump scare sound",
    "loud bang sudden",
    "car crash impact",
    "chainsaw sound",
    "drill loud",
    "industrial noise harsh",
    "electrical shock sound",

    # ── Sexually suggestive (unspoken audio) ──────────────────────────────────
    "moaning",
    "heavy breathing intense",
    "kissing sounds",
    "ASMR whispering",
    "sensual music",
    "seductive music",
    "intimate sounds",
    "provocative music",
    "adult breathing",
    "whisper seductive",

    # ── Extra unsafe — to reach 750 ───────────────────────────────────────────
    "knife stabbing sound",
    "torture scream",
    "violent struggle sounds",
    "horror violin screech",
    "dark ambient horror",
    "creepy music box",
    "sinister laughter",
    "threatening voice",
    "aggressive dog barking",
    "bones cracking horror",
    "horror heartbeat",
    "scary breathing heavy",
    "slasher sound effect",
    "thunder storm intense",
    "nuclear alarm siren",
    "air raid siren",
    "police siren chase",
    "skull cracking sound",
    "horror ambience dark",
    "violence impact sound",
    "war sounds battle",
    "sniper shot",
    "axe chopping",
    "taser shock sound",
    "horror drone dark",
]

# FREESOUND API CLIENT


class FreeSoundClient:
    BASE_URL = "https://freesound.org/apiv2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {api_key}"})

    def search(self, query: str, num_results: int = 15,
               min_duration: float = 2, max_duration: float = 30) -> list:
        params = {
            "query":     query,
            "page_size": num_results,
            "fields":    "id,name,duration,previews,license",
            "filter":    f"duration:[{min_duration} TO {max_duration}]",
            "sort":      "score",
        }
        try:
            resp = self.session.get(f"{self.BASE_URL}/search/text/",
                                    params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"    [API ERROR] '{query}': {e}")
            return []

    def download(self, sound: dict, save_path: str) -> bool:
        preview_url = sound.get("previews", {}).get("preview-hq-mp3")
        if not preview_url:
            return False
        try:
            resp = self.session.get(preview_url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            print(f"    [DOWNLOAD ERROR] {e}")
            return False


# DOWNLOAD LOGIC
def sanitize_filename(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def download_category(client: FreeSoundClient, queries: list,
                      save_dir: str, label: str):

    os.makedirs(save_dir, exist_ok=True)
    total_downloaded = 0
    total_skipped    = 0

    print(f"\n{'='*60}")
    print(f"  {label.upper()} sounds  →  {save_dir}")
    print(f"  Target: {len(queries)} queries x {CLIPS_PER_QUERY} clips "
          f"= up to {len(queries) * CLIPS_PER_QUERY} files")
    print(f"{'='*60}")

    for q_num, query in enumerate(queries, 1):
        print(f"\n  [{q_num}/{len(queries)}] Query: \"{query}\"")

        sounds = client.search(
            query,
            num_results=CLIPS_PER_QUERY,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION
        )

        if not sounds:
            print(f"    No results.")
            continue

        downloaded = 0
        for sound in sounds:
            sound_id   = sound["id"]
            sound_name = sanitize_filename(sound["name"])
            filename   = f"{label}_{sound_id}_{sound_name[:40]}.mp3"
            save_path  = os.path.join(save_dir, filename)

            if os.path.exists(save_path):
                total_skipped += 1
                continue

            if client.download(sound, save_path):
                downloaded      += 1
                total_downloaded += 1
                print(f"    ✓ [{downloaded}/{len(sounds)}] {filename}")
            else:
                print(f"    ✗ Failed: {sound_name}")

            time.sleep(REQUEST_DELAY)

        print(f"    → {downloaded} new  |  running total: {total_downloaded}")

    print(f"\n  {label.upper()} done: {total_downloaded} downloaded, "
          f"{total_skipped} already existed")
    return total_downloaded


def verify_api_key(client: FreeSoundClient) -> bool:
    print("[*] Verifying API key...")
    if client.search("music", num_results=1):
        print("[*] API key valid.\n")
        return True
    print("[ERROR] API key invalid. Check FREESOUND_API_KEY.\n")
    return False


# SUMMARY
def print_summary():
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")

    total = 0
    for label, folder in [("safe", SAFE_DIR), ("unsafe", UNSAFE_DIR)]:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]
            print(f"  {label.upper():8s}: {len(files):4d} files  →  {folder}")
            total += len(files)
        else:
            print(f"  {label.upper():8s}:    0 files")

    print(f"  {'TOTAL':8s}: {total:4d} files")
    print(f"\n  Target was 1500 (750 safe + 750 unsafe)")

    if total >= 1400:
        print(f"  Status: TARGET REACHED ✓")
    else:
        print(f"  Status: {1500 - total} more files needed")

    print(f"\n  You can now run yamnet_classifier.py\n")


# MAIN

def main():
    print("\n" + "="*60)
    print("  FreeSound Dataset Downloader — Child Safety Audio")
    print(f"  Target: 1500 clips  (750 safe + 750 unsafe)")
    print("="*60 + "\n")

    client = FreeSoundClient(FREESOUND_API_KEY)

    if not verify_api_key(client):
        return

    download_category(client, SAFE_QUERIES,   SAFE_DIR,   label="safe")
    download_category(client, UNSAFE_QUERIES, UNSAFE_DIR, label="unsafe")
    print_summary()


if __name__ == "__main__":
    main()