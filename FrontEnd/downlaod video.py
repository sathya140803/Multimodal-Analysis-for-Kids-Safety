import yt_dlp
import os

def download_video(url: str, output_folder: str = "downloads") -> str:
    """
    Downloads a YouTube video with best video+audio merged into MP4.
    Returns the path of the downloaded file.
    """

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # best quality
        'merge_output_format': 'mp4',          # ensure mp4 output
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        # Ensure final file is .mp4
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    return filename


if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()

    try:
        path = download_video(url)
        print(f"\nDownloaded successfully:")
        print(path)
    except Exception as e:
        print(f"\nDownload failed: {e}")