import sys
sys.path.insert(0, r"D:\PythonYouTube\FrontEnd")

from services.video_analyser import analyse

result = analyse(video_id="dQw4w9WgXcQ")
print("VIDEO:", result["verdict"])
print("Unsafe frames:", result["unsafe_frames"], "/", result["total_frames"])
print("Unsafe ratio:", result["unsafe_ratio"])