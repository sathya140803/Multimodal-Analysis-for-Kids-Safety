import sys
sys.path.insert(0, r"D:\PythonYouTube\FrontEnd")

from services.text_analyser import analyse

result = analyse(video_id="dQw4w9WgXcQ")  # any YouTube video ID
print(result["verdict"])
print(result["score"])
print(result["total_harmful"], "/", result["total_analysed"])