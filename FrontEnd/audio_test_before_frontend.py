import sys
sys.path.insert(0, r"/FrontEnd")

# Test audio (YAMNet)
from services.audio_analyser import analyse as audio_analyse
result = audio_analyse(video_id="dQw4w9WgXcQ")
print("AUDIO:", result["verdict"], result["score"])

# Test spoken (Whisper + RoBERTa)
from services.spoken_analyser import analyse as spoken_analyse
result = spoken_analyse(video_id="dQw4w9WgXcQ")
print("SPOKEN:", result["verdict"], result["score"])