import torch
import whisper
import transformers
import librosa
import soundfile as sf

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Whisper loaded")
print("Transformers:", transformers.__version__)
print("Librosa OK")
print("Soundfile OK")

# Optional: tiny inference test
model = whisper.load_model("tiny")
print("Whisper model loaded successfully")