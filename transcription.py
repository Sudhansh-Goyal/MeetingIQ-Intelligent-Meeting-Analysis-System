import os
import torch
import torchaudio
import whisper

from pyannote.audio import Pipeline
from huggingface_hub import login

# ================== CONFIG ==================
HF_TOKEN = "YOUR_HF_TOKEN"

INPUT_FOLDER = "data/audio"
OUTPUT_FOLDER = "data/transcripts"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

login(HF_TOKEN)

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================== LOAD MODELS ==================
print("Loading Whisper model...")
asr_model = whisper.load_model("medium", device=device)

print("Loading Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
).to(device)

# ================== HELPER ==================
def extract_annotation(result):
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    if hasattr(result, "itertracks"):
        return result
    if hasattr(result, "annotation"):
        return result.annotation
    try:
        if hasattr(result[0], "itertracks"):
            return result[0]
    except:
        pass
    raise ValueError("Could not extract annotation")

# ================== PROCESS FUNCTION ==================
def process_audio(file_path):
    filename = os.path.basename(file_path)
    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename.replace(".wav", ".txt")
    )

    if os.path.exists(output_path):
        print(f"Skipping {filename}")
        return

    print(f"Processing {filename}")

    # Transcription
    asr_result = asr_model.transcribe(file_path)

    # Diarization
    waveform, sample_rate = torchaudio.load(file_path)
    diarization = extract_annotation(
        pipeline({"waveform": waveform, "sample_rate": sample_rate})
    )

    # Save TXT
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in asr_result["segments"]:
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()

            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.end > start and turn.start < end:
                    speakers.append(speaker)

            speaker_label = (
                max(set(speakers), key=speakers.count)
                if speakers else "UNKNOWN"
            )

            f.write(
                f"[{start:07.2f}-{end:07.2f}] {speaker_label}: {text}\n"
            )

    print(f"Saved: {output_path}")

# ================== RUN ==================
files = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".wav")
]

print(f"Found {len(files)} audio files")

for i, file in enumerate(files):
    print(f"[{i+1}/{len(files)}]", end=" ")
    process_audio(file)