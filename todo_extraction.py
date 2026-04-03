
from pathlib import Path
import re
import pandas as pd
import spacy
from spacy.matcher import Matcher
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# 1. PROJECT PATHS
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "transcripts"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CACHE = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)



print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)
matcher.add("ACTION_PATTERNS", [
    [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["need", "have"]}}, {"LOWER": "to"}, {"POS": "VERB"}],
    [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["should", "must", "will"]}}, {"POS": "VERB"}],
    [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["'ll", "'d"]}}, {"POS": "VERB"}],
    [{"LOWER": "let"}, {"LOWER": "'s"}, {"POS": "VERB"}],
    [{"LOWER": "i"}, {"LOWER": {"IN": ["am", "'m"]}}, {"LOWER": "going"}, {"LOWER": "to"}, {"POS": "VERB"}]
])

WEAK_WORDS = {"think", "guess", "maybe", "wonder", "probably", "if", "depends", "say", "would", "could"}
IGNORE_VERBS = {"go", "see", "think", "talk", "say", "know", "mean", "do", "look", "be", "get", "come", "have", "want"}


# =========================
# 3. LOAD FLAN-T5 MODEL
# =========================
print("Loading FLAN-T5 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base",
    cache_dir=MODEL_CACHE
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base",
    cache_dir=MODEL_CACHE
).to(device)

print(f"Using device: {device}")


# =========================
# 4. HELPER FUNCTIONS
# =========================
def parse_transcript(path: Path):
    pattern = r"\[(\d+\.\d+) - (\d+\.\d+)\]\s+(SPEAKER_\d+):\s+(.*)"
    segments = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                start, end, speaker, text = match.groups()
                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker,
                    "text": text.strip()
                })

    return segments


def is_incomplete(text):
    return not text.strip().endswith((".", "?", "!"))


def merge_segments(segments, max_gap=1.0):
    if not segments:
        return []

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        same_speaker = current["speaker"] == next_seg["speaker"]
        time_gap = next_seg["start"] - current["end"]

        if same_speaker and time_gap <= max_gap and is_incomplete(current["text"]):
            current["text"] += " " + next_seg["text"]
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)
    return merged


def clean_text(text):
    text = re.sub(r"SPEAKER_\d+:", "", text)

    fillers = ["okay", "alright", "so", "well", "um", "uh"]
    for filler in fillers:
        text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_sentence(text):
    return len(text.split()) > 4 and not text.endswith("?")


def pre_filter_task(text):
    doc = nlp(text)

    if any(token.lower_ in WEAK_WORDS for token in doc):
        return False

    matches = matcher(doc)
    if not matches:
        return False

    _, start, end = matches[0]
    verb_token = doc[end - 1]

    if verb_token.lemma_ in IGNORE_VERBS:
        return False

    return True


def rewrite_with_llm(task_text, context_block):
    prompt = f"""
Rewrite the raw text into a short actionable to-do item.
Replace unclear words like 'it' or 'this' using the context.

Context: {context_block}
Raw Text: {task_text}
Task:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.0,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace("Task:", "").strip().capitalize()


# =========================
# 5. MAIN PROCESSOR
# =========================
def process_all_files(data_dir: Path):
    all_rows = []

    txt_files = list(data_dir.glob("*.txt"))

    if not txt_files:
        print("No transcript files found.")
        return pd.DataFrame()

    for file_path in txt_files:
        print(f"Processing: {file_path.name}")

        segments = parse_transcript(file_path)
        merged_segments = merge_segments(segments)

        conversation_history = []

        for seg in merged_segments:
            text = clean_text(seg["text"])
            conversation_history.append(text)

            if len(conversation_history) > 3:
                conversation_history.pop(0)

            if is_valid_sentence(text) and pre_filter_task(text):
                context_block = " ".join(conversation_history[:-1])

                final_task = rewrite_with_llm(text, context_block)

                all_rows.append({
                    "file": file_path.name,
                    "speaker": seg["speaker"],
                    "task": final_task,
                    "raw_text": text
                })

    return pd.DataFrame(all_rows)



if __name__ == "__main__":
    print("Starting transcript processing...\n")

    df_final = process_all_files(DATA_DIR)

    if not df_final.empty:
        output_file = OUTPUT_DIR / "meeting_todos_local_genai.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\nSaved output to: {output_file}")
        print(df_final.head(10))
    else:
        print("No tasks extracted.")