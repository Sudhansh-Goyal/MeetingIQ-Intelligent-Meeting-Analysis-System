# import re
# import json
# import pickle
# import torch
# import torch.nn as nn
# from pathlib import Path

# BASE_DIR = Path(__file__).parent
# TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
# TOPICS_DIR = BASE_DIR / "data" / "topics"
# MODEL_PATH = BASE_DIR / "models" / "bilstm.pth"
# TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

# TOPICS_DIR.mkdir(parents=True, exist_ok=True)

# THRESHOLD = 0.5

# LINE_RE = re.compile(
#     r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
# )

# # ================= MODEL =================
# class BiLSTM(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, 128)
#         self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(256, 1)

#     def forward(self, x):
#         x = self.embedding(x)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return torch.sigmoid(out)

# # ================= LOAD =================
# def load_tokenizer():
#     with open(TOKENIZER_PATH, "rb") as f:
#         return pickle.load(f)

# def load_model(vocab_size):
#     model = BiLSTM(vocab_size)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#     return model

# # ================= PARSE =================
# def parse_transcript(path):
#     utterances = []

#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             m = LINE_RE.match(line.strip())
#             if m:
#                 utterances.append({
#                     "start": float(m.group(1)),
#                     "end": float(m.group(2)),
#                     "speaker": m.group(3),
#                     "text": m.group(4)
#                 })

#     return utterances

# # ================= SEGMENT =================
# def segment(utterances, model, tokenizer):
#     segments = []
#     current = [utterances[0]]

#     for utt in utterances[1:]:
#         tokens = tokenizer.encode(utt["text"])
#         x = torch.tensor([tokens], dtype=torch.long)

#         with torch.no_grad():
#             prob = model(x).item()

#         if prob >= THRESHOLD:
#             segments.append(current)
#             current = [utt]
#         else:
#             current.append(utt)

#     segments.append(current)
#     return segments

# # ================= MAIN =================
# def main():
#     tokenizer = load_tokenizer()
#     model = load_model(len(tokenizer.word2idx))

#     files = list(TRANSCRIPTS_DIR.glob("*.txt"))

#     for file in files:
#         print(f"Processing {file.name}")

#         utterances = parse_transcript(file)
#         segments = segment(utterances, model, tokenizer)

#         output = {
#             "source": file.name,
#             "segments": []
#         }

#         for i, seg in enumerate(segments):
#             output["segments"].append({
#                 "segment_id": i,
#                 "start_time": seg[0]["start"],
#                 "end_time": seg[-1]["end"],
#                 "utterances": seg
#             })

#         out_path = TOPICS_DIR / f"{file.stem}_topics.json"

#         with open(out_path, "w") as f:
#             json.dump(output, f, indent=2)

#         print(f"Saved: {out_path}")

# if __name__ == "__main__":
#     main()


# import re
# import json
# import pickle
# import torch
# import torch.nn as nn
# from pathlib import Path

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # ================= PATHS =================
# BASE_DIR = Path(__file__).parent
# TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
# TOPICS_DIR = BASE_DIR / "data" / "topics"
# MODEL_PATH = BASE_DIR / "models" / "bilstm.pth"
# TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

# TOPICS_DIR.mkdir(parents=True, exist_ok=True)

# THRESHOLD = 0.5

# LINE_RE = re.compile(
#     r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
# )

# # ================= TOKENIZER =================
# class Tokenizer:
#     def __init__(self):
#         self.word2idx = {"<PAD>": 0, "<UNK>": 1}

#     def encode(self, text):
#         MAX_LEN = 50
#         tokens = text.lower().split()[:MAX_LEN]
#         ids = [self.word2idx.get(w, 1) for w in tokens]
#         ids += [0] * (MAX_LEN - len(ids))
#         return ids

# # ================= MODEL =================
# class BiLSTM(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, 128)
#         self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(256, 1)

#     def forward(self, x):
#         x = self.embedding(x)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return torch.sigmoid(out)

# # ================= LOAD =================
# def load_tokenizer():
#     with open(TOKENIZER_PATH, "rb") as f:
#         return pickle.load(f)

# def load_model(vocab_size):
#     model = BiLSTM(vocab_size)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#     return model

# # ================= PARSE =================
# def parse_transcript(path):
#     utterances = []

#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             m = LINE_RE.match(line.strip())
#             if m:
#                 utterances.append({
#                     "start": float(m.group(1)),
#                     "end": float(m.group(2)),
#                     "speaker": m.group(3),
#                     "text": m.group(4)
#                 })

#     return utterances

# # ================= SEGMENT =================
# def segment(utterances, model, tokenizer):
#     if not utterances:
#         return []

#     segments = []
#     current = [utterances[0]]

#     for utt in utterances[1:]:
#         # ✅ CONTEXT INPUT
#         prev_text = current[-1]["text"]
#         pair_text = prev_text + " " + utt["text"]

#         tokens = tokenizer.encode(pair_text)
#         x = torch.tensor([tokens], dtype=torch.long)

#         with torch.no_grad():
#             prob = model(x).item()

#         if prob >= THRESHOLD:
#             segments.append(current)
#             current = [utt]
#         else:
#             current.append(utt)

#     segments.append(current)
#     return segments

# # ================= MERGE SMALL SEGMENTS =================
# def merge_small_segments(segments, min_utterances=3):
#     merged = []
#     buffer = []

#     for seg in segments:
#         if len(seg) < min_utterances:
#             buffer.extend(seg)
#         else:
#             if buffer:
#                 seg = buffer + seg
#                 buffer = []
#             merged.append(seg)

#     if buffer:
#         if merged:
#             merged[-1].extend(buffer)
#         else:
#             merged.append(buffer)

#     return merged

# # ================= TOPIC LABELING =================
# TOPIC_CANDIDATES = [
#     "Product Strategy Discussion",
#     "Sales and Revenue Discussion",
#     "Marketing and Branding",
#     "Customer Feedback and Support",
#     "Technical Development",
#     "Project Planning",
#     "Budget and Finance",
#     "Team Management",
#     "General Discussion"
# ]

# def generate_topic(segment, embed_model):
#     texts = [utt["text"] for utt in segment]

#     if not texts:
#         return "General Discussion"

#     full_text = " ".join(texts)

#     segment_embedding = embed_model.encode([full_text])[0]
#     topic_embeddings = embed_model.encode(TOPIC_CANDIDATES)

#     sims = cosine_similarity([segment_embedding], topic_embeddings)[0]
#     best_idx = sims.argmax()

#     return TOPIC_CANDIDATES[best_idx]

# # ================= MAIN =================
# def main():
#     print("STARTING topic segmentation...")

#     tokenizer = load_tokenizer()
#     model = load_model(len(tokenizer.word2idx))
#     embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

#     files = list(TRANSCRIPTS_DIR.glob("*.txt"))

#     for file in files:
#         print(f"\nProcessing: {file.name}")

#         utterances = parse_transcript(file)

#         segments = segment(utterances, model, tokenizer)

#         # ✅ FIX OVER-SEGMENTATION
#         segments = merge_small_segments(segments)

#         output = {
#             "source": file.name,
#             "segments": []
#         }

#         for i, seg in enumerate(segments):
#             topic = generate_topic(seg, embed_model)

#             output["segments"].append({
#                 "segment_id": i,
#                 "topic": topic,
#                 "start_time": seg[0]["start"],
#                 "end_time": seg[-1]["end"],
#                 "utterances": seg
#             })

#         out_path = TOPICS_DIR / f"{file.stem}_topics.json"

#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(output, f, indent=2)

#         print(f"Saved: {out_path}")

#     print("\nDONE")

# # ================= RUN =================
# if __name__ == "__main__":
#     main()

import re
import json
import pickle
import torch
import torch.nn as nn
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PATHS =================
BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
TOPICS_DIR = BASE_DIR / "data" / "topics"
MODEL_PATH = BASE_DIR / "models" / "bilstm.pth"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"

TOPICS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

LINE_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
)

# ================= TOKENIZER =================
class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def encode(self, text):
        MAX_LEN = 50
        tokens = text.lower().split()[:MAX_LEN]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        ids += [0] * (MAX_LEN - len(ids))
        return ids

# ================= MODEL =================
class BiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# ================= FIXED LOAD TOKENIZER =================
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Tokenizer":
            return Tokenizer
        return super().find_class(module, name)

def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return CustomUnpickler(f).load()

def load_model(vocab_size):
    model = BiLSTM(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ================= PARSE =================
def parse_transcript(path):
    utterances = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                utterances.append({
                    "start": float(m.group(1)),
                    "end": float(m.group(2)),
                    "speaker": m.group(3),
                    "text": m.group(4)
                })

    return utterances

# ================= SEGMENT =================
def segment(utterances, model, tokenizer):
    if not utterances:
        return []

    segments = []
    current = [utterances[0]]

    for utt in utterances[1:]:
        prev_text = current[-1]["text"]
        pair_text = prev_text + " " + utt["text"]

        tokens = tokenizer.encode(pair_text)
        x = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            prob = model(x).item()

        if prob >= THRESHOLD:
            segments.append(current)
            current = [utt]
        else:
            current.append(utt)

    segments.append(current)
    return segments

# ================= MERGE SMALL SEGMENTS =================
def merge_small_segments(segments, min_utterances=3):
    merged = []
    buffer = []

    for seg in segments:
        if len(seg) < min_utterances:
            buffer.extend(seg)
        else:
            if buffer:
                seg = buffer + seg
                buffer = []
            merged.append(seg)

    if buffer:
        if merged:
            merged[-1].extend(buffer)
        else:
            merged.append(buffer)

    return merged

# ================= TOPIC LABELING =================
TOPIC_CANDIDATES = [
    "Product Strategy Discussion",
    "Sales and Revenue Discussion",
    "Marketing and Branding",
    "Customer Feedback and Support",
    "Technical Development",
    "Project Planning",
    "Budget and Finance",
    "Team Management",
    "General Discussion"
]

def generate_topic(segment, embed_model):
    texts = [utt["text"] for utt in segment]

    if not texts:
        return "General Discussion"

    full_text = " ".join(texts)

    segment_embedding = embed_model.encode([full_text])[0]
    topic_embeddings = embed_model.encode(TOPIC_CANDIDATES)

    sims = cosine_similarity([segment_embedding], topic_embeddings)[0]
    best_idx = sims.argmax()

    return TOPIC_CANDIDATES[best_idx]

# ================= MAIN =================
def main():
    print("STARTING topic segmentation...")

    tokenizer = load_tokenizer()
    model = load_model(len(tokenizer.word2idx))
    embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    files = list(TRANSCRIPTS_DIR.glob("*.txt"))

    for file in files:
        print(f"\nProcessing: {file.name}")

        utterances = parse_transcript(file)

        segments = segment(utterances, model, tokenizer)
        segments = merge_small_segments(segments)

        output = {
            "source": file.name,
            "segments": []
        }

        for i, seg in enumerate(segments):
            topic = generate_topic(seg, embed_model)

            output["segments"].append({
                "segment_id": i,
                "topic": topic,
                "start_time": seg[0]["start"],
                "end_time": seg[-1]["end"],
                "utterances": seg
            })

        out_path = TOPICS_DIR / f"{file.stem}_topics.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Saved: {out_path}")

    print("\nDONE")

# ================= RUN =================
if __name__ == "__main__":
    main()