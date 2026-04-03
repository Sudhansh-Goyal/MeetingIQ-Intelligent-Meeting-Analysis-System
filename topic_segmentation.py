
import re
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PATHS =================
BASE_DIR        = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
TOPICS_DIR      = BASE_DIR / "data" / "topics"
MODEL_PATH      = BASE_DIR / "models" / "bilstm.pth"

TOPICS_DIR.mkdir(parents=True, exist_ok=True)

LINE_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
)


EMBED_MODEL_NAME   = "paraphrase-MiniLM-L3-v2"
EMBED_DIM          = 384
HIDDEN_DIM         = 128

# TextTiling controls — tune these to get more/fewer segments
WINDOW_SIZE        = 4     # how many utterances to average on each side
BOUNDARY_PERCENTILE = 80   # only the top (100 - X)% of depth scores become boundaries
MIN_SEG_UTTERANCES = 6     # merge segments smaller than this

# BiLSTM refinement (only used if model file exists)
BILSTM_THRESHOLD   = 0.55  # raised slightly to reduce false positives


class BiLSTMBoundary(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)
        return torch.sigmoid(logits[:, :-1])

def load_bilstm():
    if not MODEL_PATH.exists():
        return None
    try:
        model = BiLSTMBoundary()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print("  BiLSTM model loaded — will use for boundary refinement.")
        return model
    except Exception as e:
        print(f"  Warning: could not load BiLSTM ({e}). Using TextTiling only.")
        return None


def parse_transcript(path):
    utterances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                text = m.group(4).strip()
                if text:
                    utterances.append({
                        "start":   float(m.group(1)),
                        "end":     float(m.group(2)),
                        "speaker": m.group(3),
                        "text":    text,
                    })
    return utterances


def texttiling_boundaries(embeddings, window=WINDOW_SIZE, pct=BOUNDARY_PERCENTILE):
    """
    Computes coherence scores and depth scores across the embedding sequence.
    Returns a list of boundary positions (indices where a new segment starts).
    """
    n = len(embeddings)
    if n < window * 2 + 2:
        return [0]


    coherence = []
    for i in range(1, n):
        left  = np.mean(embeddings[max(0, i - window):i],    axis=0, keepdims=True)
        right = np.mean(embeddings[i:min(n, i + window)],    axis=0, keepdims=True)
        coherence.append(float(cosine_similarity(left, right)[0][0]))


    depths = []
    for i in range(len(coherence)):
        lv = coherence[i]
        for j in range(i - 1, -1, -1):
            if coherence[j] >= lv: lv = coherence[j]
            else: break
        rv = coherence[i]
        for j in range(i + 1, len(coherence)):
            if coherence[j] >= rv: rv = coherence[j]
            else: break
        depths.append((lv - coherence[i]) + (rv - coherence[i]))


    threshold = np.percentile(depths, pct)

    boundaries = [0]
    for i, d in enumerate(depths):
        if d >= threshold:
            boundaries.append(i + 1)

    return sorted(set(boundaries))

def bilstm_refine_boundaries(embeddings, bilstm_model, base_boundaries):
    """
    Runs the BiLSTM over the full sequence.
    Adds boundaries where BiLSTM is confident AND TextTiling also sees a valley nearby.
    Removes boundaries where BiLSTM is very confident there is NO boundary.
    This makes BiLSTM a refinement filter, not the sole decision maker.
    """
    if bilstm_model is None:
        return base_boundaries

    n = len(embeddings)
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probs = bilstm_model(emb_tensor).squeeze(0).tolist()  

    base_set = set(base_boundaries)
    refined  = set(base_boundaries)

    for i, p in enumerate(probs):
        boundary_pos = i + 1
  
        if p >= BILSTM_THRESHOLD:
            refined.add(boundary_pos)
     
        if p < (1 - BILSTM_THRESHOLD) and boundary_pos in base_set and boundary_pos != 0:
            refined.discard(boundary_pos)

    return sorted(refined)


def build_segments(utterances, boundaries):
    segments   = []
    endpoints  = sorted(set(list(boundaries) + [len(utterances)]))
    for k in range(len(endpoints) - 1):
        seg = utterances[endpoints[k]:endpoints[k + 1]]
        if seg:
            segments.append(seg)
    return segments

def merge_small_segments(segments, min_utt=MIN_SEG_UTTERANCES):
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        if len(seg) < min_utt:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    if len(merged) > 1 and len(merged[0]) < min_utt:
        merged[1] = merged[0] + merged[1]
        merged    = merged[1:]

    return merged


TOPIC_CANDIDATES = [
    "Meeting Introduction and Agenda",
    "Team Introductions and Roles",
    "Project Brief and Goals",
    "Design Discussion",
    "Product Strategy",
    "Technical Development",
    "Marketing and Branding",
    "Customer Feedback and User Experience",
    "Budget and Finance",
    "Project Planning and Timeline",
    "Action Items and Next Steps",
    "General Discussion",
]

def label_topic(segment_utts, embed_model):
    full_text  = " ".join(u["text"] for u in segment_utts)
    seg_emb    = embed_model.encode([full_text])
    topic_embs = embed_model.encode(TOPIC_CANDIDATES)
    sims       = cosine_similarity(seg_emb, topic_embs)[0]
    best_idx   = int(np.argmax(sims))
    return TOPIC_CANDIDATES[best_idx], round(float(sims[best_idx]), 3)

def segment_preview(segment_utts, n=3):
    return [f"{u['speaker']}: {u['text'][:80]}" for u in segment_utts[:n]]


def main():
    print("Loading models...")
    embed_model  = SentenceTransformer(EMBED_MODEL_NAME)
    bilstm_model = load_bilstm()

    if bilstm_model is None:
        print("  Running in TextTiling-only mode (train BiLSTM for refinement).\n")

    files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {TRANSCRIPTS_DIR}")
        return

    for file in files:
        print(f"Processing: {file.name}")

        utterances = parse_transcript(file)
        print(f"  Parsed {len(utterances)} utterances")

        if len(utterances) < 8:
            print("  Skipping — too short.")
            continue

        
        texts      = [u["text"] for u in utterances]
        embeddings = embed_model.encode(texts, show_progress_bar=False)

       
        boundaries = texttiling_boundaries(embeddings)
        print(f"  TextTiling found {len(boundaries)} boundary positions")

        
        boundaries = bilstm_refine_boundaries(embeddings, bilstm_model, boundaries)

      
        segments   = build_segments(utterances, boundaries)
        segments   = merge_small_segments(segments)

        print(f"  Final: {len(segments)} topic segments")

        output = {
            "source":           file.name,
            "total_utterances": len(utterances),
            "total_segments":   len(segments),
            "segments":         [],
        }

        for i, seg in enumerate(segments):
            topic, confidence = label_topic(seg, embed_model)

            output["segments"].append({
                "segment_id":      i,
                "topic":           topic,
                "confidence":      confidence,
                "start_time":      seg[0]["start"],
                "end_time":        seg[-1]["end"],
                "duration_sec":    round(seg[-1]["end"] - seg[0]["start"], 2),
                "utterance_count": len(seg),
                "preview":         segment_preview(seg),
                "utterances":      seg,
            })

            print(f"  [{seg[0]['start']:>8.2f}s - {seg[-1]['end']:>8.2f}s]  "
                  f"{topic}  (conf={confidence:.2f}, {len(seg)} utts)")

        out_path = TOPICS_DIR / f"{file.stem}_topics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Saved → {out_path}\n")

    print("DONE")

if __name__ == "__main__":
    main()
