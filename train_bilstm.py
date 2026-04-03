

"""
train_bilstm.py  —  trains a BiLSTM boundary classifier on your transcripts.

ARCHITECTURE
------------
Instead of feeding raw token IDs into a BiLSTM (which learns nothing useful
from a tiny corpus), we feed a SEQUENCE of sentence embeddings per transcript.
The BiLSTM reads the full sequence and predicts at each position whether a
topic boundary exists after that utterance.

  Input  : (batch, seq_len, embed_dim)   — sentence embeddings
  Output : (batch, seq_len - 1)          — P(boundary after position i)
  Labels : depth-score TextTiling signal — consistent with inference
"""

import re
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR        = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
MODEL_DIR       = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


EMBED_MODEL_NAME    = "paraphrase-MiniLM-L3-v2"
EMBED_DIM           = 384
HIDDEN_DIM          = 128
WINDOW_SIZE         = 3
BOUNDARY_PERCENTILE = 75
EPOCHS              = 25
LR                  = 1e-3
BATCH_SIZE          = 4

LINE_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
)


def parse_transcript(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                t = m.group(4).strip()
                if t:
                    texts.append(t)
    return texts


def compute_depth_labels(embeddings, window=WINDOW_SIZE, pct=BOUNDARY_PERCENTILE):
    """
    TextTiling depth score. Returns binary labels of length (n-1):
      label[i] = 1 → boundary AFTER utterance i.
    """
    n = len(embeddings)
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
    return [1.0 if d >= threshold else 0.0 for d in depths]


class BiLSTMBoundary(nn.Module):
    """
    Sequence-level BiLSTM. Reads sentence embeddings and predicts
    boundary probability between every adjacent pair.

    Input  : (batch, seq_len, embed_dim)
    Output : (batch, seq_len - 1)
    """
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


print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

dataset = []  # list of (emb_tensor, lbl_tensor)

for file in TRANSCRIPTS_DIR.glob("*.txt"):
    texts = parse_transcript(file)
    if len(texts) < 6:
        print(f"  Skipping {file.name}: too short ({len(texts)} utterances)")
        continue

    print(f"  {file.name}: {len(texts)} utterances")
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    labels     = compute_depth_labels(embeddings)

    dataset.append((
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(labels,     dtype=torch.float32),
    ))

if not dataset:
    print("No usable transcripts found in data/transcripts/")
    exit(1)

total_boundaries = sum(lbl.sum().item() for _, lbl in dataset)
total_pairs      = sum(lbl.shape[0]     for _, lbl in dataset)
print(f"\nTranscripts  : {len(dataset)}")
print(f"Boundary ratio: {total_boundaries / total_pairs:.2%}")


model     = BiLSTMBoundary(embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn   = nn.BCELoss()

print("\nTraining BiLSTM...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    indices    = torch.randperm(len(dataset)).tolist()

    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx  = indices[i:i + BATCH_SIZE]
        batch_embs = [dataset[j][0] for j in batch_idx]
        batch_lbls = [dataset[j][1] for j in batch_idx]

        max_len = max(e.shape[0] for e in batch_embs)

        padded_embs = torch.zeros(len(batch_idx), max_len, EMBED_DIM)
        padded_lbls = torch.zeros(len(batch_idx), max_len - 1)

        for k, (emb, lbl) in enumerate(zip(batch_embs, batch_lbls)):
            padded_embs[k, :emb.shape[0]] = emb
            padded_lbls[k, :lbl.shape[0]] = lbl

        optimizer.zero_grad()
        preds = model(padded_embs)  # (batch, max_len-1)

        # loss only on real (non-padded) positions
        loss = sum(
            loss_fn(preds[k, :lbl.shape[0]], lbl)
            for k, lbl in enumerate(batch_lbls)
        ) / len(batch_lbls)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        avg = total_loss / max(len(dataset) // BATCH_SIZE, 1)
        print(f"  Epoch {epoch:02d}  loss={avg:.4f}")


torch.save(model.state_dict(), MODEL_DIR / "bilstm.pth")

meta = {
    "embed_model":          EMBED_MODEL_NAME,
    "embed_dim":            EMBED_DIM,
    "hidden_dim":           HIDDEN_DIM,
    "window_size":          WINDOW_SIZE,
    "boundary_percentile":  BOUNDARY_PERCENTILE,
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nTraining complete.")
print("Saved → models/bilstm.pth")
print("Saved → models/model_meta.json")
