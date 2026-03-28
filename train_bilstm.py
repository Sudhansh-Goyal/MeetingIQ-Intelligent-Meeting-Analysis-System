# import re
# import pickle
# import torch
# import torch.nn as nn
# from pathlib import Path

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# BASE_DIR = Path(__file__).parent
# TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
# MODEL_DIR = BASE_DIR / "models"

# MODEL_DIR.mkdir(exist_ok=True)

# MAX_LEN = 50
# SIM_THRESHOLD = 0.7

# LINE_RE = re.compile(
#     r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
# )

# # ================= TOKENIZER =================
# class Tokenizer:
#     def __init__(self):
#         self.word2idx = {"<PAD>": 0, "<UNK>": 1}

#     def fit(self, texts):
#         idx = 2
#         for text in texts:
#             for word in text.lower().split():
#                 if word not in self.word2idx:
#                     self.word2idx[word] = idx
#                     idx += 1

#     def encode(self, text):
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

# # ================= LOAD TEXT =================
# texts = []

# for file in TRANSCRIPTS_DIR.glob("*.txt"):
#     with open(file, encoding="utf-8") as f:
#         for line in f:
#             m = LINE_RE.match(line.strip())
#             if m:
#                 texts.append(m.group(4))

# print(f"Loaded {len(texts)} sentences")

# # ================= TOKENIZER =================
# tokenizer = Tokenizer()
# tokenizer.fit(texts)

# with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)

# # ================= DRIFT LABELING =================
# print("Generating drift-based labels...")

# embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
# embeddings = embed_model.encode(texts)

# labels = [0]  # first sentence no boundary

# for i in range(1, len(texts)):
#     sim = cosine_similarity(
#         [embeddings[i - 1]],
#         [embeddings[i]]
#     )[0][0]

#     # LOW similarity → drift → boundary
#     labels.append(1 if sim < SIM_THRESHOLD else 0)

# print(f"Generated {sum(labels)} boundaries")

# # ================= DATA =================
# X = [tokenizer.encode(t) for t in texts]

# X = torch.tensor(X, dtype=torch.long)
# y = torch.tensor(labels, dtype=torch.float32)

# # ================= TRAIN =================
# model = BiLSTM(len(tokenizer.word2idx))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.BCELoss()

# for epoch in range(5):
#     optimizer.zero_grad()
#     output = model(X).squeeze()
#     loss = loss_fn(output, y)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # ================= SAVE =================
# torch.save(model.state_dict(), MODEL_DIR / "bilstm.pth")

# print("Training complete")
# print("Saved models/bilstm.pth and tokenizer.pkl")


import re
import pickle
import torch
import torch.nn as nn
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MAX_LEN = 50
SIM_THRESHOLD = 0.7

LINE_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)"
)

# ================= TOKENIZER =================
class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        idx = 2
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    idx += 1

    def encode(self, text):
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

# ================= LOAD TEXT =================
texts = []

for file in TRANSCRIPTS_DIR.glob("*.txt"):
    with open(file, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                texts.append(m.group(4))

print(f"Loaded {len(texts)} sentences")

# ================= TOKENIZER =================
tokenizer = Tokenizer()
tokenizer.fit(texts)

with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ================= CONTEXT PAIRS =================
print("Building context pairs...")

paired_texts = []
for i in range(1, len(texts)):
    pair = texts[i - 1] + " " + texts[i]
    paired_texts.append(pair)

# ================= DRIFT LABELING =================
print("Generating drift-based labels...")

embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
embeddings = embed_model.encode(texts)

labels = []

for i in range(1, len(texts)):
    sim = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    labels.append(1 if sim < SIM_THRESHOLD else 0)

print(f"Generated {sum(labels)} boundaries")

# ================= DATA =================
X = [tokenizer.encode(t) for t in paired_texts]

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32)

# ================= TRAIN =================
model = BiLSTM(len(tokenizer.word2idx))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(5):
    optimizer.zero_grad()
    output = model(X).squeeze()
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ================= SAVE =================
torch.save(model.state_dict(), MODEL_DIR / "bilstm.pth")

print("Training complete")
print("Saved models/bilstm.pth and tokenizer.pkl")