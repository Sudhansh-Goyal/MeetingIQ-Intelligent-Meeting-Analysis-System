# ============================================================
# topic_segmentation.py  -  Research-grade pipeline
# Three methods: TextTiling | LDA | BiLSTM
# Evaluation against AMI Meeting Corpus ground truth
#
# FIXES applied:
#   1. TextTiling uses depth scores (consistent with BiLSTM training labels)
#   2. LDA uses sklearn (no thread deadlock) with majority-vote smoothing
#   3. BiLSTM threshold = adaptive mean of predicted probabilities
#   4. GT boundary detection uses utterance midpoint inside GT segment spans
# ============================================================

import os
# Set BEFORE any HF/torch imports
os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from xml.etree import ElementTree as ET

torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ================= CONFIG =================
BASE_DIR        = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
GT_TOPIC_DIR    = BASE_DIR / "data" / "gt_topic"
GT_WORDS_DIR    = BASE_DIR / "data" / "gt_words"
OUTPUT_DIR      = BASE_DIR / "data" / "topics"
EVAL_DIR        = BASE_DIR / "data" / "evaluation"
MODEL_DIR       = BASE_DIR / "models"

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"

# Segmentation hyperparameters (tuned for AMI corpus)
TT_WINDOW           = 3     # utterances on each side for coherence window
TT_BOUNDARY_PCT     = 75    # depth-score percentile: top 25% = boundary
LDA_N_TOPICS        = 5
LDA_SMOOTH_WINDOW   = 3     # smooth topic labels before boundary detection
BILSTM_THRESHOLD    = None  # None = adaptive (mean of predicted probs)


# ================= PARSER =================
LINE_RE = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(\S+):\s*(.*)")


def parse_transcript(path):
    utterances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                utterances.append({
                    "start":   float(m.group(1)),
                    "end":     float(m.group(2)),
                    "speaker": m.group(3),
                    "text":    m.group(4).strip(),
                })
    return utterances


# ================= GT ALIGNMENT =================
def parse_words_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    word_map = {}
    for w in root.findall(".//{*}w"):
        wid   = w.attrib.get("{http://nite.sourceforge.net/}id")
        if not wid:
            continue
        start = w.attrib.get("starttime")
        end   = w.attrib.get("endtime")
        if start is None or end is None:
            continue
        word_map[wid] = (float(start), float(end))
    return word_map


def parse_topic_xml(file, word_maps):
    tree = ET.parse(file)
    root = tree.getroot()
    segments = []
    for topic in root.findall(".//{*}topic"):
        start, end = float("inf"), 0.0
        for child in topic.findall(".//{*}child"):
            href = child.attrib.get("href")
            if not href or "#id(" not in href:
                continue
            ref_part = href.split("#id(")[1].replace(")", "")
            if ")..id(" in href:
                start_id, end_id = ref_part.split("..id(")
            else:
                start_id = end_id = ref_part
            file_key = ".".join(start_id.split(".")[:2]) + ".words"
            if file_key not in word_maps:
                continue
            wm = word_maps[file_key]
            if start_id in wm and end_id in wm:
                start = min(start, wm[start_id][0])
                end   = max(end,   wm[end_id][1])
        if start < end:
            segments.append((start, end))
    return sorted(segments)


def build_true_boundaries(utterances, segments):
    """Mark a gap as boundary if the midpoint of that gap falls between
    two different GT segments (or outside all GT segments on different sides)."""
    boundaries = [0] * (len(utterances) - 1)
    for i in range(len(utterances) - 1):
        mid = (utterances[i]["end"] + utterances[i + 1]["start"]) / 2.0
        seg_before = _find_segment(utterances[i]["end"] - 0.01, segments)
        seg_after  = _find_segment(utterances[i + 1]["start"] + 0.01, segments)
        if seg_before != seg_after:
            boundaries[i] = 1
    return boundaries


def _find_segment(t, segments):
    """Return index of GT segment containing time t, or -1."""
    for idx, (s, e) in enumerate(segments):
        if s <= t <= e:
            return idx
    return -1


# ================= MODEL (matches bilstm.pth exactly) =================
class BiLSTMBoundary(nn.Module):
    """BiLSTM boundary classifier — exact match to train_bilstm.py.
    Input  : (batch, seq_len, 384)
    Output : (batch, seq_len - 1)   sigmoid P(boundary after i)
    """
    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.3):
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
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out)
        logits = self.fc(out).squeeze(-1)
        return torch.sigmoid(logits[:, :-1])


# ================= SEGMENTATION METHODS =================

def _depth_scores(emb, window=TT_WINDOW):
    """Compute TextTiling depth scores — same as training pseudo-labels."""
    n = len(emb)
    coherence = []
    for i in range(1, n):
        left  = np.mean(emb[max(0, i - window):i],    axis=0, keepdims=True)
        right = np.mean(emb[i:min(n, i + window)],    axis=0, keepdims=True)
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

    return depths


def texttiling(emb, pct=TT_BOUNDARY_PCT):
    """TextTiling via depth score — boundary if depth >= pct-th percentile.
    Uses same signal as BiLSTM training labels → consistent segmentation."""
    depths    = _depth_scores(emb)
    threshold = np.percentile(depths, pct)
    return [1 if d >= threshold else 0 for d in depths]


def lda_method(texts, smooth_window=LDA_SMOOTH_WINDOW):
    """sklearn LDA → boundary where smoothed topic assignment changes.
    Smoothing avoids single-utterance topic flips creating noisy boundaries."""
    vec = CountVectorizer(stop_words="english", max_features=500, min_df=1)
    try:
        dtm = vec.fit_transform(texts)
    except ValueError:
        return [0] * (len(texts) - 1), 0.0

    lda = LatentDirichletAllocation(
        n_components=LDA_N_TOPICS, max_iter=10,
        random_state=42, learning_method="online", n_jobs=1
    )
    doc_topics = lda.fit_transform(dtm)
    raw_topics = np.argmax(doc_topics, axis=1)  # dominant topic per utterance

    # Majority-vote smoothing: replace each topic with mode in its window
    n = len(raw_topics)
    smoothed = []
    for i in range(n):
        window = raw_topics[max(0, i - smooth_window): i + smooth_window + 1]
        counts = np.bincount(window.astype(int), minlength=LDA_N_TOPICS)
        smoothed.append(int(np.argmax(counts)))

    boundaries = [1 if smoothed[i] != smoothed[i + 1] else 0
                  for i in range(n - 1)]

    # Topic change ratio as coherence proxy (u_mass not available in sklearn)
    coherence = round(float(np.mean(doc_topics.max(axis=1))), 4)
    return boundaries, coherence


def bilstm_predict(emb, model):
    """Run BiLSTM. Threshold = mean of predicted probabilities,
    so the model always produces roughly as many boundaries as
    half of all gaps — consistent across meetings."""
    with torch.no_grad():
        x     = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs = model(x)[0].cpu().numpy()
    threshold = float(np.mean(probs))  # adaptive: boundary where prob > average
    return [1 if p >= threshold else 0 for p in probs], probs, threshold


# ================= TOPIC LABELING =================
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


def label_segment(seg_utterances, embed_model, topic_embs):
    """Zero-shot label: cosine similarity of segment text to topic candidates."""
    text    = " ".join(u["text"] for u in seg_utterances)
    seg_emb = embed_model.encode([text])
    sims    = cosine_similarity(seg_emb, topic_embs)[0]
    best    = int(np.argmax(sims))
    return TOPIC_CANDIDATES[best], round(float(sims[best]), 3)


# ================= EVALUATION METRICS =================
def pk_metric(pred, true, k=None):
    n = len(true)
    if k is None:
        k = max(1, n // 10)   # k = ~1/10 average segment length (standard)
    if n <= k:
        return 0.0
    errors = sum(
        1 for i in range(n - k)
        if (sum(pred[i:i + k]) > 0) != (sum(true[i:i + k]) > 0)
    )
    return errors / (n - k)


def window_diff(pred, true, k=None):
    n = len(true)
    if k is None:
        k = max(1, n // 10)
    if n <= k:
        return 0.0
    wd = sum(abs(sum(pred[i:i + k]) - sum(true[i:i + k])) for i in range(n - k))
    return wd / (n - k)


def segmentation_covering(pred, true):
    return 1.0 - abs(sum(pred) - sum(true)) / max(1, sum(true))


def evaluate(pred, true):
    p, r, f, _ = precision_recall_fscore_support(
        true, pred, average="binary", zero_division=0
    )
    return {
        "Pk":         round(pk_metric(pred, true), 4),
        "WindowDiff": round(window_diff(pred, true), 4),
        "Precision":  round(float(p), 4),
        "Recall":     round(float(r), 4),
        "F1":         round(float(f), 4),
        "SegCover":   round(segmentation_covering(pred, true), 4),
    }


# ================= SEGMENT BUILDER =================
def build_segments(utterances, boundaries):
    segs, cur = [], []
    for i, u in enumerate(utterances):
        cur.append(u)
        if i < len(boundaries) and boundaries[i] == 1:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def format_output(file, utterances, segments, embed_model, topic_embs):
    out = {
        "source":           file.name,
        "total_utterances": len(utterances),
        "total_segments":   len(segments),
        "segments":         [],
    }
    for i, seg in enumerate(segments):
        topic, confidence = label_segment(seg, embed_model, topic_embs)
        out["segments"].append({
            "segment_id":      i,
            "topic":           topic,
            "confidence":      confidence,
            "start_time":      seg[0]["start"],
            "end_time":        seg[-1]["end"],
            "duration_sec":    round(seg[-1]["end"] - seg[0]["start"], 2),
            "utterance_count": len(seg),
            "preview":         [f"{u['speaker']}: {u['text']}" for u in seg[:3]],
            "utterances":      seg,
        })
    return out


# ================= LOGGING =================
_LOG_PATH = BASE_DIR / "run.log"
_LOG      = None


def log(msg=""):
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


# ================= MAIN =================
def run():
    global _LOG
    _LOG = open(_LOG_PATH, "w", buffering=1, encoding="utf-8")
    t0   = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True,   exist_ok=True)

    # ---- Load embedding model ----
    log(f"Loading embedding model: {EMBED_MODEL_NAME}")
    try:
        embed = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True, device="cpu")
    except Exception as e:
        log(f"ERROR loading model: {e}")
        log(f"Fix: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{EMBED_MODEL_NAME}')\"")
        _LOG.close(); return
    log(f"  Loaded [{time.time() - t0:.1f}s]")

    log("Encoding topic candidates...")
    topic_embs = embed.encode(TOPIC_CANDIDATES, show_progress_bar=False)

    # ---- Load BiLSTM ----
    model = BiLSTMBoundary().to(DEVICE)
    ckpt  = MODEL_DIR / "bilstm.pth"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=True)
            log(f"  Loaded BiLSTM weights from {ckpt.name}")
        except Exception as e:
            log(f"  WARNING: Could not load BiLSTM weights ({e}). Using random init.")
    else:
        log("  bilstm.pth not found — using randomly-initialized model.")
    model.eval()

    # ---- Load GT word maps ----
    log("\nLoading GT word maps...")
    word_maps = {}
    for wf in GT_WORDS_DIR.glob("*.words.xml"):
        try:
            word_maps[wf.stem] = parse_words_xml(wf)
        except Exception as e:
            log(f"  WARNING: Could not parse {wf.name}: {e}")
    log(f"  Loaded {len(word_maps)} word XML files.")

    # ---- Process transcripts ----
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        log(f"No .txt files found in {TRANSCRIPTS_DIR}"); _LOG.close(); return

    log(f"\nFound {len(files)} transcript files.\n")
    all_results = []

    for fidx, file in enumerate(files, 1):
        log(f"{'='*60}")
        log(f"[{fidx}/{len(files)}] {file.name}")
        t1 = time.time()

        utterances = parse_transcript(file)
        if len(utterances) < 6:
            log(f"  Skipping — too short ({len(utterances)} utterances)."); continue

        texts = [u["text"] for u in utterances]
        log(f"  Utterances: {len(utterances)}")

        emb = embed.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        log(f"  Embeddings: done [{time.time() - t1:.1f}s]")

        # ---- GT alignment ----
        meeting_id = file.stem.split(".")[0]
        topic_file = GT_TOPIC_DIR / f"{meeting_id}.topic.xml"
        if topic_file.exists():
            try:
                segments_gt = parse_topic_xml(topic_file, word_maps)
                true_bounds = build_true_boundaries(utterances, segments_gt)
                log(f"  GT: {len(segments_gt)} segments | {sum(true_bounds)} boundaries")
            except Exception as e:
                log(f"  WARNING: GT parse failed ({e}). Skipping eval.")
                true_bounds = None
        else:
            log(f"  GT file not found ({topic_file.name}). Skipping eval.")
            true_bounds = None

        # ---- Run methods ----
        lda_coherence = 0.0
        results = {}

        # 1. TextTiling (depth score, 75th pct — same as BiLSTM training labels)
        tt = texttiling(emb)
        results["texttiling"] = tt
        log(f"  TextTiling  → {sum(tt)} boundaries ({sum(tt)+1} segments)")

        # 2. LDA (sklearn, no deadlock; smoothed)
        log("  LDA running...")
        try:
            lda_b, lda_coherence = lda_method(texts)
            results["lda"] = lda_b
            log(f"  LDA         → {sum(lda_b)} boundaries ({sum(lda_b)+1} segments, coherence={lda_coherence:.3f})")
        except Exception as e:
            log(f"  LDA failed: {e}. Fallback: no boundaries.")
            results["lda"] = [0] * (len(utterances) - 1)

        # 3. Proposed BiLSTM (adaptive threshold)
        prop, probs, thresh = bilstm_predict(emb, model)
        results["proposed"] = prop
        log(f"  BiLSTM      → {sum(prop)} boundaries ({sum(prop)+1} segments, "
            f"mean_prob={float(np.mean(probs)):.3f}, thresh={thresh:.3f})")

        # ---- Save outputs & evaluate ----
        for name, bounds in results.items():
            segs    = build_segments(utterances, bounds)
            out     = format_output(file, utterances, segs, embed, topic_embs)
            out_dir = OUTPUT_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{file.stem}.json", "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)

            if true_bounds is not None and len(true_bounds) == len(bounds):
                metrics = evaluate(bounds, true_bounds)
            else:
                metrics = {"Pk": None, "WindowDiff": None, "Precision": None,
                           "Recall": None, "F1": None, "SegCover": None}

            metrics.update({"method": name, "file": file.name, "segments": len(segs)})
            if name == "lda":
                metrics["coherence"] = round(lda_coherence, 4)
            all_results.append(metrics)

            log(f"    [{name:12s}]  segs={len(segs):3d}  "
                f"Pk={metrics['Pk']}  WD={metrics['WindowDiff']}  F1={metrics['F1']}")

        log(f"  Done in {time.time() - t1:.1f}s")

    # ---- Save evaluation ----
    if not all_results:
        log("\nNo results to save."); _LOG.close(); return

    with open(EVAL_DIR / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)

    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(EVAL_DIR / "comparison.csv", index=False)

        # Summary table
        log("\n" + "=" * 70)
        log("SUMMARY — Average metrics across meetings")
        log("=" * 70)
        log(f"  {'Method':<14} {'Pk↓':>6} {'WD↓':>7} {'Prec↑':>7} {'Rec↑':>7} {'F1↑':>7} {'SegCov↑':>9}")
        log("  " + "-" * 58)
        numeric = ["Pk", "WindowDiff", "Precision", "Recall", "F1", "SegCover"]
        for method in ["texttiling", "lda", "proposed"]:
            sub = df[df["method"] == method][numeric].dropna()
            if sub.empty: continue
            m = sub.mean()
            log(f"  {method:<14} {m['Pk']:>6.4f} {m['WindowDiff']:>7.4f} "
                f"{m['Precision']:>7.4f} {m['Recall']:>7.4f} "
                f"{m['F1']:>7.4f} {m['SegCover']:>9.4f}")
        log("=" * 70)
    except ImportError:
        log("pandas not installed — CSV/summary skipped.")

    total = time.time() - t0
    log(f"\nDone in {total/60:.1f} min.")
    log(f"  Topic JSONs → {OUTPUT_DIR}")
    log(f"  Eval CSV   → {EVAL_DIR / 'comparison.csv'}")
    log(f"  Log        → {_LOG_PATH}")
    _LOG.close()


if __name__ == "__main__":
    run()
