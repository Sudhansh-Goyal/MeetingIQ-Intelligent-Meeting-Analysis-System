# 🧠 Meeting Intelligence System using Deep Learning

An end-to-end **AI-powered meeting analysis system** that processes raw meeting audio and generates:

- 📌 Topic-segmented transcripts  
- ✅ Actionable to-do items (speaker-attributed)

Built using **Deep Learning, NLP, and Large Language Models (LLMs)** and evaluated on the **AMI Meeting Corpus**.

---

## 🚀 Features

- 🎙️ Speech-to-Text using Whisper  
- 🧑‍🤝‍🧑 Speaker Diarization using Pyannote  
- 🧩 Topic Segmentation using TextTiling + BiLSTM  
- 📋 Action Item Extraction using spaCy + FLAN-T5  
- ⚡ FastAPI backend for real-time inference  
- 📊 Evaluation using WER, DER, ROUGE  

---

## 🏗️ System Architecture
Audio → Whisper → Pyannote → Transcript
→ Sentence Embeddings
→ TextTiling + BiLSTM
→ Topic Segments
→ NLP + FLAN-T5
→ To-Do Extraction

<img width="1273" height="653" alt="image" src="https://github.com/user-attachments/assets/d6804c04-923f-4031-a652-07e2593cf748" />

---

## 📂 Dataset

- **AMI Meeting Corpus**
- ~100 hours of multi-speaker meeting recordings  
- Real-world product design discussions  
- 4 speakers per session  

---

## 🧠 Models Used

| Component        | Model                          |
|----------------|-------------------------------|
| ASR            | Whisper (medium)              |
| Diarization    | Pyannote 3.1                 |
| Embeddings     | MiniLM-L3-v2                 |
| Segmentation   | TextTiling + BiLSTM          |
| Task Extraction| FLAN-T5                      |
| NLP            | spaCy                        |

---

## Pipeline
Transcription → Train model → Topic segmentation → To-do extraction → Run API uvicorn api

---
## ⚙️ Installation

```bash
git clone https://github.com/Sudhansh-Goyal/Meeting_ai.git
cd Meeting_ai

pip install -r requirements.txt

python transcription.py

python train_bilstm.py

python topic_segmentation.py

extraction python todo_extraction.py

app --host 0.0.0.0 --port 8000
```

---

## 📈 Results
Topic segmentation outputs
<img width="1458" height="978" alt="Screenshot 2026-04-03 213652" src="https://github.com/user-attachments/assets/404b85cc-b6f8-4795-917f-d7989cb6dc8a" />

To-do extraction outputs
<img width="904" height="556" alt="image" src="https://github.com/user-attachments/assets/473e91d5-2b56-42a3-af5f-2ce0e2389f45" />

---

## 🧪 Key Observations
- Hybrid models outperform standalone approaches
- LLM significantly improves task clarity
- Weak supervision limits segmentation performance

---

## 🔮 Future Enhancements
Transformer-based segmentation (BERT / Longformer)
Fine-tuning on AMI annotated boundaries
Real-time meeting processing
Multi-language support
Improved coreference resolution
Task deduplication and ranking

---

## 🛠️ Tech Stack
- PyTorch
- HuggingFace Transformers
- SentenceTransformers
- spaCy
- FastAPI
- Whisper
- Pyannote

## 📚 References
- AMI Meeting Corpus
- OpenAI Whisper
- Pyannote Audio
- Sentence-BERT
- TextTiling
- FLAN-T5

