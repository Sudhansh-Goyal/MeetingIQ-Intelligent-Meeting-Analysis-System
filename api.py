# from fastapi import FastAPI
# from schemas import ProcessRequest, ProcessResponse

# import torch
# import pickle

# # import your functions
# from topic_segmentation import segment, load_model, load_tokenizer
# from todo_extraction import generator, build_prompt

# app = FastAPI()

# # ================== LOAD MODEL ==================
# device = torch.device("cpu")

# tokenizer = load_tokenizer()
# model = load_model(len(tokenizer.word2idx))

# # ================== TODO USING LLM ==================
# def extract_tasks_llm(transcript_text):

#     prompt = build_prompt(transcript_text[:2000])

#     result = generator(
#         prompt,
#         max_length=512,
#         do_sample=False
#     )

#     output = result[0]["generated_text"]

#     try:
#         import json
#         return json.loads(output)
#     except:
#         return []

# # ================== ROUTE ==================
# @app.post("/process", response_model=ProcessResponse)
# def process(data: ProcessRequest):

#     # Convert input → internal format
#     utterances = []
#     for item in data.transcript:
#         utterances.append({
#             "start": 0,
#             "end": 0,
#             "speaker": item.speaker,
#             "text": item.text
#         })

#     # ================= TOPIC SEGMENTATION =================
#     segments_raw = segment(utterances, model, tokenizer)

#     topics = []
#     for i, seg in enumerate(segments_raw):
#         topics.append({
#             "segment_id": i,
#             "utterances": [u["text"] for u in seg]
#         })

#     # ================= TODO EXTRACTION =================
#     full_text = "\n".join(
#         [f"{u['speaker']}: {u['text']}" for u in utterances]
#     )

#     todos = extract_tasks_llm(full_text)

#     return {
#         "topics": topics,
#         "todos": todos
#     }

from fastapi import FastAPI
from schemas import ProcessRequest, ProcessResponse

import json
import re
from contextlib import asynccontextmanager

# ===== IMPORT YOUR MODULES =====
from topic_segmentation import segment, load_model, load_tokenizer
from todo_extraction import call_grok, build_prompt, GROK_API_KEY_1, GROK_API_KEY_2

# ===== GLOBAL VARIABLES =====
tokenizer = None
model = None

# ===== LIFESPAN (NEW METHOD) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    print("Loading models...")

    tokenizer = load_tokenizer()
    model = load_model(len(tokenizer.word2idx))

    print("Models loaded successfully")

    yield  # app runs here

    print("Shutting down...")

# ===== APP =====
app = FastAPI(title="Meeting AI", lifespan=lifespan)

# ===== SAFE JSON PARSER =====
def safe_json_parse(output):
    try:
        return json.loads(output)
    except:
        match = re.search(r"\[.*\]", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []

# ===== TODO EXTRACTION =====
def extract_tasks_llm(text):

    prompt = build_prompt(text[:4000])

    try:
        print("Trying API 1...")
        output = call_grok(GROK_API_KEY_1, prompt)
    except Exception as e:
        print("Primary failed:", e)
        try:
            print("Trying API 2...")
            output = call_grok(GROK_API_KEY_2, prompt)
        except Exception as e:
            print("Fallback failed:", e)
            return []

    return safe_json_parse(output)

# ===== MAIN ROUTE =====
@app.post("/process", response_model=ProcessResponse)
def process(data: ProcessRequest):

    if tokenizer is None or model is None:
        return {"error": "Model not loaded"}

    # ===== PREPARE INPUT =====
    utterances = []
    for item in data.transcript:
        utterances.append({
            "start": 0,
            "end": 0,
            "speaker": item.speaker,
            "text": item.text
        })

    # ===== TOPIC SEGMENTATION =====
    segments_raw = segment(utterances, model, tokenizer)

    topics = []
    for i, seg in enumerate(segments_raw):
        topics.append({
            "segment_id": i,
            "utterances": seg
        })

    # ===== TODO EXTRACTION =====
    full_text = "\n".join(
        [f"{u['speaker']}: {u['text']}" for u in utterances]
    )

    todos = extract_tasks_llm(full_text)

    return {
        "topics": topics,
        "todos": todos
    }