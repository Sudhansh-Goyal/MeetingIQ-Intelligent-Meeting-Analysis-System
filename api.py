from fastapi import FastAPI
from schemas import ProcessRequest, ProcessResponse

import re
from contextlib import asynccontextmanager

import spacy
import torch
from spacy.matcher import Matcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from topic_segmentation import (
    load_bilstm,
    texttiling_boundaries,
    bilstm_refine_boundaries,
    build_segments,
    merge_small_segments,
    EMBED_MODEL_NAME
)



embed_model = None
topic_model = None
nlp = None
matcher = None
tokenizer = None
todo_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


WEAK_WORDS = {"think", "guess", "maybe", "wonder", "probably", "if", "depends", "say", "would", "could"}
IGNORE_VERBS = {"go", "see", "think", "talk", "say", "know", "mean", "do", "look", "be", "get", "come", "have", "want"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, topic_model, nlp, matcher, tokenizer, todo_model

    print("Loading models...")

    # Topic segmentation models
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    topic_model = load_bilstm()

    # spaCy
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    matcher.add("ACTION_PATTERNS", [
        [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["need", "have"]}}, {"LOWER": "to"}, {"POS": "VERB"}],
        [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["should", "must", "will"]}}, {"POS": "VERB"}],
        [{"LOWER": {"IN": ["we", "i"]}}, {"LOWER": {"IN": ["'ll", "'d"]}}, {"POS": "VERB"}],
        [{"LOWER": "let"}, {"LOWER": "'s"}, {"POS": "VERB"}],
        [{"LOWER": "i"}, {"LOWER": {"IN": ["am", "'m"]}}, {"LOWER": "going"}, {"LOWER": "to"}, {"POS": "VERB"}]
    ])

    # Local FLAN-T5
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    todo_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

    print("All models loaded successfully")
    yield
    print("Shutting down...")


app = FastAPI(title="Meeting AI", lifespan=lifespan)



def clean_text(text: str):
    fillers = ["okay", "alright", "so", "well", "um", "uh"]
    for filler in fillers:
        text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def is_valid_sentence(text: str):
    return len(text.split()) > 4 and not text.endswith("?")


def pre_filter_task(text: str):
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


def rewrite_task_with_llm(task_text: str, context_block: str):
    prompt = f"""
Rewrite the raw text into a short actionable to-do item.
Replace unclear references like 'it' using context.

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

    outputs = todo_model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.0,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace("Task:", "").strip().capitalize()


def extract_tasks_local(utterances):
    todos = []
    conversation_history = []

    for u in utterances:
        text = clean_text(u["text"])
        conversation_history.append(text)

        if len(conversation_history) > 3:
            conversation_history.pop(0)

        if is_valid_sentence(text) and pre_filter_task(text):
            context_block = " ".join(conversation_history[:-1])

            task = rewrite_task_with_llm(text, context_block)

            todos.append({
                "speaker": u["speaker"],
                "task": task,
                "source_text": text
            })

    return todos



@app.post("/process", response_model=ProcessResponse)
def process(data: ProcessRequest):

    if embed_model is None:
        return {"error": "Models not loaded"}

    # Build utterances
    utterances = []
    for item in data.transcript:
        utterances.append({
            "start": 0,
            "end": 0,
            "speaker": item.speaker,
            "text": item.text
        })

   
    texts = [u["text"] for u in utterances]
    embeddings = embed_model.encode(texts, show_progress_bar=False)

    boundaries = texttiling_boundaries(embeddings)
    boundaries = bilstm_refine_boundaries(embeddings, topic_model, boundaries)

    segments_raw = build_segments(utterances, boundaries)
    segments_raw = merge_small_segments(segments_raw)

    topics = []
    for i, seg in enumerate(segments_raw):
        topics.append({
            "segment_id": i,
            "utterances": seg
        })

   
    todos = extract_tasks_local(utterances)

    return {
        "topics": topics,
        "todos": todos
    }