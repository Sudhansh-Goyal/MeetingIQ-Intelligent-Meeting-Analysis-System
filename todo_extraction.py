# import json
# from pathlib import Path
# from transformers import pipeline

# BASE_DIR = Path(__file__).parent
# TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
# OUTPUT_DIR = BASE_DIR / "data" / "todos"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # ================== LOAD MODEL ==================
# print("Loading local LLM (Flan-T5)...")

# generator = pipeline(
#     "text-generation",
#     model="google/flan-t5-base",   # good balance (can upgrade to large)
#     device=-1  # CPU (use 0 if GPU available)
# )

# # ================== PROMPT ==================
# def build_prompt(text):
#     return f"""
# Extract action items from the meeting transcript.

# For each task:
# - Identify the speaker
# - Extract the task clearly

# Return ONLY JSON like:
# [
#   {{"speaker": "SPEAKER_1", "task": "Do something"}}
# ]

# Transcript:
# {text}
# """

# # ================== EXTRACT ==================
# def extract_tasks(file_path):
#     with open(file_path, encoding="utf-8") as f:
#         transcript = f.read()

#     prompt = build_prompt(transcript[:2000])  # limit size for stability

#     result = generator(
#         prompt,
#         max_length=512,
#         do_sample=False
#     )

#     output = result[0]["generated_text"]

#     try:
#         tasks = json.loads(output)
#     except:
#         print("Could not parse JSON. Raw output:")
#         print(output)
#         tasks = []

#     return tasks

# # ================== MAIN ==================
# def main():
#     files = list(TRANSCRIPTS_DIR.glob("*.txt"))

#     if not files:
#         print("No transcripts found")
#         return

#     for file in files:
#         print(f"Processing: {file.name}")

#         tasks = extract_tasks(file)

#         out_path = OUTPUT_DIR / f"{file.stem}_todos.json"

#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(tasks, f, indent=2)

#         print(f"Saved: {out_path}")

#     print("To-do extraction complete")

# if __name__ == "__main__":
#     main()


import json
import requests
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

GROK_API_KEY_1 = os.getenv("GROQ_KEY_1")
GROK_API_KEY_2 = os.getenv("GROQ_KEY_2")

BASE_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
OUTPUT_DIR = BASE_DIR / "data" / "todos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== CONFIG ==================
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"



MODEL_NAME = "llama-3.3-70b-versatile"   # change if needed

# ================== PROMPT ==================
def build_prompt(text):
    return f"""
Extract action items from the meeting transcript.

For each task:
- Identify the speaker
- Extract the task clearly

Return ONLY JSON like:
[
  {{"speaker": "SPEAKER_1", "task": "Do something"}}
]

Transcript:
{text}
"""

# ================== API CALL ==================
def call_grok(api_key, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract tasks from transcripts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(GROK_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]

# ================== EXTRACT ==================
def extract_tasks(file_path):
    with open(file_path, encoding="utf-8") as f:
        transcript = f.read()

    prompt = build_prompt(transcript[:4000])

    # Try primary API
    try:
        print("Trying Grok API 1...")
        output = call_grok(GROK_API_KEY_1, prompt)
    except Exception as e:
        print("Primary failed:", e)

        # Fallback API
        try:
            print("Trying Grok API 2...")
            output = call_grok(GROK_API_KEY_2, prompt)
        except Exception as e:
            print("Fallback failed:", e)
            return []

    # Parse JSON safely
    try:
        tasks = json.loads(output)
    except:
        print("JSON parse failed. Raw output:")
        print(output)
        tasks = []

    return tasks

# ================== MAIN ==================
def main():
    files = list(TRANSCRIPTS_DIR.glob("*.txt"))

    if not files:
        print("No transcripts found")
        return

    for file in files:
        print(f"\nProcessing: {file.name}")

        tasks = extract_tasks(file)

        out_path = OUTPUT_DIR / f"{file.stem}_todos.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)

        print(f"Saved: {out_path}")

    print("\nTo-do extraction complete")

# ================== RUN ==================
if __name__ == "__main__":
    main()