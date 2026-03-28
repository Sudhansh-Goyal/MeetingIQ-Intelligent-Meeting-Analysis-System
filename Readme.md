# Meeting AI System

## Pipeline

1. Transcription
python transcription.py

2. Train model
python train_bilstm.py

3. Topic segmentation
python topic_segmentation.py

4. To-do extraction
python todo_extraction.py

5. Run API
uvicorn api:app --host 0.0.0.0 --port 8000

---

## Folders

data/audio → input  
data/transcripts → transcripts     
data/topics → segmented topics  
data/todos → extracted tasks  

---

## Docker

docker build -t meeting-ai .
docker run -p 8000:8000 meeting-ai