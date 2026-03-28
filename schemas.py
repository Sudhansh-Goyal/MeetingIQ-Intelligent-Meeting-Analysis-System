from pydantic import BaseModel
from typing import List

class TranscriptItem(BaseModel):
    speaker: str
    text: str

class ProcessRequest(BaseModel):
    transcript: List[TranscriptItem]

class TopicSegment(BaseModel):
    segment_id: int
    utterances: List[dict]

class TodoItem(BaseModel):
    speaker: str
    task: str

class ProcessResponse(BaseModel):
    topics: List[TopicSegment]
    todos: List[TodoItem]