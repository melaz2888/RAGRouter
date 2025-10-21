from typing import List, Literal, Optional
from pydantic import BaseModel

Route = Literal["direct", "rag"]

class Passage(BaseModel):
    score: float
    text: str
    source: str

class RagAnswer(BaseModel):
    answer: str
    passages: List[Passage]
    timing_ms: int

class DirectAnswer(BaseModel):
    answer: str
    timing_ms: int

class QAItem(BaseModel):
    id: str
    question: str
    answer: str
    domain: str
