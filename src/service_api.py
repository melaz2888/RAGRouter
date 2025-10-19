from fastapi import FastAPI
from pydantic import BaseModel
from .route_infer import route
from .answer_direct import answer_direct
from .answer_rag import answer_rag

class AskIn(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask")
def ask(inq: AskIn):
    q = inq.question.strip()
    r = route(q)
    if r == "direct":
        out = answer_direct(q)
        passages = []
        timing = out.timing_ms
        ans = out.answer
    else:
        out = answer_rag(q)
        passages = [p.model_dump() for p in out.passages]
        timing = out.timing_ms
        ans = out.answer
    return {"route": r, "answer": ans, "passages": passages, "timing_ms": timing}
