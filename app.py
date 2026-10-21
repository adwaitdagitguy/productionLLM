import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import support_rag
from support_rag import LLMBackendError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # keep provider URLs/noise out of logs
log = logging.getLogger("support_api")

MAX_QUESTION_CHARS = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    support_rag.init()          # embeddings + FAISS + Groq client: loaded once, not per request
    yield


app = FastAPI(title="Customer Support RAG API", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS, examples=["How can I track my order?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class AskResponse(BaseModel):
    answer: str
    fallback_used: bool
    intent: str
    sources: list[str]
    latency_ms: float


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # custom handler so the rejected input is never echoed back or logged
    problems = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": e["msg"]}
                for e in exc.errors()]
    log.warning("ask rejected status=422 path=%s problems=%d", request.url.path, len(problems))
    return JSONResponse(status_code=422, content={"detail": problems})


@app.get("/health")
def health():
    # Liveness only: says the process is up and the KB is loaded. It does NOT check that Groq is reachable.
    return {"status": "ok", "kb_rows": support_rag.kb_size()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # plain `def` -> FastAPI runs it in a threadpool, so the blocking Groq calls don't stall the event loop
    t0 = time.perf_counter()
    q_len = len(req.question)          # log length only, not the question text
    try:
        result = support_rag.answer_question(req.question)
    except LLMBackendError as e:
        ms = (time.perf_counter() - t0) * 1000
        log.error("ask failed status=503 reason=%s q_len=%d latency_ms=%.0f", e, q_len, ms)
        raise HTTPException(status_code=503, detail="The answer service is temporarily unavailable. Please try again later.")
    except Exception:
        ms = (time.perf_counter() - t0) * 1000
        log.exception("ask failed status=500 q_len=%d latency_ms=%.0f", q_len, ms)   # traceback stays in server log
        raise HTTPException(status_code=500, detail="Internal server error.")

    ms = (time.perf_counter() - t0) * 1000
    log.info("ask ok status=200 fallback=%s intent=%s q_len=%d latency_ms=%.0f",
             result["fallback_used"], result["intent"], q_len, ms)
    return AskResponse(**result, latency_ms=round(ms, 1))
