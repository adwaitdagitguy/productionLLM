"""RAG pipeline from Experiment 6, with the local HF model swapped for GroqCloud.

Retrieval (FAISS + MiniLM) and the LangGraph workflow are unchanged.
Only llm_generate() talks to the model backend.
Call init() once at startup; then call answer_question() per request.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, GroqError
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()
log = logging.getLogger("support_rag")

# ------------------------- configuration -------------------------
CSV_PATH = os.getenv("CSV_PATH", "csd.csv")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")   # cheapest Groq text model
GROQ_TIMEOUT_S = float(os.getenv("GROQ_TIMEOUT_S", "20"))

CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 450, 60, 3
USE_COSINE_CHECK = True
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))  # re-calibrate for your csd.csv
USE_LLM_JUDGE = True
COMBINE_MODE = "strict"      # "strict" | "lenient"
USE_MARKER_CHECK = True
MAX_RETRIES = 1

# ------------------------- module state (set once in init) -------------------------
_client: Optional[Groq] = None
_vector_store = None
_graph = None
_intent_keywords: Dict[str, List[str]] = {}
_kb_rows = 0


class LLMBackendError(RuntimeError):
    """Model/provider failure (bad key, quota, network, model name...). Message is safe to log."""


# ------------------------- knowledge base -------------------------
def load_faq(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = next((s for s in ("\t", "|", ";", ",") if s in header), None)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"question", "answer", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    df = df[["question", "answer", "category"]].dropna()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df[(df.question != "") & (df.answer != "")].drop_duplicates().reset_index(drop=True)


# ------------------------- LLM backend (the only Groq-specific part) -------------------------
def llm_generate(system: str, user: str, max_tokens: int = 160) -> str:
    try:
        resp = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
    except GroqError as e:
        # log only the exception type + HTTP status: never the key, prompt, or provider message
        status = getattr(e, "status_code", None)
        log.error("Groq call failed: %s status=%s", type(e).__name__, status)
        raise LLMBackendError(f"{type(e).__name__} (status={status})") from e
    return (resp.choices[0].message.content or "").strip()


# ------------------------- graph state + nodes -------------------------
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    retrieved: List[Dict[str, Any]]
    context: str
    top_score: float
    draft: str
    insufficient: bool
    retry_count: int
    final_answer: str
    fallback_used: bool
    trace: List[Dict[str, Any]]


BASE_INTENT_KEYWORDS = {
    "Returns":   ["refund", "return", "exchange", "replace", "send back", "money back"],
    "Shipping":  ["track", "shipment", "shipping", "delivery", "deliver", "dispatch", "package",
                  "parcel", "order status", "where is my order", "arrive"],
    "Account":   ["password", "sign in", "log in", "login", "account", "verification", "otp", "username"],
    "Billing":   ["charged", "payment", "billing", "invoice", "card", "price", "pay"],
    "Technical": ["website", "error", "loading", "browser", "technical", "app crash", "bug", "not working"],
}


def retrieve_scored(query: str, k: int = TOP_K):
    hits = _vector_store.similarity_search_with_score(query, k=k)
    return [(d, float(np.clip(s, -1.0, 1.0))) for d, s in hits]


def classify_intent(state: SupportState):
    q = state["question"].lower()
    intent = "general_support"
    for label, kws in _intent_keywords.items():
        if any(re.search(r"\b" + re.escape(k), q) for k in kws):
            intent = label
            break
    return {"intent": intent, "search_query": state["question"],
            "retry_count": state.get("retry_count", 0), "trace": [], "fallback_used": False}


def retrieve_context(state: SupportState):
    hits = retrieve_scored(state["search_query"], k=TOP_K)
    retrieved = [{"text": d.page_content, "category": d.metadata.get("category", ""), "score": s}
                 for d, s in hits]
    return {"retrieved": retrieved,
            "context": "\n\n".join(r["text"] for r in retrieved),
            "top_score": max((r["score"] for r in retrieved), default=0.0)}


DRAFT_SYSTEM = (
    "You are a polite customer-support assistant. "
    "Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION. "
    "Do not invent order status, refund approvals, timelines, prices or company policies. "
    "If the context does not contain the answer, output exactly: INSUFFICIENT_CONTEXT "
    "Never ask for passwords, one-time codes, or full payment-card details. "
    "Keep the answer clear and concise."
)

JUDGE_SYSTEM = (
    "You are a strict reviewer for a customer-support assistant. "
    "You decide whether the FAQ CONTEXT contains the specific information needed to answer the CUSTOMER QUESTION. "
    "Requests for live data (e.g. real-time order status), private credentials (e.g. passwords), "
    "or topics the FAQ never mentions are NOT answerable from the context."
)


def draft_answer(state: SupportState):
    user = f"FAQ CONTEXT:\n{state.get('context', '')}\n\nCUSTOMER QUESTION:\n{state['question']}\n\nANSWER:"
    return {"draft": llm_generate(DRAFT_SYSTEM, user, max_tokens=160)}


def llm_judge_sufficiency(question: str, context: str) -> bool:
    user = (f"FAQ CONTEXT:\n{context}\n\nCUSTOMER QUESTION:\n{question}\n\n"
            "Does the FAQ CONTEXT contain the specific information needed to answer the CUSTOMER QUESTION? "
            "Reply with exactly one word: YES or NO.")
    return llm_generate(JUDGE_SYSTEM, user, max_tokens=4).upper().startswith("YES")


def review_answer(state: SupportState):
    draft = state.get("draft", "").strip()
    top = state.get("top_score", 0.0)

    marker_hit = USE_MARKER_CHECK and ("INSUFFICIENT_CONTEXT" in draft.upper() or draft == "")
    cosine_ok = (top >= SIMILARITY_THRESHOLD) if USE_COSINE_CHECK else None
    llm_ok = llm_judge_sufficiency(state["question"], state.get("context", "")) if USE_LLM_JUDGE else None

    enabled = [c for c in (cosine_ok, llm_ok) if c is not None]
    if not enabled:
        sufficient = True
    elif COMBINE_MODE == "strict":
        sufficient = all(enabled)
    else:
        sufficient = any(enabled)
    insufficient = bool(marker_hit) or not sufficient

    entry = {"attempt": state.get("retry_count", 0) + 1, "top_score": round(top, 4),
             "cosine_ok": cosine_ok, "llm_ok": llm_ok, "marker_hit": bool(marker_hit),
             "insufficient": insufficient}
    return {"insufficient": insufficient,
            "final_answer": "" if insufficient else draft,
            "trace": state.get("trace", []) + [entry]}


def route_after_review(state: SupportState):
    if state.get("final_answer"):
        return "finish"
    return "retry" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"


def rewrite_query(state: SupportState):
    intent = state.get("intent", "general_support")
    extra = (f"{intent} " + " ".join(_intent_keywords[intent][:4])) if intent in _intent_keywords \
        else "customer support help contact"
    return {"search_query": state["question"] + " " + extra,
            "retry_count": state.get("retry_count", 0) + 1}


def fallback_answer(state: SupportState):
    return {"fallback_used": True,
            "final_answer": ("I couldn't find a reliable answer in the available FAQ. "
                             "Please contact the support team through the Help section. "
                             "Do not share your password, one-time code, or full payment-card details.")}


def _build_graph():
    g = StateGraph(SupportState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve", retrieve_context)
    g.add_node("draft_answer", draft_answer)
    g.add_node("review_answer", review_answer)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("fallback", fallback_answer)
    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "retrieve")
    g.add_edge("retrieve", "draft_answer")
    g.add_edge("draft_answer", "review_answer")
    g.add_conditional_edges("review_answer", route_after_review,
                            {"finish": END, "retry": "rewrite_query", "fallback": "fallback"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("fallback", END)
    return g.compile()


# ------------------------- public API -------------------------
def init() -> None:
    """Load everything expensive exactly once (call from FastAPI startup)."""
    global _client, _vector_store, _graph, _intent_keywords, _kb_rows

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Put it in .env (never in source code).")
    _client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_S, max_retries=1)

    faq_df = load_faq(CSV_PATH)
    _kb_rows = len(faq_df)

    csv_cats = {c.lower(): c for c in faq_df.category.unique()}
    _intent_keywords = {csv_cats.get(k.lower(), k): v for k, v in BASE_INTENT_KEYWORDS.items()}
    for lc, c in csv_cats.items():
        _intent_keywords.setdefault(c, [lc])

    docs = [Document(page_content=f"Question: {r.question}\nAnswer: {r.answer}",
                     metadata={"source": os.path.basename(CSV_PATH), "row": int(i), "category": r.category})
            for i, r in faq_df.iterrows()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    _vector_store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    _graph = _build_graph()
    log.info("RAG ready: %d FAQ rows, %d chunks, model=%s", _kb_rows, len(chunks), GROQ_MODEL)


def kb_size() -> int:
    return _kb_rows


def answer_question(question: str) -> Dict[str, Any]:
    """Run the RAG graph. Raises LLMBackendError if the Groq call fails."""
    res = _graph.invoke({"question": question, "retry_count": 0})
    fb = bool(res.get("fallback_used"))
    return {
        "answer": res["final_answer"],
        "fallback_used": fb,
        "intent": res.get("intent", "general_support"),
        "sources": [] if fb else sorted({r["category"] for r in res.get("retrieved", []) if r["category"]}),
    }
