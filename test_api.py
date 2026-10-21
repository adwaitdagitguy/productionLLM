"""Runs the Section 8 checklist against a running server and prints rows for the observation table.
Usage: uvicorn app:app  (in one terminal)   |   python test_api.py  (in another)
"""
import requests

BASE = "http://127.0.0.1:8000"
SUPPORTED = "How can I track my order?"          # edit to match your csd.csv
UNSUPPORTED = "Do you sell home insurance?"

cases = [
    ("Health endpoint",      lambda: requests.get(f"{BASE}/health", timeout=10),
     lambda r: r.status_code == 200 and r.json().get("status") == "ok"),
    ("Supported question",   lambda: requests.post(f"{BASE}/ask", json={"question": SUPPORTED}, timeout=60),
     lambda r: r.status_code == 200 and not r.json()["fallback_used"]),
    ("Unsupported question", lambda: requests.post(f"{BASE}/ask", json={"question": UNSUPPORTED}, timeout=60),
     lambda r: r.status_code == 200 and r.json()["fallback_used"]),
    ("Empty question",       lambda: requests.post(f"{BASE}/ask", json={"question": ""}, timeout=10),
     lambda r: r.status_code == 422),
    ("Whitespace-only",      lambda: requests.post(f"{BASE}/ask", json={"question": "   "}, timeout=10),
     lambda r: r.status_code == 422),
    ("Too long (1001 chars)", lambda: requests.post(f"{BASE}/ask", json={"question": "a" * 1001}, timeout=10),
     lambda r: r.status_code == 422),
]

for name, call, ok in cases:
    r = call()
    body = r.json()
    extra = f" latency_ms={body['latency_ms']}" if isinstance(body, dict) and "latency_ms" in body else ""
    print(f"{'PASS' if ok(r) else 'FAIL'} | {name:22} | HTTP {r.status_code}{extra}")
    if r.status_code == 200 and "answer" in body:
        print(f"       answer: {body['answer'][:150]}")
