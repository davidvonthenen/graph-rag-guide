#!/usr/bin/env python3
import sys
import json
import argparse
from typing import Optional
import requests

DEFAULT_URL = "http://127.0.0.1:8000/ner"

def read_stdin() -> str:
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input on stdin. Pass --text or pipe text to stdin.")
    return data

def post_ner(url: str, text: str, timeout: float = 5.0) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    # Raise for non-2xx to surface useful diagnostics
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # try to show server-provided JSON error if present
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise SystemExit(f"HTTP {r.status_code} from server: {msg}") from e
    try:
        return r.json()
    except Exception as e:
        raise SystemExit(f"Server returned non-JSON response: {r.text[:2000]}") from e

def main():
    ap = argparse.ArgumentParser(description="Client for spaCy NER REST API")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"NER endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--text", help="Text to analyze. If omitted, reads from stdin.")
    ap.add_argument("--timeout", type=float, default=8.0, help="Request timeout seconds (default: 8.0)")
    args = ap.parse_args()

    # text: Optional[str] = args.text
    # if args.text is None:
    #     print("Reading input text from stdin...", file=sys.stderr)
    #     text = read_stdin()

    text = "Microsoft is located in Redmond, and OpenAI is based in San Francisco."
    resp = post_ner(args.url, text, timeout=args.timeout)

    # Latest server response shape:
    # {
    #   "text": "...",
    #   "model": "en_core_web_sm",
    #   "entities": ["openai","san francisco", ...],
    #   "queue_size": N,
    #   "request_id": "uuid"
    # }
    model = resp.get("model")
    request_id = resp.get("request_id")
    entities = resp.get("entities", [])

    print(f"Model:        {model}")
    print(f"Request ID:   {request_id}")
    print(f"Entity Count: {len(entities)}")
    print("Entities:")
    for i, ent in enumerate(entities, 1):
        print(f"  {i:2d}. {ent}")

if __name__ == "__main__":
    main()
