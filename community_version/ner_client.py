#!/usr/bin/env python3
import sys
import argparse
from typing import Sequence

from common import call_ner_service, parse_entity_pairs

DEFAULT_URL = "http://127.0.0.1:8000/ner"


def read_stdin() -> str:
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input on stdin. Pass --text or pipe text to stdin.")
    return data


def post_ner(
    url: str,
    text: str,
    *,
    labels: Sequence[str] | None = None,
    promote: bool | None = None,
    ttl_ms: int | None = None,
    timeout: float = 5.0,
) -> dict:
    return call_ner_service(
        text,
        labels=labels,
        promote=promote,
        ttl_ms=ttl_ms,
        timeout=timeout,
        url=url,
    )

def main():
    ap = argparse.ArgumentParser(description="Client for spaCy NER REST API")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"NER endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--text", help="Text to analyze. If omitted, reads from stdin.")
    ap.add_argument("--timeout", type=float, default=8.0, help="Request timeout seconds (default: 8.0)")
    ap.add_argument("--no-promote", action="store_true", help="Disable cache promotion on the request")
    ap.add_argument("--ttl-ms", type=int, help="Override promotion TTL in milliseconds")
    args = ap.parse_args()

    # text: Optional[str] = args.text
    # if args.text is None:
    #     print("Reading input text from stdin...", file=sys.stderr)
    #     text = read_stdin()

    text = "Microsoft is located in Redmond, and OpenAI is based in San Francisco."
    resp = post_ner(
        args.url,
        text,
        promote=None if not args.no_promote else False,
        ttl_ms=args.ttl_ms,
        timeout=args.timeout,
    )

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
    entity_pairs = parse_entity_pairs(resp)

    print(f"Model:        {model}")
    print(f"Request ID:   {request_id}")
    print(f"Entity Count: {len(entities)}")
    print("Entities:")
    for i, ent in enumerate(entities, 1):
        print(f"  {i:2d}. {ent}")
    if entity_pairs:
        print("Entity pairs:")
        for i, (name, label) in enumerate(entity_pairs, 1):
            print(f"  {i:2d}. {name} [{label}]")

if __name__ == "__main__":
    main()
