# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
graph_rag_query.py ― Query pipeline for the Graph-based RAG agent
=================================================================

This *helper* script demonstrates the **full query loop** for the dual-memory
Graph-based RAG reference implementation:

    1. Send the user’s question to the local NER REST service (spaCy runs
       inside that service) to detect named entities.
    2. The service promotes any *new* entities (and their supporting
       paragraphs / documents) from **long-term memory** into the
       **short-term cache**.
    3. Retrieve the best-matching paragraphs from the cache.
    4. Feed that context plus the question into a local Llama model.
    5. Return a concise, grounded answer.

Nothing here is production-grade orchestration; the goal is to show *how* the
pieces connect so you can copy/paste the bits you need.

---------------------------------------------------------------------------  
Environment variables (defaults match the hard-coded fallbacks below)  
---------------------------------------------------------------------------

| Variable                | Default                         | Purpose                                  |
|-------------------------|---------------------------------|------------------------------------------|
| NER_SERVICE_URL         | http://127.0.0.1:8000/ner       | Endpoint for entity extraction + promo  |
| SHORT_NEO4J_URI         | bolt://localhost:7689           | Short-term cache                         |
| SHORT_NEO4J_PASSWORD    | neo4jneo4j2                     | Short-term DB password                   |
| NEO4J_USER              | neo4j                           | User name for the short-term database    |
| MODEL_PATH              | ~/models/neural-chat-7b-v3…gguf | Path to a local **llama-cpp** model      |
| PROMOTION_TTL_MS        | 86400000                        | Optional TTL override sent to the service|

Notes
-----

- The NER REST service is responsible for talking to both long-term and
  short-term Neo4j instances. The environment variables `LONG_NEO4J_URI`,
  `LONG_NEO4J_PASSWORD`, and `PROMOTE_DOCUMENT_NODES` are still respected—but
  they are read by the service process rather than this script.
- Short-term memory is given a shorter TTL (24 h by default) via an `expiration`
  property on every cached relationship.

"""

##############################################################################
# Imports
##############################################################################

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

from llama_cpp import Llama
from neo4j import GraphDatabase, Session
from ner_api import NerServiceError, call_ner_service, parse_entity_pairs

##############################################################################
# Configuration - change via ENV or edit defaults for local testing
##############################################################################

# Short-term cache (NVMe / RAM disk / FlexCache)
SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

# Path to a local gguf model for llama-cpp (≈ 4 GB quantised 7-B Llama)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf"),
)

# Only these entity labels are considered "interesting" for promotion
INTERESTING_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    "LOC",
}

# Time-to-live for cached facts (24 hours, expressed in ms)
TTL_MS = 24 * 60 * 60 * 1000

##############################################################################
# Helpers - LLM
##############################################################################


@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """Lazy-load the llama-cpp model once and reuse it across calls."""
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        chat_format="chatml",
        verbose=False,
    )
##############################################################################
# Neo4j connection helpers - keep driver boilerplate out of the main logic
##############################################################################


def connect_short():
    """Return a driver bound to the short-term cache."""
    return GraphDatabase.driver(
        SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD)
    )


##############################################################################
# Retrieval - rank cached paragraphs by "how many entities match"
##############################################################################

FETCH_PARAS_QUERY = """
MATCH (e:Entity)-[m:MENTIONS]->(t:Paragraph)
WHERE [toLower(e.name), e.label] IN $entity_list
  AND ( m.expiration IS NULL
     OR m.expiration = 0
     OR m.expiration > $now )
OPTIONAL MATCH (t)-[:PART_OF]->(d:Document)
WITH t, d, count(e) AS matchingEntities
ORDER BY matchingEntities DESC,
         coalesce(t.index, 0) ASC
LIMIT $topK
RETURN
  t.text                                   AS text,
  t.index                                  AS idx,
  coalesce(d.title, 'Untitled')            AS title,
  coalesce(d.category, 'N/A')              AS category,
  matchingEntities
"""


def fetch_paragraphs(
    session: Session, entity_pairs: Iterable[Tuple[str, str]], top_k: int = 100
):
    """Return the top-k paragraphs for the given entity set."""
    if not entity_pairs:
        return []

    now_ms = int(time.time() * 1000)
    entity_list = [[name, label] for name, label in entity_pairs]

    return [
        dict(r)
        for r in session.run(
            FETCH_PARAS_QUERY,
            entity_list=entity_list,
            now=now_ms,
            topK=top_k,
        )
    ]


##############################################################################
# LLM answer generation - vanilla llama-cpp chat completion
##############################################################################


def generate_answer(llm: Llama, question: str, context: str) -> str:
    """Wrap the LLM call so the rest of the code stays readable."""
    sys_msg = (
        "You are an expert assistant answering the user's question using only "
        "the provided context."
    )
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer concisely."
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=4096,
    )
    return resp["choices"][0]["message"]["content"].strip()


##############################################################################
# Demo loop - minimal conversational wrapper for manual testing
##############################################################################


def ask(llm: Llama, question: str, short_driver):
    """
    Full cycle for *one* question:

        1. Extract entities via the REST NER service.
        2. Fetch context from the short-term cache (promotion happens server-side).
        3. Generate an answer with the LLM.
    """
    print(f"\n\u25B6  {question}")

    try:
        ner_response = call_ner_service(
            question,
            promote=True,
            ttl_ms=TTL_MS,
            labels=INTERESTING_ENTITY_TYPES,
        )
    except NerServiceError as exc:
        print(f"  Failed to contact NER service: {exc}")
        return
    ent_pairs = parse_entity_pairs(ner_response)
    if not ent_pairs:
        print("  (no interesting entities detected)")
        return

    print(f"Entities found: {ent_pairs}")

    with short_driver.session() as sess:
        paras = fetch_paragraphs(sess, ent_pairs, top_k=100)

    if not paras:
        print("  No relevant context found.")
        return

    # Build a human-readable context block for the LLM
    context_block = ""
    for p in paras:
        snippet = p["text"][:350].replace("\n", " ")
        context_block += (
            f"\n---\nDoc: {p['title']} | Para #{p['idx']} "
            f"| Matches: {p['matchingEntities']}\n{snippet}…"
        )

    answer = generate_answer(llm, question, context_block)
    print("\n=== Answer ===")
    print(answer)
    print("===============")


def main():
    """Entry point - initialise drivers, models, and run a single-shot demo."""
    short_driver = connect_short()
    llm = load_llm()

    # Demo query - replace with your own questions
    start_time = time.time()
    ask(
        llm,
        "Windsurf was bought by OpenAI for how much?",
        short_driver,
    )
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Neo4j drivers auto-close via GC, but an explicit message is nice UX
        print("\nSession ended.")
