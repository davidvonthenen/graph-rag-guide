# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
graph_rag_query.py ― Query pipeline for the Graph-based RAG agent

Core responsibilities
--------------------
- Detect entities in a user question via the dedicated NER REST service
  (spaCy runs inside that process).
- The service promotes those entities (plus their paragraphs and, optionally,
  the parent document) from the authoritative **long-term** graph store into
  the high-speed **short-term** cache.
- Retrieve the most relevant paragraphs from the cache.
- Feed that context into an on-device Llama model to generate the answer.

Runtime conveniences
--------------------
- Stores a TTL on the **relationship** layer; when the TTL expires the
  edge becomes invisible to Cypher without destructive deletes—perfect
  for governance audits.

Environment variables
---------------------
| Variable                | Default                         | Purpose                                  |
|-------------------------|---------------------------------|------------------------------------------|
| NER_SERVICE_URL         | http://127.0.0.1:8000/ner       | Endpoint for entity extraction + promo  |
| LONG_NEO4J_URI          | bolt://localhost:7688           | Long-term graph store                    |
| SHORT_NEO4J_URI         | bolt://localhost:7689           | Short-term cache                         |
| NEO4J_USER              | neo4j                           | Username for BOTH stores                 |
| LONG_NEO4J_PASSWORD     | neo4jneo4j1                     | Password for long-term store             |
| SHORT_NEO4J_PASSWORD    | neo4jneo4j2                     | Password for short-term cache            |
| PROMOTE_DOCUMENT_NODES  | 1 (truthy)                      | Controls document promotion inside service |
| MODEL_PATH              | ~/models/neural-chat-7b-v3…     | gguf checkpoint consumed by llama-cpp    |
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

import bm25s
import Stemmer
from llama_cpp import Llama
from neo4j import GraphDatabase, Session

from common import NerServiceError, call_ner_service, create_indexes, parse_entity_pairs

##############################################################################
# Configuration
##############################################################################

# Long-term store (authoritative knowledge base)
LONG_NEO4J_URI = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
LONG_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
LONG_NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

# Short-term cache (high-speed working set)
SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

# Local Llama checkpoint
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf"),
)

# Labels considered "interesting" for promotion
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

TTL_MS = 60 * 60 * 1000  # one hour in milliseconds

# Re-ranker configuration
BM25_TOP_K = 10
_STEMMER = Stemmer.Stemmer("english")

# Helpers - LLM
##############################################################################


@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """Load llama-cpp only once per process (thread-safe)."""
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
# Neo4j connections
##############################################################################


def connect_short():
    """Return a Neo4j driver pointed at the short-term cache."""
    return GraphDatabase.driver(SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD))


def connect_long():
    """Return a Neo4j driver pointed at the long-term store."""
    return GraphDatabase.driver(LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD))


##############################################################################
# Retrieval query (short-term cache only)
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
    """Return up to *top_k* paragraphs ranked by entity overlap."""
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


def rerank_paragraphs(question: str, paragraphs: List[dict], top_k: int = BM25_TOP_K) -> List[dict]:
    """Re-rank cached paragraphs with BM25-S and return the top ``top_k`` items.

    Args:
        question: The natural-language query supplied by the user.
        paragraphs: Raw paragraph dictionaries retrieved from Neo4j.
        top_k: Number of paragraphs to return after BM25 ranking.

    Returns:
        The highest-scoring ``top_k`` paragraphs with an added ``bm25_score`` field.
    """

    if not paragraphs or not question.strip():
        return []

    corpus = [p.get("text", "") for p in paragraphs]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=_STEMMER)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    query_tokens = bm25s.tokenize(question, stemmer=_STEMMER)
    k = min(top_k, len(corpus))
    doc_ids, scores = retriever.retrieve(query_tokens, k=k)

    reranked = []
    for doc_id, score in zip(doc_ids[0], scores[0]):
        para = paragraphs[int(doc_id)].copy()
        para["bm25_score"] = float(score)
        reranked.append(para)

    return reranked


##############################################################################
# LLM answer generation
##############################################################################


def generate_answer(llm: Llama, question: str, context: str) -> str:
    """Call llama-cpp with the retrieved context."""
    sys_msg = "You are an expert assistant answering the user's question using only the provided context."
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
# Demo loop helpers
##############################################################################


def ask(llm: Llama, question: str, short_driver, top_k: int = BM25_TOP_K) -> None:
    """
    End-to-end cycle for a single question:
      1. NER → entity extraction via the REST service (which also promotes data)
      2. Retrieval from the short-term cache
      3. LLM answer generation
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

    # Note: spelling in print statement preserved to avoid code changes
    print(f"Erntities found: {ent_pairs}")

    # Retrieve paragraphs after promotion
    with short_driver.session() as sess:
        paras = fetch_paragraphs(sess, ent_pairs, top_k=(top_k * 10))

    if not paras:
        print("  No relevant context found.")
        return

    reranked_paras = rerank_paragraphs(question, paras, top_k=top_k)
    if not reranked_paras:
        print("  Unable to rerank retrieved context.")
        return

    # Build a compact context block for the LLM
    context_block = ""
    for p in reranked_paras:
        snippet = p["text"][:350].replace("\n", " ")
        context_block += (
            f"\n---\nDoc: {p['title']} | Para #{p['idx']} "
            f"| Matches: {p['matchingEntities']} | BM25: {p.get('bm25_score', 0):.2f}\n{snippet}…"
        )

    answer = generate_answer(llm, question, context_block)
    print("\n=== Answer ===")
    print(answer)
    print("===============")


def main():
    """Simple demo—asks two hard-coded questions and prints timing."""
    short_driver = connect_short()
    llm = load_llm()

    # First question
    start_time = time.time()
    ask(llm, "Tell me about the connection between Ernie Wise and Vodafone.", short_driver)
    # ask(llm, "How much did Google purchase Windsurf for?", short_driver)
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")

    # Second question (should hit warm cache)
    start_time = time.time()
    ask(llm, "Tell me something about the personal life of Ernie Wise?", short_driver)
    # ask(llm, "How much did OpenAI purchase Windsurf for?", short_driver)
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\nSession ended.")
