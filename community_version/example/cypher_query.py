# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
graph_rag_query.py ― Query pipeline for the Graph-based RAG agent
=================================================================

This *helper* script demonstrates the **full query loop** for the dual-memory
Graph-based RAG reference implementation:

    1. Parse the user’s question, extract named entities with spaCy.
    2. Promote any *new* entities (and their supporting paragraphs / documents)
       from **long-term memory** into the **short-term cache**.
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
| LONG_NEO4J_URI          | bolt://localhost:7688           | Long-term graph store                    |
| LONG_NEO4J_PASSWORD     | neo4jneo4j1                     | Long-term DB password                    |
| SHORT_NEO4J_URI         | bolt://localhost:7689           | Short-term cache                         |
| SHORT_NEO4J_PASSWORD    | neo4jneo4j2                     | Short-term DB password                   |
| NEO4J_USER              | neo4j                           | User name for *both* databases           |
| PROMOTE_DOCUMENT_NODES  | 1 (truthy)                      | 0 → copy only paragraphs, not documents  |
| MODEL_PATH              | ~/models/neural-chat-7b-v3…gguf | Path to a local **llama-cpp** model      |

Notes  
-----

- Two separate Neo4j instances are shown for clarity, **but the Cypher works
  against any graph database** that supports the same semantics.
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

import spacy
from llama_cpp import Llama
from neo4j import GraphDatabase, Session

##############################################################################
# Configuration - change via ENV or edit defaults for local testing
##############################################################################

# Long-term memory (cheaper storage, complete history)
LONG_NEO4J_URI = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
LONG_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
LONG_NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

# Short-term cache (NVMe / RAM disk / FlexCache)
SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

# Promote the *Document* node as well as its Paragraphs?
PROMOTE_DOCUMENT_NODES = os.getenv("PROMOTE_DOCUMENT_NODES", "1").lower() not in {
    "0",
    "false",
    "no",
}

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
# Thread-local state - prevents re-promoting the same entity in one session
##############################################################################

_seen_entities: set[Tuple[str, str]] = set()  # {(name_lower, label)}

##############################################################################
# Helpers - LLM & spaCy
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


@lru_cache(maxsize=1)
def load_spacy():
    """Load the lightweight spaCy NER model exactly once."""
    return spacy.load("en_core_web_sm")


def extract_entities(nlp: spacy.Language, text: str) -> List[Tuple[str, str]]:
    """
    Pull (entity_text, entity_label) pairs from *text*.

    * Lower-case the text for stable matching in the graph.
    * Ignore labels outside INTERESTING_ENTITY_TYPES.
    * Skip trivial one- or two-character entities.
    """
    doc = nlp(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in INTERESTING_ENTITY_TYPES and len(ent.text.strip()) >= 3
    ]


##############################################################################
# Neo4j connection helpers - keep driver boilerplate out of the main logic
##############################################################################


def connect_short():
    """Return a driver bound to the short-term cache."""
    return GraphDatabase.driver(
        SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD)
    )


def connect_long():
    """Return a driver bound to the long-term store."""
    return GraphDatabase.driver(
        LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD)
    )


##############################################################################
# Promotion queries - Cypher that copies an entity-centric sub-graph
##############################################################################

# 1) Find the entity
# 2) Grab the Document (or Paragraph) it’s attached to
# 3) Collect *all* paragraphs in that document
# We copy those nodes/relationships into the cache in one go
PROMOTION_QUERY = """
MATCH (e:Entity {name:$name, label:$label})-[m:MENTIONS]->(t)
WHERE  m.expiration IS NULL
   OR  m.expiration = 0
   OR  m.expiration > $now
OPTIONAL MATCH (t)-[:PART_OF]->(d:Document)
WITH e, COALESCE(d, t) AS doc              /* t may itself be a Document */
OPTIONAL MATCH (doc)<-[:PART_OF]-(p:Paragraph)
WITH e, doc, collect(DISTINCT p) AS paras
RETURN e, doc, paras
"""

##############################################################################
# Low-level MERGE helpers - keep the Cypher in one place for clarity
##############################################################################


def _merge_entity(
    sess: Session,
    ent_uuid: str,
    name: str,
    label: str,
    exp_ts: int,
) -> None:
    """Insert or update an Entity node in the cache."""
    sess.run(
        """
        MERGE (e:Entity {ent_uuid:$uuid})
        ON CREATE SET e.name=$name,
                      e.label=$label,
                      e.expiration=$exp
        SET e.expiration=$exp
        """,
        uuid=ent_uuid,
        name=name,
        label=label,
        exp=exp_ts,
    )


def _merge_paragraph(
    sess: Session,
    para_node,
    exp_ts: int,
) -> None:
    """Insert or update a Paragraph node in the cache."""
    sess.run(
        """
        MERGE (p:Paragraph {para_uuid:$uuid})
        ON CREATE SET p.text=$text,
                      p.index=$idx,
                      p.doc_uuid=$doc_uuid,
                      p.expiration=$exp
        SET p.expiration=$exp
        """,
        uuid=para_node["para_uuid"],
        text=para_node["text"],
        idx=para_node["index"],
        doc_uuid=para_node["doc_uuid"],
        exp=exp_ts,
    )


def _merge_document(sess: Session, doc_node, exp_ts: int) -> None:
    """Insert or update a Document node in the cache."""
    sess.run(
        """
        MERGE (d:Document {doc_uuid:$uuid})
        ON CREATE SET d.title=$title,
                      d.content=$content,
                      d.category=$category,
                      d.expiration=$exp
        SET d.expiration=$exp
        """,
        uuid=doc_node["doc_uuid"],
        title=doc_node["title"],
        content=doc_node.get("content", ""),
        category=doc_node["category"],
        exp=exp_ts,
    )


def _merge_part_of(sess: Session, para_uuid: str, doc_uuid: str) -> None:
    """Ensure (Paragraph)-[:PART_OF]->(Document) exists."""
    sess.run(
        """
        MATCH (p:Paragraph {para_uuid:$p}), (d:Document {doc_uuid:$d})
        MERGE (p)-[:PART_OF]->(d)
        """,
        p=para_uuid,
        d=doc_uuid,
    )


def _merge_mentions(
    sess: Session,
    ent_uuid: str,
    target_label: str,
    target_id_name: str,
    target_id_value: str,
    exp_ts: int,
) -> None:
    """Create/refresh (Entity)-[:MENTIONS]->(Paragraph|Document) edges."""
    sess.run(
        f"""
        MATCH (e:Entity {{ent_uuid:$e_uuid}}),
              (t:{target_label} {{{target_id_name}:$tid}})
        MERGE (e)-[m:MENTIONS]->(t)
        SET   m.expiration=$exp
        """,
        e_uuid=ent_uuid,
        tid=target_id_value,
        exp=exp_ts,
    )


def promote_entity(
    name: str,
    label: str,
    long_sess: Session,
    short_sess: Session,
    now_ms: int,
) -> None:
    """
    Copy an entity, its paragraphs, and (optionally) its document
    from long-term memory into the short-term cache.

    * TTL is encoded on every node/edge so the cache self-expires.
    * MERGE keeps the operation idempotent—safe to retry.
    """
    exp_ts = now_ms + TTL_MS

    for rec in long_sess.run(PROMOTION_QUERY, name=name, label=label, now=now_ms):
        e_node = rec["e"]
        doc_node = rec["doc"]  # May be None if entity only linked to a paragraph
        para_nodes = rec["paras"]  # list[Node]

        # --- Entity --------------------------------------------------------
        _merge_entity(short_sess, e_node["ent_uuid"], e_node["name"], e_node["label"], exp_ts)

        # --- Document (optional) ------------------------------------------
        if doc_node and PROMOTE_DOCUMENT_NODES:
            _merge_document(short_sess, doc_node, exp_ts)
            _merge_mentions(
                short_sess,
                e_node["ent_uuid"],
                "Document",
                "doc_uuid",
                doc_node["doc_uuid"],
                exp_ts,
            )

        # --- Paragraphs ----------------------------------------------------
        for p in para_nodes:
            _merge_paragraph(short_sess, p, exp_ts)
            _merge_part_of(short_sess, p["para_uuid"], p["doc_uuid"])
            _merge_mentions(
                short_sess,
                e_node["ent_uuid"],
                "Paragraph",
                "para_uuid",
                p["para_uuid"],
                exp_ts,
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


def ask(llm: Llama, nlp: spacy.Language, question: str, short_driver, long_driver):
    """
    Full cycle for *one* question:

        1. Extract entities.
        2. Promote unseen ones.
        3. Fetch context from cache.
        4. Generate answer with the LLM.
    """
    print(f"\n\u25B6  {question}")

    ent_pairs = extract_entities(nlp, question)
    if not ent_pairs:
        print("  (no interesting entities detected)")
        return

    print(f"Entities found: {ent_pairs}")

    now_ms = int(time.time() * 1000)
    with long_driver.session() as long_sess, short_driver.session() as short_sess:
        for name, label in ent_pairs:
            key = (name, label)
            if key not in _seen_entities:
                print("  Promoting entity:", key)
                promote_entity(name, label, long_sess, short_sess, now_ms)
                _seen_entities.add(key)

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
    long_driver = connect_long()
    llm = load_llm()
    nlp = load_spacy()

    # Demo query - replace with your own questions
    start_time = time.time()
    ask(
        llm,
        nlp,
        "Windsurf was bought by OpenAI for how much?",
        short_driver,
        long_driver,
    )
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Neo4j drivers auto-close via GC, but an explicit message is nice UX
        print("\nSession ended.")
