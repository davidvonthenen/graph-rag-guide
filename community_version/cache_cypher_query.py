#!/usr/bin/env python3
"""
graph_rag_query.py ― Query pipeline for the Graph-based RAG agent

Core responsibilities
---------------------
- Detect entities in a user question (spaCy NER).
- Promote those entities (plus their paragraphs and, optionally, the
  parent document) from the authoritative **long-term** graph store
  into the high-speed **short-term** cache.
- Retrieve the most relevant paragraphs from the cache.
- Feed that context into an on-device Llama model to generate the answer.

Runtime conveniences
--------------------
- Keeps an in-memory `_seen_entities` set so each entity is promoted at
  most once per interactive session.
- Stores a TTL on the **relationship** layer; when the TTL expires the
  edge becomes invisible to Cypher without destructive deletes—perfect
  for governance audits.

Environment variables
---------------------
| Variable                | Default                         | Purpose                                  |
|-------------------------|---------------------------------|------------------------------------------|
| LONG_NEO4J_URI          | bolt://localhost:7688           | Long-term graph store                    |
| SHORT_NEO4J_URI         | bolt://localhost:7689           | Short-term cache                         |
| NEO4J_USER              | neo4j                           | Username for BOTH stores                 |
| LONG_NEO4J_PASSWORD     | neo4jneo4j1                     | Password for long-term store             |
| SHORT_NEO4J_PASSWORD    | neo4jneo4j2                     | Password for short-term cache            |
| PROMOTE_DOCUMENT_NODES  | 1 (truthy)                      | 0/false → skip document-level promotion  |
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

import spacy
from llama_cpp import Llama
from neo4j import GraphDatabase, Session

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

# Toggle: also copy the parent Document when promoting an Entity
PROMOTE_DOCUMENT_NODES = os.getenv("PROMOTE_DOCUMENT_NODES", "1").lower() not in {
    "0", "false", "no"
}

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

##############################################################################
# Thread-local state
##############################################################################

# Keeps track of promoted entities during this process lifetime
_seen_entities: set[Tuple[str, str]] = set()  # {(name_lower, label)}

##############################################################################
# Helpers - LLM & spaCy
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


@lru_cache(maxsize=1)
def load_spacy():
    """Load the spaCy pipeline once—NER doesn’t need to be re-instantiated."""
    return spacy.load("en_core_web_sm")


def extract_entities(nlp: spacy.Language, text: str) -> List[Tuple[str, str]]:
    """
    Extract (name, label) pairs for entities that matter.

    * Lower-case the surface form for stable matching.
    * Ignore very short spans to cut down noise.
    """
    doc = nlp(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in INTERESTING_ENTITY_TYPES and len(ent.text.strip()) >= 3
    ]


##############################################################################
# Neo4j connections
##############################################################################


def connect_short():
    """Return a Neo4j driver pointed at the short-term cache."""
    return GraphDatabase.driver(SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD))


def connect_long():
    """Return a Neo4j driver pointed at the long-term store."""
    return GraphDatabase.driver(LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD))


###############################################################################
# Index helpers
###############################################################################


def create_indexes(session) -> None:
    """
    Create indexes that matter for read latency.
    Safe to call multiple times; uses IF NOT EXISTS.
    """
    session.run(
        """
        CREATE RANGE INDEX ent_name_label IF NOT EXISTS
        FOR (e:Entity) ON (e.name, e.label)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX ent_uuid_idx IF NOT EXISTS
        FOR (e:Entity) ON (e.ent_uuid)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX para_uuid_idx IF NOT EXISTS
        FOR (p:Paragraph) ON (p.para_uuid)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX doc_uuid_idx IF NOT EXISTS
        FOR (d:Document) ON (d.doc_uuid)
        """
    )
    # Wait until all indexes are online before answering queries.
    session.run("CALL db.awaitIndexes()")


##############################################################################
# Promotion Cypher
##############################################################################

# Find the entity, the paragraph(s) that mention it, and the parent document.
# Filter out expired relationships in the long-term store.
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
# Low-level MERGE helpers (single responsibility, idempotent)
##############################################################################


def _merge_entity(
    sess: Session,
    ent_uuid: str,
    name: str,
    label: str,
    exp_ts: int,
) -> None:
    """Copy or update an Entity inside the short-term cache."""
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
    """Copy or update a Paragraph node."""
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
    """Copy or update the parent Document (optional)."""
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
    """Ensure (Paragraph)-[:PART_OF]->(Document) edge exists."""
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
    """Create or refresh the (Entity)-[:MENTIONS]->(Paragraph|Document) edge."""
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
    Copy an entity (plus context) from long-term store into short-term cache.
    Respects PROMOTE_DOCUMENT_NODES toggle and applies TTL.
    """
    exp_ts = now_ms + TTL_MS

    for rec in long_sess.run(PROMOTION_QUERY, name=name, label=label, now=now_ms):
        e_node = rec["e"]
        doc_node = rec["doc"]              # May be None if entity only linked to a paragraph
        para_nodes = rec["paras"]          # List[Node]

        # 1. Entity itself
        _merge_entity(short_sess, e_node["ent_uuid"], e_node["name"], e_node["label"], exp_ts)

        # 2. Optional document
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

        # 3. All paragraphs mentioning the entity
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


def ask(llm: Llama, nlp: spacy.Language, question: str, short_driver, long_driver):
    """
    End-to-end cycle for a single question:
      1. NER → entity extraction
      2. Promotion of unseen entities
      3. Retrieval from cache
      4. LLM answer generation
    """
    print(f"\n\u25B6  {question}")

    ent_pairs = extract_entities(nlp, question)
    if not ent_pairs:
        print("  (no interesting entities detected)")
        return
    
    # Note: spelling in print statement preserved to avoid code changes
    print(f"Erntities found: {ent_pairs}")

    now_ms = int(time.time() * 1000)
    with long_driver.session() as long_sess, short_driver.session() as short_sess:
        for name, label in ent_pairs:
            key = (name, label)
            if key not in _seen_entities:
                print("  Promoting entity:", key)
                promote_entity(name, label, long_sess, short_sess, now_ms)
                _seen_entities.add(key)

    # Retrieve paragraphs after promotion
    with short_driver.session() as sess:
        paras = fetch_paragraphs(sess, ent_pairs, top_k=100)

    if not paras:
        print("  No relevant context found.")
        return

    # Build a compact context block for the LLM
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
    """Simple demo—asks two hard-coded questions and prints timing."""
    short_driver = connect_short()
    long_driver = connect_long()
    llm = load_llm()
    nlp = load_spacy()

    # First question
    start_time = time.time()
    ask(llm, nlp, "Tell me about the connection between Ernie Wise and Vodafone.", short_driver, long_driver)
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")

    # Second question (should hit warm cache)
    start_time = time.time()
    ask(llm, nlp, "Tell me something about the personal life of Ernie Wise?", short_driver, long_driver)
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\nSession ended.")
