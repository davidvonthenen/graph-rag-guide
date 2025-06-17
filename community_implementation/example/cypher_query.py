#!/usr/bin/env python3
"""
graph_rag_query.py ― Query pipeline for the Graph-based RAG agent

Fixes & improvements
--------------------
1. **Full-document promotion** – when an entity is promoted, *all* paragraphs
   belonging to the referenced document are copied into the short-term store,
   not just the first paragraph that happened to match.
2. **Optional document promotion** – set `PROMOTE_DOCUMENT_NODES=0` to copy only
   the paragraphs; the `Document` node itself (and the entity→document
   MENTIONS edge) is then skipped.
3. **Correct default Neo4j ports** – the short-term DB is `:7688`, long-term
   is `:7687`, matching the top-of-file comment.
4. **Proper edge creation** – the code now distinguishes between paragraph and
   document targets when creating `MENTIONS` relationships.

Environment variables
---------------------
| Variable                | Default                       | Purpose                               |
|-------------------------|-------------------------------|---------------------------------------|
| LONG_NEO4J_URI          | bolt://localhost:7687         | Long-term store                       |
| SHORT_NEO4J_URI         | bolt://localhost:7688         | Short-term cache                      |
| NEO4J_USER              | neo4j                         | Neo4j user (both DBs)                 |
| LONG_NEO4J_PASSWORD     | neo4jneo4j1                   | Long-term password                    |
| SHORT_NEO4J_PASSWORD    | neo4jneo4j2                   | Short-term password                   |
| PROMOTE_DOCUMENT_NODES  | 1 (truthy)                    | 0/false → skip document promotion     |
| MODEL_PATH              | ~/models/neural-chat-7b-v3…   | gguf path for llama-cpp               |
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

LONG_NEO4J_URI = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
LONG_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
LONG_NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

PROMOTE_DOCUMENT_NODES = os.getenv("PROMOTE_DOCUMENT_NODES", "1").lower() not in {
    "0",
    "false",
    "no",
}

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf"),
)

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

TTL_MS = 24 * 60 * 60 * 1000  # one day

##############################################################################
# Thread-local state
##############################################################################

_seen_entities: set[Tuple[str, str]] = set()  # {(name_lower, label)}

##############################################################################
# Helpers – LLM & spaCy
##############################################################################


@lru_cache(maxsize=1)
def load_llm() -> Llama:
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
    return spacy.load("en_core_web_sm")


def extract_entities(nlp: spacy.Language, text: str) -> List[Tuple[str, str]]:
    """Lower-case entity text for stable matching."""
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
    return GraphDatabase.driver(SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD))


def connect_long():
    return GraphDatabase.driver(LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD))


##############################################################################
# Promotion queries
##############################################################################

# 1. Find every paragraph (p) and its parent document (d) in which the entity appears.
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
# Promotion logic
##############################################################################


def _merge_entity(
    sess: Session,
    ent_uuid: str,
    name: str,
    label: str,
    exp_ts: int,
) -> None:
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
    """Copy entity + related paragraphs (and optionally the document) into the cache."""
    exp_ts = now_ms + TTL_MS

    for rec in long_sess.run(PROMOTION_QUERY, name=name, label=label, now=now_ms):
        e_node = rec["e"]
        doc_node = rec["doc"]  # may be None if entity only linked to a paragraph
        para_nodes = rec["paras"]  # list[Node]

        # Entity
        _merge_entity(short_sess, e_node["ent_uuid"], e_node["name"], e_node["label"], exp_ts)

        # Document (optional)
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

        # All paragraphs inside the doc that mention the entity
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
# Retrieval from short-term cache
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
# Demo loop
##############################################################################


def ask(llm: Llama, nlp: spacy.Language, question: str, short_driver, long_driver):
    """Full query → promotion → retrieval → answer cycle for a single question."""
    print(f"\n\u25B6  {question}")

    ent_pairs = extract_entities(nlp, question)
    if not ent_pairs:
        print("  (no interesting entities detected)")
        return
    
    print(f"Erntities found: {ent_pairs}")

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
    short_driver = connect_short()
    long_driver = connect_long()
    llm = load_llm()
    nlp = load_spacy()

    # Demo
    start_time = time.time()
    ask(llm, nlp, "Windsurf was bought by OpenAI for how much?", short_driver, long_driver)
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\nSession ended.")
