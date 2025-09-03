# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
transfer_and_query.py
=====================

Interactive demo that shows how short-term facts are either **promoted**
into long-term memory or allowed to expire, then runs a quick RAG sanity
check to prove the promotion worked.

Workflow
--------

1. **Scan short-term memory** for any document/paragraph that still has an
   unexpired `:MENTIONS` edge (identified by `expiration > now`).

2. For every candidate, ask the human operator:

      • **yes**    → "Promote": set `expiration = 0`, meaning *permanent*
      • **expire** → "Force expire": set `expiration` to two days ago
      • **no**     → Leave the edge as-is

3. After the review, run five tech-news questions (`TECH_CHECK`) through a
   lightweight RAG loop that pulls **paragraph-level** context from the
   graph and feeds it to a local LLaMA model.

Graph schema (shared with *ingest.py* and *reinforcement_learning.py*):

    (:Document)<-[:PART_OF]-(:Paragraph)
    (:Entity)-[:MENTIONS {expiration}]->(:Paragraph | :Document)

Only the comments have changed—**the executable code is identical**.
"""

from __future__ import annotations

import time
import os
from pathlib import Path

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

##############################################################################
# Configuration
##############################################################################

# Path to the local GGUF model. Change to taste or wire in an env-var.
MODEL_PATH = str(Path.home() / "models" / "neural-chat-7B-v3-3.Q4_K_M.gguf")

# Quick sanity-check questions we’ll ask after the promotion phase.
TECH_CHECK = [
    "Windsurf was bought by OpenAI for how much?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?",
]

# Utility: "two days" expressed in milliseconds.
TWO_DAYS_MS = 2 * 24 * 3600 * 1000

##############################################################################
# Neo4j helpers
##############################################################################

def connect(uri_env: str, user_env: str, pass_env: str) -> GraphDatabase.driver:
    """
    Generic helper so we can reuse the same pattern for other databases.

    Environment variables
    ---------------------
    uri_env   - connection string, e.g. bolt://localhost:7687
    user_env  - database user
    pass_env  - database password
    """
    uri  = os.getenv(uri_env,  "bolt://localhost:7687")
    user = os.getenv(user_env, "neo4j")
    pw   = os.getenv(pass_env, "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pw))

def connect_neo4j():
    """
    Convenience wrapper that looks up SHORT_* variables specifically for the
    short-term cache instance.
    """
    return connect("SHORT_NEO4J_URI", "SHORT_NEO4J_USER", "SHORT_NEO4J_PASSWORD")

# ────────────────────────────────────────────────────────────────────────────
# Short-term document discovery
# ────────────────────────────────────────────────────────────────────────────

def list_unexpired_docs(tx):
    """
    Return all *Document* nodes that still have at least one unexpired
    paragraph-level mention.

    A mention is considered "unexpired" when:
        • m.expiration IS NOT NULL
        • m.expiration > current-time-in-ms
    """
    now_ms = int(time.time() * 1000)
    query = """
    MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
    WHERE m.expiration IS NOT NULL AND m.expiration > $now               // still valid
    WITH d, collect(DISTINCT e.name) AS ents
    RETURN d.doc_uuid AS uuid,
           d.title    AS title,
           left(d.content, 400) AS snippet,                              // preview text
           ents
    ORDER BY title
    """
    return [dict(r) for r in tx.run(query, now=now_ms)]

# ────────────────────────────────────────────────────────────────────────────
# Expiration / promotion utilities
# ────────────────────────────────────────────────────────────────────────────

def _set_expiration(tx, doc_uuid: str, new_exp: int | None):
    """
    Helper that touches **all** incoming :MENTIONS edges for a given
    document *and* its paragraphs, then sets `expiration`:

        • new_exp == 0     → promote (make permanent)
        • new_exp is None  → leave untouched                (not used here)
        • new_exp > 0      → set specific timestamp         (force-expire)

    The Cypher is written idempotently so repeated calls are safe.
    """
    tx.run(
        """
        MATCH (d:Document {doc_uuid:$uuid})
        OPTIONAL MATCH (d)<-[:PART_OF]-(p:Paragraph)
        WITH collect(d) + collect(p) AS nodes
        UNWIND nodes AS n
        MATCH (e:Entity)-[m:MENTIONS]->(n)
        // CASE 1: promote → expiration = 0  (permanent)
        FOREACH (_ IN CASE WHEN $exp IS NULL THEN [1] ELSE [] END |
                 SET m.expiration = 0)
        // CASE 2: force-expire → expiration = past-timestamp
        FOREACH (_ IN CASE WHEN $exp IS NOT NULL THEN [1] ELSE [] END |
                 SET m.expiration = $exp)
        """,
        uuid=doc_uuid,
        exp=new_exp,
    )

def promote_to_long_term(tx, doc_uuid: str):
    """Wrapper for readability: sets expiration=0 (permanent)."""
    _set_expiration(tx, doc_uuid, None)

def force_expire(tx, doc_uuid: str):
    """Sets expiration to ‘two days ago’, making the edge immediately invalid."""
    past_ms = int(time.time() * 1000) - TWO_DAYS_MS
    _set_expiration(tx, doc_uuid, past_ms)

##############################################################################
# Retrieval helpers (paragraph-level, name+label pairs)
##############################################################################

def extract_entities(nlp, text: str):
    """Simple spaCy-based NER that returns (text, label) tuples ≥3 chars."""
    doc = nlp(text)
    return [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if len(ent.text.strip()) >= 3
    ]

def fetch_paragraphs(tx, entity_pairs: list[tuple[str, str]], top_k: int = 8):
    """
    Retrieve up to *top_k* paragraphs whose incoming :MENTIONS edges match
    ANY of the (name, label) pairs and are unexpired.

    Ordering:
        1. Number of entity matches (desc) - surfaces highly relevant paras
        2. Paragraph index          (asc)  - keeps original document order
    """
    if not entity_pairs:
        return []

    now_ms = int(time.time() * 1000)
    # Normalize names to lowercase because ingestion did the same.
    entity_list = [[n.lower(), l] for n, l in entity_pairs]

    query = """
    MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
    WHERE [toLower(e.name), e.label] IN $elist
      AND (m.expiration = 0 OR m.expiration > $now)       // permanent or unexpired
    WITH p, d, count(e) AS matchCnt
    ORDER BY matchCnt DESC, p.index ASC
    LIMIT $k
    RETURN p.text   AS text,
           p.index  AS idx,
           d.title  AS title,
           matchCnt
    """
    return [dict(r) for r in tx.run(query, elist=entity_list, now=now_ms, k=top_k)]

##############################################################################
# LLM helpers
##############################################################################

def load_llm(path=MODEL_PATH):
    """
    Load a local GGUF LLaMA model via llama-cpp-python.

    * n_ctx=32768  - keeps us safe for long prompts
    * temperature  - low for deterministic answers
    """
    print("Loading local LLaMA model…")
    return Llama(
        model_path=path,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        chat_format="chatml",
    )

def generate_answer(llm, question: str, context: str):
    """
    Very thin wrapper around llama-cpp’s chat_completion API.
    Prompts the model to answer **exclusively** from the supplied context.
    """
    system = "You answer using ONLY the provided context."
    user = (
        f"You are given the following context from multiple documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a concise answer."
    )
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.2,
        top_p=0.95,
        max_tokens=4096,
    )
    return resp["choices"][0]["message"]["content"].strip()

##############################################################################
# Main workflow
##############################################################################

def review_short_term(driver):
    """
    Interactive CLI:
        • Lists every unexpired short-term document
        • Prompts operator for action (yes / expire / no)
        • Applies the chosen mutation immediately
    """
    with driver.session() as sess:
        docs = sess.read_transaction(list_unexpired_docs)
        if not docs:
            print("No unexpired short-term documents found.")
            return

        for d in docs:
            print("\n────────────────────────────────────────")
            print(f"Title   : {d['title']}")
            print(f"UUID    : {d['uuid']}")
            print(f"Entities: {d['ents']}")
            print(f"Snippet : {d['snippet'].replace(chr(10), ' ')}…")
            choice = input("Promote to long-term [yes] / "
                           "force-expire [expire] / leave [no]? ").lower()

            if choice == "yes":
                sess.write_transaction(promote_to_long_term, d["uuid"])
                print("  ↳ promoted.")
            elif choice == "expire":
                sess.write_transaction(force_expire, d["uuid"])
                print("  ↳ expired.")
            else:
                print("  ↳ left unchanged.")

def rag_test(driver, llm, nlp):
    """
    Mini RAG loop that:
        1. Extracts entities from each test question
        2. Fetches up to 10 relevant paragraphs from the graph
        3. Calls the LLM and prints the answer
    """
    with driver.session() as sess:
        for q in TECH_CHECK:
            print("\n============================================================")
            print(f"Question: {q}")
            ents = extract_entities(nlp, q)
            paras = fetch_paragraphs(sess, ents, top_k=10)

            if not paras:
                print("No relevant paragraphs found.")
                continue

            # Build a compact, human-readable context block.
            context = ""
            for p in paras:
                snippet = p["text"][:350].replace("\n", " ")
                context += (
                    f"\n---\n{p['title']} • Para#{p['idx']} "
                    f"(matches: {p['matchCnt']})\n{snippet}…"
                )

            answer = generate_answer(llm, q, context)
            print("\n=== Answer ===")
            print(answer)

def main():
    print("=== Transfer-and-Query Demo (paragraph granular, expire option) ===")

    nlp = spacy.load("en_core_web_sm")
    llm = load_llm()

    driver = connect_neo4j()

    # 1. Manual review of short-term memory
    review_short_term(driver)

    # 2. Quick RAG sanity-check
    rag_test(driver, llm, nlp)

    driver.close()
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
