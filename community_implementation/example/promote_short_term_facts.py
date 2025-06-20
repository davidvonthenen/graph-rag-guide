#!/usr/bin/env python3
"""
transfer_and_query.py
=====================

Review every *short-term* document (identified by unexpired `:MENTIONS.expiration`
edges on **paragraphs OR their parent document**), let the user:

  • **yes**    → promote to long-term (`expiration = 0`)
  • **expire** → force-expire (`expiration = now - 2 days`)
  • **no**     → leave unchanged

Then run a quick RAG sanity-check over the five TECH_CHECK questions,
retrieving **paragraph-level** context only.

Graph schema (identical to *ingest.py* and *reinforcement_learning.py*):

    (:Document)<-[:PART_OF]-(:Paragraph)
    (:Entity)-[:MENTIONS {expiration}]->(:Paragraph | :Document)
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

MODEL_PATH = str(Path.home() / "models" / "neural-chat-7B-v3-3.Q4_K_M.gguf")

TECH_CHECK = [
    "Windsurf was bought by OpenAI for how much?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?",
]

# two days in ms
TWO_DAYS_MS = 2 * 24 * 3600 * 1000

##############################################################################
# Neo4j helpers
##############################################################################

def connect(uri_env: str, user_env: str, pass_env: str) -> GraphDatabase.driver:
    uri  = os.getenv(uri_env,  "bolt://localhost:7687")
    user = os.getenv(user_env, "neo4j")
    pw   = os.getenv(pass_env, "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pw))

def connect_neo4j():
    return connect("SHORT_NEO4J_URI", "SHORT_NEO4J_USER", "SHORT_NEO4J_PASSWORD")


def list_unexpired_docs(tx):
    """
    Return a list of docs that still have *any* unexpired paragraph-level mentions.
    """
    now_ms = int(time.time() * 1000)
    query = """
    MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
    WHERE m.expiration IS NOT NULL AND m.expiration > $now
    WITH d, collect(DISTINCT e.name) AS ents
    RETURN d.doc_uuid AS uuid,
           d.title    AS title,
           left(d.content, 400) AS snippet,
           ents
    ORDER BY title
    """
    return [dict(r) for r in tx.run(query, now=now_ms)]


def _set_expiration(tx, doc_uuid: str, new_exp: int):
    # collect doc + paragraphs, then touch every incoming MENTIONS edge
    tx.run(
        """
        MATCH (d:Document {doc_uuid:$uuid})
        OPTIONAL MATCH (d)<-[:PART_OF]-(p:Paragraph)
        WITH collect(d) + collect(p) AS nodes
        UNWIND nodes AS n
        MATCH (e:Entity)-[m:MENTIONS]->(n)
        FOREACH (_ IN CASE WHEN $exp IS NULL THEN [1] ELSE [] END |
                 SET m.expiration = 0)
        FOREACH (_ IN CASE WHEN $exp IS NOT NULL THEN [1] ELSE [] END |
                 SET m.expiration = $exp)
        """,
        uuid=doc_uuid,
        exp=new_exp,
    )


def promote_to_long_term(tx, doc_uuid: str):
    _set_expiration(tx, doc_uuid, 0)


def force_expire(tx, doc_uuid: str):
    past_ms = int(time.time() * 1000) - TWO_DAYS_MS
    _set_expiration(tx, doc_uuid, past_ms)


##############################################################################
# Retrieval helpers (paragraph-level, name+label pair like rag_query.py)
##############################################################################


def extract_entities(nlp, text: str):
    doc = nlp(text)
    return [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if len(ent.text.strip()) >= 3
    ]


def fetch_paragraphs(tx, entity_pairs: list[tuple[str, str]], top_k: int = 8):
    if not entity_pairs:
        return []
    now_ms = int(time.time() * 1000)
    entity_list = [[n.lower(), l] for n, l in entity_pairs]

    query = """
    MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
    WHERE [toLower(e.name), e.label] IN $elist
      AND (m.expiration = 0 OR m.expiration > $now)
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
    with driver.session() as sess:
        for q in TECH_CHECK:
            print("\n============================================================")
            print(f"Question: {q}")
            ents = extract_entities(nlp, q)
            paras = fetch_paragraphs(sess, ents, top_k=10)

            if not paras:
                print("No relevant paragraphs found.")
                continue

            context = ""
            for p in paras:
                snippet = p["text"][:350].replace("\n", " ")
                context += (
                    f"\n---\n{p['title']} • Para#{p['idx']} "
                    f"(matches: {p['matchCnt']})\n{snippet}…"
                )

            answer = generate_answer(llm, q, context)
            print("\nAnswer:")
            print(answer)


def main():
    print("=== Transfer-and-Query Demo (paragraph granular, expire option) ===")

    nlp = spacy.load("en_core_web_sm")
    llm = load_llm()

    driver = connect_neo4j()

    # 1. Review / promote / expire
    review_short_term(driver)

    # 2. Quick RAG sanity-check
    rag_test(driver, llm, nlp)

    driver.close()
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
