#!/usr/bin/env python3
"""
Paragraph-level RAG query against **long-term memory only** (no cache layer).

Workflow
--------
1. A user question is hard-coded in `__main__` for demonstration.
2. spaCy extracts (entity text, label) pairs from the question.
3. A Cypher query retrieves **Paragraph** nodes that those entities
   *directly* mention via the pattern (:Entity)-[:MENTIONS]->(:Paragraph).
   Document-level hops are deliberately skipped; we add the document’s
   title/category only for metadata.
4. The top-k paragraphs are concatenated into a context block and passed
   to a local LLaMA model (via llama-cpp-python) to generate the answer.

Why paragraph granularity?
--------------------------
Smaller chunks reduce token waste and keep answers focused.  By ignoring
(:Entity)-[:MENTIONS]->(:Document) edges, we avoid stuffing the LLM with
entire articles when a single paragraph will do.

Test hardware / model
---------------------
Script was validated with TheBloke/neural-chat-7b-v3-3.Q4_K_M.gguf on an
8-core workstation.  Adjust `n_threads`, `n_ctx`, and `MODEL_PATH` as
needed for your setup.
"""

import os
import time
from pathlib import Path
from functools import lru_cache

import spacy
from neo4j import GraphDatabase               # graph driver (code-specific)
from llama_cpp import Llama                   # pip install llama-cpp-python

# ─────────────────────────────────────────────────────────────────────────────
# Neo4j connection details - set via env-vars so CI / prod can override them
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("LONG_NEO4J_URI",      "bolt://localhost:7688")
NEO4J_USER     = os.getenv("LONG_NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

# ─────────────────────────────────────────────────────────────────────────────
# Llama-cpp configuration - model path & runtime knobs
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

@lru_cache(maxsize=1)          # ensure the heavy GGUF loads only once
def load_llm() -> Llama:
    """
    Instantiate and cache a llama-cpp model.

    * `n_ctx=32768` provides plenty of room for long contexts.
    * `temperature`, `top_p`, and `repeat_penalty` are tuned for
       deterministic yet mildly creative answers.
    """
    print(f"Loading model from {MODEL_PATH} …")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        verbose=False,
        chat_format="chatml",   # Neural-Chat follows the ChatML template
    )

# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────
def connect_neo4j():
    """Return a Neo4j driver instance using the env-var credentials."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ─────────────────────────────────────────────────────────────────────────────
# spaCy NER
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities_spacy(text: str, nlp):
    """
    Run NER over the input text.

    We keep entities whose surface form has ≥ 3 characters to skip
    "at", "UK", etc.  The function returns a list of `(name, label)` pairs.
    """
    doc = nlp(text)
    return [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if len(ent.text.strip()) >= 3
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Paragraph lookup
# ─────────────────────────────────────────────────────────────────────────────
def fetch_paragraphs_by_entities(session, entity_pairs, top_k: int = 100):
    """
    Fetch paragraphs that mention *any* of the given entities.

    Parameters
    ----------
    session : Neo4j session
    entity_pairs : list[tuple[str, str]]
        Each tuple is (lower-cased entity name, spaCy label).
    top_k : int
        Maximum number of paragraphs to return.

    Returns
    -------
    list[dict]
        Keys: text, idx, title, category, match_count
    """
    if not entity_pairs:
        return []

    query = """
    MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
    WHERE [toLower(e.name), e.label] IN $entity_list
      AND (m.expiration = 0 OR m.expiration > $now)
    WITH p, d, count(e) AS matchingEntities
    ORDER BY matchingEntities DESC, p.index ASC
    LIMIT $topK
    RETURN p.text      AS text,
           p.index     AS idx,
           d.title     AS title,
           d.category  AS category,
           matchingEntities
    """

    # Convert [('Vodafone', 'ORG'), …] → [['vodafone', 'ORG'], …]
    entity_list = [[name.lower(), label] for name, label in entity_pairs]
    now_ms = int(time.time() * 1000)

    results = session.run(
        query,
        entity_list=entity_list,
        topK=top_k,
        now=now_ms,
    )
    return [dict(r) for r in results]

# ─────────────────────────────────────────────────────────────────────────────
# LLM answer generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(llm: Llama, question: str, context: str) -> str:
    """
    Feed context + question into the local LLaMA model and return the reply.
    """
    system_msg = (
        "You are an expert assistant answering questions using the given "
        "context."
    )
    user_prompt = (
        "You are given the following context from multiple documents:\n"
        f"{context}\n\nQuestion: {question}\n\nProvide a concise answer."
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=32768,
    )
    return response["choices"][0]["message"]["content"].strip()

# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example prompts - replace with argparse / UI in real usage
    user_query = "Tell me about the connection between Ernie Wise and Vodafone."
    print(f"User Query: {user_query}")

    # Load heavy deps once
    print("Loading LLM model…")
    llm = load_llm()
    nlp = spacy.load("en_core_web_sm")

    # ── Pipeline start
    start_time = time.time()

    entity_pairs = extract_entities_spacy(user_query, nlp)
    print("Recognized entities:", entity_pairs)

    driver = connect_neo4j()
    with driver.session() as session:
        paragraphs = fetch_paragraphs_by_entities(session, entity_pairs, top_k=8)

    if not paragraphs:
        print("No paragraphs found matching those entities.")
        exit(0)

    # Build the context block shown to the LLM.
    combined_context = ""
    for para in paragraphs:
        snippet = para["text"][:350].replace("\n", " ")
        combined_context += (
            f"\n---\nDoc Title: {para['title']} | Category: {para['category']} "
            f"| Para #{para['idx']} | MatchCnt: {para['matchingEntities']}\n"
            f"{snippet}…\n"
        )

    answer = generate_answer(llm, user_query, combined_context)

    # ── Stats
    duration = time.time() - start_time
    print("\n================ RAG Answer ================\n")
    print(answer)
    print(f"\nAnswer generated in {duration:.2f} seconds.")
