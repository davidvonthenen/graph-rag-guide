#!/usr/bin/env python3
"""
Retrieves from Long-Term Memory (LTM) only. No caching.

Paragraph-level RAG pipeline (Neo4j + spaCy + llama-cpp-python)

* Retrieves only (:Paragraph) nodes that are directly the target of
  [:MENTIONS] relationships from (:Entity) nodes.
* Ignores document-level edges so the LLM receives fine-grained context.

Model tested with TheBloke/neural-chat-7B-v3-3.Q4_K_M.gguf
"""

import os
import time
from pathlib import Path
from functools import lru_cache

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama   # pip install llama-cpp-python

##############################################################################
# Neo4j connection details
##############################################################################

NEO4J_URI      = os.getenv("LONG_NEO4J_URI",      "bolt://localhost:7688")
NEO4J_USER     = os.getenv("LONG_NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

##############################################################################
# Llama-cpp configuration
##############################################################################

MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """
    Load the GGUF model once, cache, and reuse.
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
        chat_format="chatml",  # Neural-Chat uses the ChatML template
    )


##############################################################################
# Neo4j helpers
##############################################################################

def connect_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

##############################################################################
# spaCy Named-Entity Recognition
##############################################################################

def extract_entities_spacy(text, nlp):
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) >= 3]

##############################################################################
# Graph query – fetch paragraphs mentioning entities
##############################################################################

def fetch_paragraphs_by_entities(session, entity_pairs, top_k=100):
    """
    Returns a list of dicts with keys:
      text, idx, title, category, match_count
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
    RETURN p.text   AS text,
           p.index  AS idx,
           d.title  AS title,
           d.category AS category,
           matchingEntities
    """

    # prepare parameter as list of [lowercased name, label] pairs
    # entity_list = [[name.lower(), label] for name, label in entity_pairs]
    entity_list = [[name.lower(), label] for name, label in entity_pairs]
    now_ms = int(time.time() * 1000)

    results = session.run(query,
                          entity_list=entity_list,
                          topK=top_k,
                          now=now_ms)

    return [dict(r) for r in results]

##############################################################################
# LLM-based answer generation
##############################################################################

def generate_answer(llm: Llama, question: str, context: str) -> str:
    system_msg = "You are an expert assistant answering questions using the given context."
    user_prompt = (
        f"You are given the following context from multiple documents:\n"
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

##############################################################################
# Main
##############################################################################

if __name__ == "__main__":
    # user_query = "What do these articles say about Ernie Wise?"
    user_query = "Tell me about the connection between Ernie Wise and Vodafone."
    # user_query = "Tell me about the connection between Ernie Wise and Vodafone."
    print(f"User Query: {user_query}")

    # LLM load
    print("Loading LLM model...")
    llm = load_llm()

    # Load spaCy model once
    nlp = spacy.load("en_core_web_sm")

    # start time
    start_time = time.time()

    # NER over user query
    entity_pairs = extract_entities_spacy(user_query, nlp)
    print("Recognized entities:", entity_pairs)

    # Neo4j — fetch docs
    driver = connect_neo4j()
    with driver.session() as session:
        paragraphs = fetch_paragraphs_by_entities(session, entity_pairs, top_k=8)

    if not paragraphs:
        print("No paragraphs found matching those entities.")
        exit(0)

    # ---------------------------------------------------------------- context
    combined_context = ""
    for para in paragraphs:
        snippet = para["text"][:350].replace("\n", " ")
        combined_context += (
            f"\n---\nDoc Title: {para['title']} | Category: {para['category']} "
            f"| Para #{para['idx']} | MatchCnt: {para['matchingEntities']}\n"
            f"{snippet}…\n"
        )

    # -------------------------------------------------------------- generate
    answer = generate_answer(llm, user_query, combined_context)
    end_time = time.time()
    print("\n================ RAG Answer ================\n")
    print(answer)
    print(f"Answer generated in {end_time - start_time:.2f} seconds.")
