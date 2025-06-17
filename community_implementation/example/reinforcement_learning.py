#!/usr/bin/env python3
"""
reinforcement_learning.py
=========================

Interactive demo that lets you *reinforce* five tech-news facts into
**short-term memory** (24 h expiry) and immediately run RAG queries that
retrieve only paragraph-level context.

The graph schema matches *ingest.py*:

    (:Document)<-[:PART_OF]-(:Paragraph)
    (:Entity)-[:MENTIONS {expiration}]->(:Paragraph | :Document)

Short-term facts are written with `expiration = now + 24 h` (in ms).  
Long-term data—ingested by *ingest.py*—has `expiration = 0`.

Prereqs
--------
* Python ≥3.10
* Neo4j Python driver 5.x
* spaCy (`en_core_web_sm`)
* llama-cpp-python
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase
from llama_cpp import Llama

##############################################################################
# Configuration
##############################################################################

MODEL_PATH = str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf")

# ─────────────────────────────────────────────────────────────────────────────

# Five BBC-style technology facts
TECH_FACTS = [
    """
    OpenAI has agreed to buy artificial intelligence-assisted coding tool Windsurf for about $3 billion, Bloomberg News reported on Monday, citing people familiar with the matter.
    The deal has not yet closed, the report added.

    OpenAI declined to comment, while Windsurf did not immediately respond to Reuters' requests for comment.

    Windsurf, formerly known as Codeium, had recently been in talks with investors, including General Catalyst and Kleiner Perkins, to raise funding at a $3 billion valuation, according to Bloomberg News.
    
    It was valued at $1.25 billion last August following a $150 million funding round led by venture capital firm General Catalyst. Other investors in the company include Kleiner Perkins and Greenoaks.
    
    The deal, which would be OpenAI's largest acquisition to date, would complement ChatGPT's coding capabilities. The company has been rolling out improvements in coding with the release of each of its newer models, but the competition is heating up.
    
    OpenAI has made several purchases in recent years to boost different segments of its AI products. It bought search and database analytics startup Rockset in a nine-figure stock deal last year, to provide better infrastructure for its enterprise products.
    
    OpenAI's weekly active users surged past 400 million in February, jumping sharply from the 300 million weekly active users in December.
    """,
    """
    Will the Apple Vision Pro be discontinued? It's certainly starting to look that way. In the last couple of months, numerous reports have emerged suggesting that Apple is either slowing down or completely halting production of its flagship headset.

    So, what does that mean for Apple's future in the extended reality market?

    Apple has had a rough time with its Vision Pro headset. Despite incredibly hype leading up to the initial release, and the fact that preorders for the device sold out almost instantly, demand for headset has consistently dropped over the last year.

    In fact, sales have diminished to the point that rumors have been coming thick and fast. For a while now, industry analysts and tech enthusiasts believe Apple might give up on its XR journey entirely and return its focus to other types of tech (like smartphones).

    However, while Apple has failed to achieve its sales targets with the Vision Pro, I don't think they will abandon the XR market entirely. It seems more likely that Apple will view the initial Vision Pro as an experiment, using it to pave the way to new, more popular devices.

    Here's what we know about Apple's XR journey right now.
    """,
    """
    OpenAI sees itself paying a lower share of revenue to its investor and close partner Microsoft by 2030 than it currently does, The Information reported, citing financial documents.

    The news comes after OpenAI this week changed tack on a major restructuring plan to pursue a new plan that would see its for-profit arm becoming a public benefit corporation (PBC) but continue to be controlled by its nonprofit division.

    OpenAI currently has an agreement to share 20% of its top line with Microsoft, but the AI company has told investors it expects to share 10% of revenue with its business partners, including Microsoft, by the end of this decade, The Information reported.

    Microsoft has invested tens of billions in OpenAI, and the two companies currently have a contract until 2030 that includes revenue sharing from both sides. The deal also gives Microsoft rights to OpenAI IP within its AI products, as well as exclusivity on OpenAI's APIs on Azure.

    Microsoft has not yet approved OpenAI's proposed corporate structure, Bloomberg reported on Monday, as the bigger tech company reportedly wants to ensure the new structure protects its multi-billion-dollar investment.

    OpenAI and Microsoft did not immediately return requests for comment.
    """,
    """
    Perplexity, the developer of an AI-powered search engine, is raising a $50 million seed and pre-seed investment fund, CNBC reported. Although the majority of the capital is coming from limited partners, Perplexity is using some of the capital it raised for the company's growth to anchor the fund. Perplexity reportedly raised $500 million at a $9 billion valuation in December.

    Perplexity's fund is managed by general partners Kelly Graziadei and Joanna Lee Shevelenko, who in 2018 co-founded an early-stage venture firm, F7 Ventures, according to PitchBook data. F7 has invested in startups like women's health company Midi. It's not clear if Graziadei and Shevelenko will continue to run F7 or if they will focus all their energies on Perplexity's venture fund.

    OpenAI also manages an investment fund known as the OpenAI Startup Fund. However, unlike Perplexity, OpenAI claims it does not use its own capital for these investments.
    """,
    """
    DeepSeek-R2 is the upcoming AI model from Chinese startup DeepSeek, promising major advancements in multilingual reasoning, code generation, and multimodal capabilities. Scheduled for early 2025, DeepSeek-R2 combines innovative training techniques with efficient resource usage, positioning itself as a serious global competitor to Silicon Valley's top AI technologies.

    In the rapidly evolving landscape of artificial intelligence, a new contender is emerging from China that promises to reshape global AI dynamics. DeepSeek, a relatively young AI startup, is making waves with its forthcoming DeepSeek-R2 model—a bold step in China's ambition to lead the global AI race.

    As Western tech giants like OpenAI, Anthropic, and Google dominate headlines, DeepSeek's R2 model represents a significant milestone in AI development from the East. With its unique approach to training, multilingual capabilities, and resource efficiency, DeepSeek-R2 isn't just another language model—it's potentially a game-changer for how we think about AI development globally.

    What is DeepSeek-R2?
    DeepSeek-R2 is a next-generation large language model that builds upon the foundation laid by DeepSeek-R1. According to reports from Reuters, DeepSeek may be accelerating its launch timeline, potentially bringing this advanced AI system to market earlier than the original May 2025 target.

    What sets DeepSeek-R2 apart is not just its improved performance metrics but its underlying architecture and training methodology. While R1 established DeepSeek as a serious competitor with strong multilingual and coding capabilities, R2 aims to push these boundaries significantly further while introducing new capabilities that could challenge the dominance of models like GPT-4 and Claude.

    DeepSeek-R2 represents China's growing confidence and technical capability in developing frontier AI technologies. The model has been designed from the ground up to be more efficient with computational resources—a critical advantage in the resource-intensive field of large language model development.
    """
]
TECH_CHECKS =[
    "Windsurf was bought by OpenAI for how much?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?"
]

# 24 hours in ms
EXPIRY_MS = 24 * 3600 * 1000

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

def extract_entities(nlp, text: str):
    doc = nlp(text)
    # print(f"Extracted entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) >= 3]


def insert_fact_with_expiry(tx, fact: str, nlp, expiry_ms: int):
    """
    Store a *single-paragraph* Document plus Entity links
    with `expiration = now + 24 h` on both doc- and paragraph-level edges.
    """
    ts_now  = int(time.time() * 1000)
    expire  = ts_now + expiry_ms
    doc_id  = str(uuid.uuid4())
    para_id = str(uuid.uuid4())

    # Document + Paragraph + PART_OF
    tx.run(
        """
        CREATE (d:Document {
            doc_uuid: $doc_id,
            title:    $title,
            content:  $content,
            expiration: 0,
            created_at: $ts
        })
        CREATE (p:Paragraph {
            para_uuid: $para_id,
            text:      $content,
            index:     0,
            doc_uuid:  $doc_id,
            expiration: 0,
            created_at: $ts
        })-[:PART_OF {expiration: 0}]->(d)
        """,
        doc_id=doc_id,
        para_id=para_id,
        title=fact[:80],
        content=fact.strip(),
        ts=ts_now,
    )

    # Entities + MENTIONS (→ Paragraph & Document)
    entity_pairs = extract_entities(nlp, fact)
    entities = [[name.lower(), label] for name, label in entity_pairs]
    print(f"entities: {entities}")

    for ent_name, ent_label in entities:
        if len(ent_name) < 3:
            continue

        tx.run(
            """
            MERGE (e:Entity {name:$name})
            ON CREATE SET
                e.ent_uuid   = $uuid,
                e.label      = $label,
                e.expiration = 0
            """,
            name=ent_name,
            uuid=str(uuid.uuid4()),
            label=ent_label,
        )

        tx.run(
            """
            MATCH (e:Entity {name:$name}),
                  (p:Paragraph {para_uuid:$para_id}),
                  (d:Document {doc_uuid:$doc_id})
            MERGE (e)-[mp:MENTIONS]->(p)
            ON CREATE SET mp.expiration = $exp
            MERGE (e)-[md:MENTIONS]->(d)
            ON CREATE SET md.expiration = $exp
            """,
            name=ent_name,
            para_id=para_id,
            doc_id=doc_id,
            exp=expire,
        )

    return doc_id


##############################################################################
# Retrieval helpers
##############################################################################


def fetch_paragraphs(tx, entity_pairs: list[tuple[str, str]], top_k: int = 8):
    if not entity_pairs:
        return []

    entity_list = [[name.lower(), label] for name, label in entity_pairs]
    print(f"entity_list: {entity_list}")
    now_ms = int(time.time() * 1000)

    result = tx.run(
        """
        MATCH (e:Entity)-[m:MENTIONS]->(p:Paragraph)-[:PART_OF]->(d:Document)
        WHERE [toLower(e.name), e.label] IN $elist
          AND (m.expiration = 0 OR m.expiration > $now)
        WITH p, d, count(e) AS matchCnt
        ORDER BY matchCnt DESC, p.index ASC
        LIMIT $k
        RETURN p.text AS text,
               p.index AS idx,
               d.title AS title,
               d.category AS category,
               matchCnt
        """,
        elist=entity_list,
        now=now_ms,
        k=top_k,
    )
    return [dict(r) for r in result]


def generate_answer(llm, question: str, context: str) -> str:
    system_msg = "You answer using ONLY the provided context."
    user_msg = (
        f"You are given the following context from multiple documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a concise answer."
    )
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user",   "content": user_msg}],
        temperature=0.2,
        top_p=0.95,
        max_tokens=4096,
    )
    return resp["choices"][0]["message"]["content"].strip()


##############################################################################
# Main loop
##############################################################################

def main():
    print("\n=== Reinforcement-Learning Memory Demo ===")

    # Load spaCy
    nlp = spacy.load("en_core_web_sm")

    # Load LLaMA model
    print("Loading local LLaMA model; please wait...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=32768,
        n_threads=8,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
    )

    driver = connect_neo4j()
    stored_docs: list[str] = []

    # ---------------------------------------------------------------- store
    with driver.session() as session:
        for fact in TECH_FACTS:
            print("\n────────────────────────────────────────")
            print(f"Fact:\n{fact.strip()}\n")
            if input("Store this fact for 24 h? [y/N] ").lower().startswith("y"):
                doc_id = insert_fact_with_expiry(session, fact, nlp, EXPIRY_MS)
                stored_docs.append(doc_id)

                print(f"↳ stored (doc_uuid={doc_id})")
            else:
                print("↳ skipped")

    # ---------------------------------------------------------------- query
    with driver.session() as session:
        for question in TECH_CHECKS:
            print("\n============================================================")
            print(f"User question: {question}")

            ents = extract_entities(nlp, question)
            paras = fetch_paragraphs(session, ents, top_k=10)

            if not paras:
                print("No relevant paragraphs found.\n")
                continue

            context = ""
            for p in paras:
                snippet = p['text'][:350].replace("\n", " ")
                context += (
                    f"\n---\n{p['title']} • Para#{p['idx']} "
                    f"(matches: {p['matchCnt']})\n{snippet}…"
                )

            answer = generate_answer(llm, question, context)
            print("\nAnswer:")
            print(answer)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
