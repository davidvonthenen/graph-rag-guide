#!/usr/bin/env python3
"""
ingest.py – Filtered NER ingest for Graph-based RAG

Creates a BBC-style knowledge graph in Neo4j:

  (:Document)<-[:PART_OF]-(:Paragraph)
  (:Entity)-[:MENTIONS {expiration:0}]->(:Paragraph|:Document)

Only entity labels listed in the NER_TYPES environment variable are stored.
If NER_TYPES is unset or empty, all spaCy entity labels are accepted.

Environment variables
---------------------
NEO4J_URI       – Neo4j Bolt URI (e.g. bolt://localhost:7688)
NEO4J_USER      – Neo4j username
NEO4J_PASSWORD  – Neo4j password
DATA_DIR        – Path to BBC dataset root (default: ./bbc)
NER_TYPES       – Comma-separated list of spaCy labels to ingest
                  e.g. "PERSON,ORG,GPE"   (case-insensitive)

Requires:  spaCy (en_core_web_sm), neo4j (>=5.0)
"""

import os
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase

###############################################################################
# Configuration
###############################################################################

NEO4J_URI      = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER     = os.getenv("LONG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")
DATASET_PATH   = os.getenv("DATA_DIR", "./bbc")

# _raw_labels = [lab.strip().upper() for lab in os.getenv("NER_TYPES", "").split(",") if lab.strip()]
_raw_labels = {
    "PERSON", "ORG", "PRODUCT", "GPE", "EVENT",
    "WORK_OF_ART", "NORP", "LOC"
}
ALLOWED_LABELS = set(_raw_labels) if _raw_labels else None     # None == accept all

###############################################################################
# Cypher helpers
###############################################################################


def merge_entity(tx, ent_uuid: str, name: str, label: str) -> None:
    name = name.lower().strip()
    tx.run(
        """
        MERGE (e:Entity {name:$name})
        ON CREATE SET
            e.ent_uuid   = $ent_uuid,
            e.label      = $label,
            e.expiration = 0
        """,
        name=name,
        ent_uuid=ent_uuid,
        label=label,
    )


def create_document(tx, doc_uuid: str, title: str, content: str, category: str) -> None:
    tx.run(
        """
        MERGE (d:Document {doc_uuid:$doc_uuid})
        ON CREATE SET
            d.title      = $title,
            d.content    = $content,
            d.category   = $category,
            d.expiration = 0
        """,
        doc_uuid=doc_uuid,
        title=title,
        content=content,
        category=category,
    )


def create_paragraph(tx, para_uuid: str, text: str, idx: int, doc_uuid: str) -> None:
    tx.run(
        """
        MERGE (p:Paragraph {para_uuid:$para_uuid})
        ON CREATE SET
            p.text       = $text,
            p.index      = $idx,
            p.doc_uuid   = $doc_uuid,
            p.expiration = 0
        """,
        para_uuid=para_uuid,
        text=text,
        idx=idx,
        doc_uuid=doc_uuid,
    )

    tx.run(
        """
        MATCH (p:Paragraph {para_uuid:$para_uuid}),
              (d:Document {doc_uuid:$doc_uuid})
        MERGE (p)-[r:PART_OF]->(d)
        ON CREATE SET r.expiration = 0
        """,
        para_uuid=para_uuid,
        doc_uuid=doc_uuid,
    )


def link_mentions(tx, ent_uuid: str, doc_uuid: str, para_uuid: str) -> None:
    # Paragraph-level mention
    tx.run(
        """
        MATCH (e:Entity {ent_uuid:$ent_uuid}),
              (p:Paragraph {para_uuid:$para_uuid})
        MERGE (e)-[m:MENTIONS]->(p)
        ON CREATE SET m.expiration = 0
        """,
        ent_uuid=ent_uuid,
        para_uuid=para_uuid,
    )

    # Document-level mention
    tx.run(
        """
        MATCH (e:Entity {ent_uuid:$ent_uuid}),
              (d:Document {doc_uuid:$doc_uuid})
        MERGE (e)-[m:MENTIONS]->(d)
        ON CREATE SET m.expiration = 0
        """,
        ent_uuid=ent_uuid,
        doc_uuid=doc_uuid,
    )

###############################################################################
# Ingest logic
###############################################################################


def ingest_file(nlp, session, category: str, path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    title, body = lines[0], "\n".join(lines[1:])
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    doc_uuid = str(uuid.uuid4())

    print(f"\u27A4  {title}  [{category}]")   # nice arrow prefix

    session.execute_write(create_document, doc_uuid, title, body, category)

    for idx, text in enumerate(paragraphs):
        para_uuid = str(uuid.uuid4())
        session.execute_write(create_paragraph, para_uuid, text, idx, doc_uuid)

        for ent in nlp(text).ents:
            if ALLOWED_LABELS and ent.label_.upper() not in ALLOWED_LABELS:
                continue

            ent_uuid = str(uuid.uuid4())
            session.execute_write(merge_entity, ent_uuid, ent.text, ent.label_)
            session.execute_write(link_mentions, ent_uuid, doc_uuid, para_uuid)


def main() -> None:
    print("Loading spaCy model …")
    nlp = spacy.load("en_core_web_sm")

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver, driver.session() as session:
        # Clean slate
        print("Clearing old data from Neo4j …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.\n")

        # Walk dataset
        for category in sorted(os.listdir(DATASET_PATH)):
            category_path = Path(DATASET_PATH) / category
            if not category_path.is_dir():
                continue

            print(f"\n=== Category: {category} ===")
            for txt in sorted(category_path.glob("*.txt")):
                ingest_file(nlp, session, category, txt)

    if ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        print(f"\nFinished. Ingest restricted to entity types: {allowed}")
    else:
        print("\nFinished. All entity types ingested.")


if __name__ == "__main__":
    main()
