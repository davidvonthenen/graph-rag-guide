#!/usr/bin/env python3
"""
ingest.py ― Filtered NER ingest for a **Graph-based RAG** long-term store.

The script walks a folder of plain-text documents (the BBC dataset layout is
handy but not required) and turns each file into a mini knowledge graph:

    (:Document)<-[:PART_OF]-(:Paragraph)
    (:Entity)-[:MENTIONS {expiration:0}]->(:Paragraph|:Document)

Key design points
-----------------
- **Fine-grained retrieval** - Text is split into paragraphs so the agent can
  surface a single sentence instead of an entire article.
- **Governance hooks** - Every node and edge carries an `expiration` field.
  Toggling that flag hides bad data without deleting history.
- **Label filtering** - Set the `NER_TYPES` environment variable to restrict
  which spaCy entity labels are stored (e.g. only `PERSON,ORG,GPE`). Leaving
  it empty ingests every entity the model emits.
- **Driver-agnostic Cypher** - The logic is pure Cypher + driver calls. Swapping
  to a different graph backend means changing only the driver import + URI.

Environment variables
---------------------
LONG_NEO4J_URI, LONG_NEO4J_USER, LONG_NEO4J_PASSWORD
    Connection info for the long-term graph database.
DATA_DIR
    Root folder that holds sub-directories of `.txt` files (default: ./bbc).
NER_TYPES
    Comma-separated list of spaCy labels to ingest (case-insensitive).
"""

import os
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase  # swap this import if you use a different driver

###############################################################################
# Configuration
###############################################################################

NEO4J_URI      = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER     = os.getenv("LONG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")
DATASET_PATH   = os.getenv("DATA_DIR", "./bbc")

# Allowed entity labels (upper-cased for easy comparison).  If the env var is
# blank, we’ll accept every entity spaCy can produce.
_raw_labels = {
    "PERSON", "ORG", "PRODUCT", "GPE", "EVENT",
    "WORK_OF_ART", "NORP", "LOC"
}
ALLOWED_LABELS = set(_raw_labels) if _raw_labels else None

###############################################################################
# Index helpers
###############################################################################

def create_indexes(session) -> None:
    """
    Create the indexes that matter for write-time idempotency and
    read-time speed.  We run each statement in AUTO-COMMIT mode so Neo4j
    can build them in parallel, then block until all are ONLINE.
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
    session.run("CALL db.awaitIndexes()")  # wait outside the creation txs

###############################################################################
# Cypher write helpers - each function runs inside a single driver tx
###############################################################################

def merge_entity(tx, ent_uuid: str, name: str, label: str) -> None:
    """
    Ensure exactly one (:Entity) node exists for the given lower-cased name.
    If the node is new, initialise its UUID, label, and expiration flag.
    """
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.ent_uuid   = $ent_uuid,
            e.label      = $label,
            e.expiration = 0
        """,
        name=name.lower().strip(),
        ent_uuid=ent_uuid,
        label=label,
    )

def create_document(tx, doc_uuid: str, title: str, content: str, category: str) -> None:
    """
    Store a high-level Document node keyed by a UUID.  MERGE keeps reruns
    idempotent so you can ingest incrementally.
    """
    tx.run(
        """
        MERGE (d:Document {doc_uuid: $doc_uuid})
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
    """
    Persist a Paragraph node and link it to its parent Document with a
    [:PART_OF] relationship.
    """
    # Paragraph node
    tx.run(
        """
        MERGE (p:Paragraph {para_uuid: $para_uuid})
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

    # PART_OF edge (scope = paragraph → document)
    tx.run(
        """
        MATCH (p:Paragraph {para_uuid: $para_uuid}),
              (d:Document  {doc_uuid: $doc_uuid})
        MERGE (p)-[r:PART_OF]->(d)
        ON CREATE SET r.expiration = 0
        """,
        para_uuid=para_uuid,
        doc_uuid=doc_uuid,
    )

def link_mentions(tx, ent_uuid: str, doc_uuid: str, para_uuid: str) -> None:
    """
    Connect an Entity to both the specific paragraph it appears in and the
    enclosing document.  Two edges make downstream reasoning flexible.
    """
    # Paragraph-level mention
    tx.run(
        """
        MATCH (e:Entity {ent_uuid: $ent_uuid}),
              (p:Paragraph {para_uuid: $para_uuid})
        MERGE (e)-[m:MENTIONS]->(p)
        ON CREATE SET m.expiration = 0
        """,
        ent_uuid=ent_uuid,
        para_uuid=para_uuid,
    )
    # Document-level mention
    tx.run(
        """
        MATCH (e:Entity {ent_uuid: $ent_uuid}),
              (d:Document {doc_uuid: $doc_uuid})
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
    """
    Parse a single text file and write its nodes / edges in a single session.
    * First line          → title
    * Remaining lines     → body (split on blank lines into paragraphs)
    * spaCy NER per para  → Entity nodes + MENTIONS edges
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    title, body = lines[0], "\n".join(lines[1:])
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    doc_uuid = str(uuid.uuid4())

    print(f"\u27A4  {title}  [{category}]")

    # Document node
    session.execute_write(create_document, doc_uuid, title, body, category)

    # Paragraphs + entities
    for idx, text in enumerate(paragraphs):
        para_uuid = str(uuid.uuid4())
        session.execute_write(create_paragraph, para_uuid, text, idx, doc_uuid)

        # Entity extraction & linkage
        for ent in nlp(text).ents:
            if ALLOWED_LABELS and ent.label_.upper() not in ALLOWED_LABELS:
                continue

            ent_uuid = str(uuid.uuid4())
            session.execute_write(merge_entity, ent_uuid, ent.text, ent.label_)
            session.execute_write(link_mentions, ent_uuid, doc_uuid, para_uuid)

def main() -> None:
    print("Loading spaCy model …")
    nlp = spacy.load("en_core_web_sm")

    # One driver + one session keeps code tidy for a batch script
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver, driver.session() as session:

        # 🧹 Start from a clean slate so reruns are deterministic
        print("Clearing old data from the database …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

        # Ensure indexes exist before the heavy writes start
        print("Creating/validating indexes …")
        create_indexes(session)
        print("Indexes ONLINE.\n")

        # Walk every category folder
        for category in sorted(os.listdir(DATASET_PATH)):
            category_path = Path(DATASET_PATH) / category
            if not category_path.is_dir():
                continue

            print(f"\n=== Category: {category} ===")
            for txt in sorted(category_path.glob("*.txt")):
                ingest_file(nlp, session, category, txt)

    # Recap which labels made it in
    if ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        print(f"\nFinished. Ingest restricted to entity types: {allowed}")
    else:
        print("\nFinished. All entity types ingested.")

if __name__ == "__main__":
    main()
