# Copyright 2025 NetApp, Inc. All Rights Reserved.

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
  which entity labels are stored (e.g. only `PERSON,ORG,GPE`). Leaving it
  empty ingests every entity the NER service emits.
- **Driver-agnostic Cypher** - The logic is pure Cypher + driver calls. Swapping
  to a different graph backend means changing only the driver import + URI.

Environment variables
---------------------
LONG_NEO4J_URI, LONG_NEO4J_USER, LONG_NEO4J_PASSWORD
    Connection info for the long-term graph database.
DATA_DIR
    Root folder that holds sub-directories of `.txt` files (default: ./bbc).
NER_TYPES
    Comma-separated list of entity labels to ingest (case-insensitive). The
    NER REST service applies this filter server-side.
"""

import os
import uuid
from pathlib import Path

from neo4j import GraphDatabase  # swap this import if you use a different driver
from common import call_ner_service, create_indexes, parse_entity_pairs

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
# Cypher write helpers - each function runs inside a single driver tx
###############################################################################

def merge_entity(tx, ent_uuid: str, name: str, label: str) -> str:
    """Return the UUID for the (:Entity) identified by ``name``.

    Args:
        tx: Neo4j transaction context provided by ``session.execute_write``.
        ent_uuid: Fresh UUID candidate used when the entity is first created.
        name: Canonical (case-insensitive) entity surface form.
        label: spaCy entity label (PERSON, ORG, …).

    Returns:
        The persistent UUID stored on the entity node.  Existing nodes keep
        their original UUID; new nodes adopt ``ent_uuid``.
    """

    record = tx.run(
        """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.ent_uuid   = $ent_uuid,
            e.label      = $label,
            e.expiration = 0
        SET e.ent_uuid = coalesce(e.ent_uuid, $ent_uuid)
        RETURN e.ent_uuid AS ent_uuid
        """,
        name=name.lower().strip(),
        ent_uuid=ent_uuid,
        label=label,
    ).single()

    return record["ent_uuid"]

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

def ingest_file(session, category: str, path: Path) -> None:
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

        response = call_ner_service(
            text,
            promote=False,
            labels=sorted(ALLOWED_LABELS) if ALLOWED_LABELS else None,
        )
        entity_pairs = parse_entity_pairs(response)
        if ALLOWED_LABELS:
            entity_pairs = [
                (name, label)
                for name, label in entity_pairs
                if label.upper() in ALLOWED_LABELS
            ]

        for name, label in entity_pairs:
            ent_uuid = session.execute_write(
                merge_entity, str(uuid.uuid4()), name, label
            )
            session.execute_write(link_mentions, ent_uuid, doc_uuid, para_uuid)

def main() -> None:
    print("Loading NER service …")

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
                ingest_file(session, category, txt)

    # Recap which labels made it in
    if ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        print(f"\nFinished. Ingest restricted to entity types: {allowed}")
    else:
        print("\nFinished. All entity types ingested.")

if __name__ == "__main__":
    main()
