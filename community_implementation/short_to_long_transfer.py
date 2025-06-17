#!/usr/bin/env python3
"""
short_to_long_transfer.py – promote sub‑graphs from the short‑term Neo4j
cache (port 7689) into the long‑term knowledge store (port 7688).

Toggle validation requirement with `REQUIRE_VALIDATED=1` if you want to copy
*only* `Document` nodes that carry `validated:true`.
"""

##############################################################################
# Imports & helpers
##############################################################################

import os
from typing import Any, Dict

from neo4j import GraphDatabase, Driver, Transaction


def _driver(uri_env: str, user_env: str, pw_env: str, default_uri: str) -> Driver:
    """Return a Neo4j driver using env vars or sensible defaults."""
    return GraphDatabase.driver(
        os.getenv(uri_env, default_uri),
        auth=(
            os.getenv(user_env, "neo4j"),
            os.getenv(pw_env,   "neo4j"),
        ),
    )


# --------------------------------------------------------------------------- #
# DB drivers – short‑term (cache) and long‑term (knowledge base)
# --------------------------------------------------------------------------- #
SHORT: Driver = _driver(
    "SHORT_NEO4J_URI", "SHORT_NEO4J_USER", "SHORT_NEO4J_PASSWORD",
    "bolt://localhost:7689",
)
LONG: Driver = _driver(
    "LONG_NEO4J_URI", "LONG_NEO4J_USER", "LONG_NEO4J_PASSWORD",
    "bolt://localhost:7688",
)

REQUIRE_VALIDATED = os.getenv("REQUIRE_VALIDATED", "0").lower() in ("1", "true", "yes")

##############################################################################
# Cypher snippets
##############################################################################

_Q_MERGE_DOCUMENT = """
MERGE (d:Document {doc_uuid:$doc_uuid})
SET   d += $props
"""

_Q_MERGE_PARAGRAPH = """
MATCH (d:Document {doc_uuid:$doc_uuid})
MERGE (p:Paragraph {para_uuid:$para_uuid})
SET   p += $props
MERGE (p)-[:PART_OF]->(d)
"""

_Q_MERGE_ENTITY = """
MERGE (e:Entity {ent_uuid:$ent_uuid})
SET   e += $props
"""

##############################################################################
# Merge helpers
##############################################################################


def _merge_document(tx: Transaction, props: Dict[str, Any]) -> None:
    tx.run(_Q_MERGE_DOCUMENT, doc_uuid=props["doc_uuid"], props=props)


def _merge_paragraph(tx: Transaction, p_props: Dict[str, Any], doc_uuid: str) -> None:
    tx.run(
        _Q_MERGE_PARAGRAPH,
        para_uuid=p_props["para_uuid"],
        props=p_props,
        doc_uuid=doc_uuid,
    )


def _merge_entity(tx: Transaction, e_props: Dict[str, Any]) -> None:
    tx.run(_Q_MERGE_ENTITY, ent_uuid=e_props["ent_uuid"], props=e_props)


def _merge_mentions(
    tx: Transaction, ent_uuid: str, target_label: str, target_uuid: str
) -> None:
    key = "doc_uuid" if target_label == "Document" else "para_uuid"
    cypher = f"""
        MATCH (e:Entity {{ent_uuid:$e}}),
              (t:{target_label} {{{key}:$t}})
        MERGE (e)-[m:MENTIONS]->(t)
        REMOVE m.expiration
    """
    tx.run(cypher, e=ent_uuid, t=target_uuid)


##############################################################################
# Promotion logic
##############################################################################


def _promote_one(doc_node, stx: Transaction, ltx: Transaction) -> None:
    """Copy one Document plus neighbourhood from short‑term to long‑term."""
    doc_props = dict(doc_node)
    doc_uuid = doc_props["doc_uuid"]

    # Document ----------------------------------------------------------------
    _merge_document(ltx, doc_props)

    # Paragraphs --------------------------------------------------------------
    for rec in stx.run(
        "MATCH (p:Paragraph)-[:PART_OF]->(:Document {doc_uuid:$du}) RETURN p",
        du=doc_uuid,
    ):
        _merge_paragraph(ltx, dict(rec["p"]), doc_uuid)

    # Entities ----------------------------------------------------------------
    for rec in stx.run(
        """
        MATCH (e:Entity)-[:MENTIONS]->(x)
        WHERE (x:Document  AND x.doc_uuid  =$du)
           OR (x:Paragraph AND x.doc_uuid  =$du)
        RETURN DISTINCT e
        """,
        du=doc_uuid,
    ):
        _merge_entity(ltx, dict(rec["e"]))

    # MENTIONS edges ----------------------------------------------------------
    for rec in stx.run(
        """
        MATCH (e:Entity)-[:MENTIONS]->(x)
        WHERE (x:Document  AND x.doc_uuid  =$du)
           OR (x:Paragraph AND x.doc_uuid  =$du)
        RETURN e.ent_uuid AS e_id,
               CASE WHEN x:Paragraph THEN x.para_uuid ELSE x.doc_uuid END AS tgt_id,
               CASE WHEN x:Paragraph THEN 'Paragraph' ELSE 'Document' END     AS tgt_lbl
        """,
        du=doc_uuid,
    ):
        _merge_mentions(ltx, rec["e_id"], rec["tgt_lbl"], rec["tgt_id"])

    # Mark as promoted --------------------------------------------------------
    stx.run(
        """
        MATCH (d:Document {doc_uuid:$du})
        SET   d.promoted   = true,
              d.promotedAt = timestamp()
        """,
        du=doc_uuid,
    )


##############################################################################
# Main entry‑point
##############################################################################


def main() -> None:
    where_clause = (
        "WHERE COALESCE(d.promoted,false)=false "
        + ("AND d.validated=true " if REQUIRE_VALIDATED else "")
    )
    query = f"MATCH (d:Document) {where_clause} RETURN d"

    with SHORT.session() as s_sess, LONG.session() as l_sess:
        docs = [rec["d"] for rec in s_sess.run(query)]
        print(
            f"🚀  {len(docs)} document(s) to promote "
            f"({'validation required' if REQUIRE_VALIDATED else 'no validation check'})"
        )

        promoted = 0
        for doc in docs:
            # separate write transactions per DB – no access_mode arg needed
            with s_sess.begin_transaction() as stx, l_sess.begin_transaction() as ltx:
                _promote_one(doc, stx, ltx)
                stx.commit()
                ltx.commit()
            promoted += 1

    print(f"✅  Promotion complete – {promoted} document(s) copied.")


if __name__ == "__main__":
    main()
