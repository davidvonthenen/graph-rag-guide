# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
short_to_long_transfer.py — Promote *validated* knowledge chunks from the **short-term
graph cache** into the **authoritative long-term store**.

Two Neo4j instances are assumed:

- **Short-term cache** - volatile, high-speed (e.g. NVMe/RAM disk) listening on port 7689
- **Long-term store**  - durable, compliance-grade storage on port 7688

Promotion criteria:

- By default every `Document` node that is still **unpromoted** (`promoted=false`) is
  eligible.  
- Set the environment variable **`REQUIRE_VALIDATED=1`** to only copy documents that
  have already been manually or programmatically flagged `validated:true`.

All connectivity details (URI, user, password) can be overridden with environment
variables so that the same script works in *any* environment without code changes.
"""

##############################################################################
# Imports & helpers
##############################################################################

import os
from typing import Any, Dict

from neo4j import GraphDatabase, Driver, Transaction


def _driver(uri_env: str, user_env: str, pw_env: str, default_uri: str) -> Driver:
    """
    Return a Neo4j driver using environment variables *or* a sensible default.

    Parameters
    ----------
    uri_env : str
        Name of the env var that may contain the bolt URI (e.g. bolt://host:port).
    user_env : str
        Name of the env var that may contain the DB username.
    pw_env   : str
        Name of the env var that may contain the DB password.
    default_uri : str
        Fallback URI when the env var is absent.

    This small helper keeps the driver construction DRY.
    """
    return GraphDatabase.driver(
        os.getenv(uri_env, default_uri),
        auth=(
            os.getenv(user_env, "neo4j"),
            os.getenv(pw_env, "neo4j"),
        ),
    )


# --------------------------------------------------------------------------- #
# DB drivers - short-term (cache) and long-term (knowledge base)
# --------------------------------------------------------------------------- #
# In typical local demos we simply run two Neo4j instances on adjacent ports.
# Swap the URIs or credentials via environment variables to point at real clusters.
SHORT: Driver = _driver(
    "SHORT_NEO4J_URI", "SHORT_NEO4J_USER", "SHORT_NEO4J_PASSWORD",
    "bolt://localhost:7689",
)
LONG: Driver = _driver(
    "LONG_NEO4J_URI", "LONG_NEO4J_USER", "LONG_NEO4J_PASSWORD",
    "bolt://localhost:7688",
)

# Copy-only documents that were previously marked as validated when this flag is set.
REQUIRE_VALIDATED = os.getenv("REQUIRE_VALIDATED", "0").lower() in ("1", "true", "yes")

##############################################################################
# Cypher snippets - parameterised fragments reused by the merge helpers
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
# Merge helpers - keep all write statements in one place for readability
##############################################################################


def _merge_document(tx: Transaction, props: Dict[str, Any]) -> None:
    """Idempotently create/update a Document node in the long-term store."""
    tx.run(_Q_MERGE_DOCUMENT, doc_uuid=props["doc_uuid"], props=props)


def _merge_paragraph(tx: Transaction, p_props: Dict[str, Any], doc_uuid: str) -> None:
    """Idempotently create/update a Paragraph node and attach it to its Document."""
    tx.run(
        _Q_MERGE_PARAGRAPH,
        para_uuid=p_props["para_uuid"],
        props=p_props,
        doc_uuid=doc_uuid,
    )


def _merge_entity(tx: Transaction, e_props: Dict[str, Any]) -> None:
    """Idempotently create/update an Entity node."""
    tx.run(_Q_MERGE_ENTITY, ent_uuid=e_props["ent_uuid"], props=e_props)


def _merge_mentions(
    tx: Transaction, ent_uuid: str, target_label: str, target_uuid: str
) -> None:
    """
    Re-create a MENTIONS edge in the long-term store.

    The edge’s `expiration` property (used for cache invalidation) is **removed**
    so that the relationship becomes permanent.
    """
    key = "doc_uuid" if target_label == "Document" else "para_uuid"
    cypher = f"""
        MATCH (e:Entity {{ent_uuid:$e}}),
              (t:{target_label} {{{key}:$t}})
        MERGE (e)-[m:MENTIONS]->(t)
        REMOVE m.expiration         // strip TTL—this is now long-term data
    """
    tx.run(cypher, e=ent_uuid, t=target_uuid)


##############################################################################
# Promotion logic - copy one sub-graph in three passes (docs, paras, entities)
##############################################################################


def _promote_one(doc_node, stx: Transaction, ltx: Transaction) -> None:
    """
    Copy *one* document and its neighbourhood from the short-term cache to the
    long-term store in a single, idempotent transaction pair.
    """
    doc_props = dict(doc_node)
    doc_uuid = doc_props["doc_uuid"]

    # --------------------------------------------------------------------- #
    # 1) Document
    # --------------------------------------------------------------------- #
    _merge_document(ltx, doc_props)

    # --------------------------------------------------------------------- #
    # 2) Paragraphs (and PART_OF edges)
    # --------------------------------------------------------------------- #
    for rec in stx.run(
        "MATCH (p:Paragraph)-[:PART_OF]->(:Document {doc_uuid:$du}) RETURN p",
        du=doc_uuid,
    ):
        _merge_paragraph(ltx, dict(rec["p"]), doc_uuid)

    # --------------------------------------------------------------------- #
    # 3) Entities
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # 4) MENTIONS edges
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # 5) Mark the source document as promoted so we don't copy it again
    # --------------------------------------------------------------------- #
    stx.run(
        """
        MATCH (d:Document {doc_uuid:$du})
        SET   d.promoted   = true,
              d.promotedAt = timestamp()
        """,
        du=doc_uuid,
    )


##############################################################################
# Main entry-point
##############################################################################


def main() -> None:
    """
    Find every *unpromoted* (and optionally *validated*) document in the cache
    and copy its complete sub-graph into the long-term store.
    """
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
            # Separate write transactions per DB keep ACID guarantees intact.
            with s_sess.begin_transaction() as stx, l_sess.begin_transaction() as ltx:
                _promote_one(doc, stx, ltx)
                stx.commit()
                ltx.commit()
            promoted += 1

    print(f"✅  Promotion complete - {promoted} document(s) copied.")


if __name__ == "__main__":
    main()
