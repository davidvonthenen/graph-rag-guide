# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
evict_expired_short_term.py ― TTL garbage-collector for short-term memory
===========================================================================

Purpose
-------
Remove any *relationships* **and** *nodes* from the short-term graph whose
`expiration` property (stored as a POSIX timestamp in **milliseconds**) is
earlier than the current time.

The script is safe to run as a cron job. It loops in small batches so a
very large graph cannot starve the transaction log.

Typical usage
-------------
    $ python evict_expired_short_term.py            # reads env vars below

Environment variables (override defaults as needed)
---------------------------------------------------
SHORT_NEO4J_URI        bolt://localhost:7687
SHORT_NEO4J_USER       neo4j
SHORT_NEO4J_PASSWORD   neo4jneo4j

Example cron entry (hourly run with basic logging)
--------------------------------------------------
    0 * * * * /usr/bin/python /path/to/evict_expired_short_term.py \
              >> /var/log/rag_gc.log
"""

import time
from typing import Tuple

from neo4j import Driver

from common import build_driver

###############################################################################
# Connection helpers
###############################################################################

def connect() -> Driver:
    """
    Build a Neo4j driver for the short-term memory store.

    Returns
    -------
    neo4j.GraphDatabase.driver
        Ready-to-use driver instance configured from environment variables.
    """
    return build_driver(
        "SHORT_NEO4J_URI",
        "SHORT_NEO4J_USER",
        "SHORT_NEO4J_PASSWORD",
        "bolt://localhost:7687",
        default_password="neo4jneo4j",
    )


###############################################################################
# Core eviction logic
###############################################################################

def purge_expired(tx, now_ms: int) -> Tuple[int, int]:
    """
    Delete expired relationships first, then detach-delete expired nodes.

    The two-step approach avoids dangling relationships and keeps each
    transaction small by limiting deletions to 10 000 entities at a time.

    Parameters
    ----------
    tx : neo4j.Workspace
        Active transaction context.
    now_ms : int
        Current time in milliseconds since epoch.

    Returns
    -------
    Tuple[int, int]
        (number_of_relationships_deleted, number_of_nodes_deleted)
    """

    # -------- Step 1: delete expired relationships -------------------------
    rel_res = tx.run(
        """
        MATCH ()-[r]-()
        WHERE r.expiration < $now
        WITH r LIMIT 10000          // cap batch size for large graphs
        DELETE r
        RETURN count(r) AS cnt
        """,
        now=now_ms,
    ).single()
    rel_cnt = rel_res["cnt"]

    # -------- Step 2: delete expired nodes (with all remaining rels) -------
    node_res = tx.run(
        """
        MATCH (n)
        WHERE n.expiration < $now
        WITH n LIMIT 10000
        DETACH DELETE n             // removes n and any attached rels
        RETURN count(n) AS cnt
        """,
        now=now_ms,
    ).single()
    node_cnt = node_res["cnt"]

    return rel_cnt, node_cnt


###############################################################################
# Script entry-point
###############################################################################

def main() -> None:
    """Run the garbage-collection loop until nothing remains to purge."""
    driver = connect()
    total_rels = total_nodes = 0

    # Keep purging in batches until both queries return zero.
    with driver.session() as sess:
        while True:
            rels, nodes = sess.write_transaction(
                purge_expired,
                int(time.time() * 1000)  # current time in ms
            )
            total_rels  += rels
            total_nodes += nodes
            if rels == 0 and nodes == 0:
                break   # graph is clean; exit loop

    # Simple stdout metrics for cron logs or pipeline monitors.
    print(f"Expired relationships deleted: {total_rels}")
    print(f"Expired nodes deleted:         {total_nodes}")


if __name__ == "__main__":
    main()
