"""
evict_expired_short_term.py  –  Short-term memory TTL garbage-collector
===========================================================

Deletes any *nodes* **or** *relationships* in the short-term Neo4j
instance whose `expiration` property is a POSIX timestamp (milliseconds)
older than **now**.

Usage
-----

    $ python evict_expired_short_term.py           # uses env vars below

Environment
-----------

    NEO4J_URI        (default bolt://localhost:7687)
    NEO4J_USER       (default neo4j)
    NEO4J_PASSWORD   (default neo4j)

A cron job (e.g. hourly) keeps the short-term graph small:

    0 * * * * /usr/bin/python /path/to/evict_expired_short_term.py >> /var/log/rag_gc.log
"""

import os
import time
from neo4j import GraphDatabase
from typing import Tuple

###############################################################################
# Helpers
###############################################################################

def connect() -> GraphDatabase.driver:
    uri  = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("SHORT_NEO4J_USER", "neo4j")
    pw   = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j")
    return GraphDatabase.driver(uri, auth=(user, pw))


def purge_expired(tx, now_ms: int) -> Tuple[int, int]:
    """Detach-delete all expired nodes; delete expired relationships first.

    Returns (rels_deleted, nodes_deleted).
    """
    # 1) delete expired relationships
    rel_res = tx.run("""
        MATCH ()-[r]-()
        WHERE r.expiration < $now
        WITH r LIMIT 10000   // keep tx small for large graphs
        DELETE r
        RETURN count(r) AS cnt
    """, now=now_ms).single()
    rel_cnt = rel_res["cnt"]

    # 2) detach-delete expired nodes
    node_res = tx.run("""
        MATCH (n)
        WHERE n.expiration < $now
        WITH n LIMIT 10000
        DETACH DELETE n
        RETURN count(n) AS cnt
    """, now=now_ms).single()
    node_cnt = node_res["cnt"]

    return rel_cnt, node_cnt


###############################################################################
# Main
###############################################################################

def main() -> None:
    driver = connect()
    total_rels = total_nodes = 0

    # keep running until no more expired entities remain
    with driver.session() as sess:
        while True:
            rels, nodes = sess.write_transaction(purge_expired, int(time.time() * 1000))
            total_rels  += rels
            total_nodes += nodes
            if rels == 0 and nodes == 0:
                break   # nothing left to purge

    print(f"Expired relationships deleted: {total_rels}")
    print(f"Expired nodes deleted:         {total_nodes}")


if __name__ == "__main__":
    main()
