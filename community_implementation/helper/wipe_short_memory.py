#!/usr/bin/env python3
"""
wipe_short_memory.py – Wipe all data from SHORT TERM Neo4j databases
"""

import os
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase

###############################################################################
# Configuration
###############################################################################

SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

###############################################################################
# Main
###############################################################################

def main() -> None:
    with GraphDatabase.driver(SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD)) as driver, driver.session() as session:
        # Clean slate
        print("Clearing old data from Neo4j …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.\n")


if __name__ == "__main__":
    main()
