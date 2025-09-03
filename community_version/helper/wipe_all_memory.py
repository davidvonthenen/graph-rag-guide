# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
wipe_all_memory.py - Wipe all data from Neo4j databases
"""

import os
import uuid
from pathlib import Path

import spacy
from neo4j import GraphDatabase

###############################################################################
# Configuration
###############################################################################

LONG_NEO4J_URI = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
LONG_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
LONG_NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

###############################################################################
# Main
###############################################################################

def main() -> None:
    with GraphDatabase.driver(LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD)) as driver, driver.session() as session:
        # Clean slate
        print("Clearing old data from Neo4j …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.\n")

    with GraphDatabase.driver(SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD)) as driver, driver.session() as session:
        # Clean slate
        print("Clearing old data from Neo4j …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.\n")


if __name__ == "__main__":
    main()
