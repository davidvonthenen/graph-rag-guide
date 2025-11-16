#!/usr/bin/env python3
"""
Synchronous NER + promotion service for the community Graph RAG demo.

- Extracts entities with spaCy and normalises them for downstream matching.
- When ``promote`` is requested, the service copies the supporting subgraph
  from the authoritative **long-term** Neo4j instance into the **short-term**
  cache, applying the requested TTL to the relationships it touches.
- Provides lightweight health reporting so shell scripts can block until the
  worker is ready.

Endpoints
---------
GET  /health
POST /ner
    Request (JSON):
        {
          "text": "Your input text...",
          "labels": ["PERSON","ORG","GPE"],   # optional override of allowed labels
          "promote": true,                    # optional, defaults to true
          "ttl_ms": 86400000                  # optional TTL override for promotion
        }
    Response (JSON):
        {
          "text": "...",
          "model": "en_core_web_sm",
          "entities": ["openai", "san francisco"],
          "entity_pairs": [{"name": "openai", "label": "ORG"}, ...],
          "promotion": {"enabled": true, "promoted": 2, "ttl_ms": 86400000},
          "request_id": "..."
        }

Run
---
$ export SPACY_MODEL=en_core_web_sm
$ pip install flask spacy neo4j requests
$ python -m spacy download en_core_web_sm
$ python ner_service.py
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

from flask import Flask, jsonify, request
from functools import lru_cache
from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError
import spacy

# ---------------------------
# Configuration helpers
# ---------------------------

def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _scheme(ssl: bool) -> str:
    return "https" if ssl else "http"

def _base_url(host: str, port: int, ssl: bool) -> str:
    return f"{_scheme(ssl)}://{host}:{port}"

def _basic_auth(user: str, pwd: str):
    return (user, pwd) if (user and pwd) else None

# ---------------------------
# Environment
# ---------------------------

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# HOT (dest) - how THIS service reaches it
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "127.0.0.1")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9201"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")
OPENSEARCH_SSL = _as_bool(os.getenv("OPENSEARCH_SSL"), False)
OPENSEARCH_VERIFY_SSL = _as_bool(os.getenv("OPENSEARCH_VERIFY_SSL"), True)

# Neo4j destinations for promotion
LONG_NEO4J_URI = os.getenv("LONG_NEO4J_URI", "bolt://localhost:7688")
LONG_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
LONG_NEO4J_PASSWORD = os.getenv("LONG_NEO4J_PASSWORD", "neo4jneo4j1")

SHORT_NEO4J_URI = os.getenv("SHORT_NEO4J_URI", "bolt://localhost:7689")
SHORT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
SHORT_NEO4J_PASSWORD = os.getenv("SHORT_NEO4J_PASSWORD", "neo4jneo4j2")

PROMOTE_DOCUMENT_NODES = _as_bool(os.getenv("PROMOTE_DOCUMENT_NODES"), True)
DEFAULT_PROMOTION_TTL_MS = int(
    os.getenv("PROMOTION_TTL_MS", str(24 * 60 * 60 * 1000))
)

# Default set of "interesting" entity types.
DEFAULT_INTERESTING_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    "LOC",
    "FAC",
}

# Deterministic namespace for UUIDv5 when the long-term node lacks ent_uuid
_UUID5_NAMESPACE = uuid.UUID("b0b00000-0000-4000-8000-0000000000e1")

# ---------------------------
# spaCy load
# ---------------------------

@lru_cache(maxsize=1)
def load_spacy():
    return spacy.load(SPACY_MODEL)

@lru_cache(maxsize=1)
def _long_driver():
    return GraphDatabase.driver(
        LONG_NEO4J_URI, auth=(LONG_NEO4J_USER, LONG_NEO4J_PASSWORD)
    )

@lru_cache(maxsize=1)
def _short_driver():
    return GraphDatabase.driver(
        SHORT_NEO4J_URI, auth=(SHORT_NEO4J_USER, SHORT_NEO4J_PASSWORD)
    )

nlp = load_spacy()

def _node_get(node, key, default=None):
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        return node[key]
    except (KeyError, TypeError):
        return default

def _rec_get(rec, key, default=None):
    try:
        return rec[key]
    except KeyError:
        return default

# ---------------------------
# Entity Extraction
# ---------------------------

def _extract_entities(
    nlp_obj: spacy.Language, text: str, allowed_labels: set[str]
) -> List[Tuple[str, str]]:
    doc = nlp_obj(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in allowed_labels and len(ent.text.strip()) >= 3
    ]

def _extract_normalized_entities(
    nlp_obj: spacy.Language, text: str, allowed_labels: set[str]
) -> List[str]:
    ent_pairs = _extract_entities(nlp_obj, text, allowed_labels)
    seen = set()
    normalized: List[str] = []
    for name, _label in ent_pairs:
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized

# ---------------------------
# Flask App (synchronous path)
# ---------------------------

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": SPACY_MODEL,
        "dest": {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
            "ssl": OPENSEARCH_SSL,
        },
    }), 200

# ---- Long-term read (projects compact maps, one row per doc target) ----
PROMOTION_QUERY = """
MATCH (e:Entity {name:$name, label:$label})-[m:MENTIONS]->(t)
WHERE  m.expiration IS NULL
   OR  m.expiration = 0
   OR  m.expiration > $now
OPTIONAL MATCH (t)-[:PART_OF]->(d:Document)
WITH e,
     t,
     COALESCE(d, t) AS doc_node
WITH e { .ent_uuid, .name, .label } AS entity,
     doc_node,
     CASE WHEN doc_node IS NULL THEN NULL
          ELSE doc_node { .doc_uuid, .title, .content, .category }
     END AS doc
OPTIONAL MATCH (doc_node)<-[:PART_OF]-(p:Paragraph)
WITH entity,
     doc,
     collect(DISTINCT p { .para_uuid, .text, .index, .doc_uuid }) AS paras
RETURN entity, doc, paras
"""

# ---- Short-term write (single roundtrip, no importing-WITH violations) ----
CACHE_PROMOTION_QUERY = """
MERGE (e:Entity {ent_uuid:$entity.ent_uuid})
ON CREATE SET e.name=$entity.name,
              e.label=$entity.label,
              e.expiration=$exp
SET e.expiration=$exp,
    e.name = coalesce(e.name, $entity.name),
    e.label = coalesce(e.label, $entity.label)
WITH e
CALL {
    WITH e
    UNWIND (CASE WHEN $promote_documents THEN coalesce($documents, []) ELSE [] END) AS doc
    WITH e, doc
    WHERE doc.doc_uuid IS NOT NULL
    MERGE (d:Document {doc_uuid:doc.doc_uuid})
    ON CREATE SET d.title=doc.title,
                  d.content=doc.content,
                  d.category=doc.category,
                  d.expiration=$exp
    SET d.expiration=$exp
    MERGE (e)-[md:MENTIONS]->(d)
    SET md.expiration=$exp
    RETURN count(*) AS _
}
WITH e
CALL {
    WITH e
    UNWIND coalesce($paragraphs, []) AS para
    WITH e, para
    WHERE para.para_uuid IS NOT NULL
    MERGE (p:Paragraph {para_uuid:para.para_uuid})
    ON CREATE SET p.text=para.text,
                  p.index=para.index,
                  p.doc_uuid=para.doc_uuid,
                  p.expiration=$exp
    SET p.expiration=$exp
    WITH e, para, p
    OPTIONAL MATCH (d:Document {doc_uuid:para.doc_uuid})
    WITH e, p, d
    WHERE d IS NOT NULL
    MERGE (p)-[:PART_OF]->(d)
    WITH e, p
    MERGE (e)-[mp:MENTIONS]->(p)
    SET mp.expiration=$exp
    RETURN count(*) AS _
}
RETURN e.ent_uuid AS ent_uuid
"""

def _deterministic_ent_uuid(name: str, label: str, ent_uuid: Optional[str]) -> str:
    if ent_uuid:
        return ent_uuid
    seed = f"{label.strip().upper()}|{name.strip().lower()}"
    return str(uuid.uuid5(_UUID5_NAMESPACE, seed))

def _normalize_entity(node: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not node:
        return None
    name = _node_get(node, "name")
    label = _node_get(node, "label")
    if not name or not label:
        return None
    ent_uuid = _deterministic_ent_uuid(name, label, _node_get(node, "ent_uuid"))
    return {"ent_uuid": ent_uuid, "name": name, "label": label}

def _normalize_document(node: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not node:
        return None
    doc_uuid = _node_get(node, "doc_uuid")
    if not doc_uuid:
        return None
    return {
        "doc_uuid": doc_uuid,
        "title": _node_get(node, "title", ""),
        "content": _node_get(node, "content", ""),
        "category": _node_get(node, "category", ""),
    }

def _normalize_paragraphs(paragraph_nodes: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not paragraph_nodes:
        return normalized
    for node in paragraph_nodes:
        if not node:
            continue
        para_uuid = _node_get(node, "para_uuid")
        if not para_uuid:
            continue  # avoid MERGE with NULL key
        normalized.append(
            {
                "para_uuid": para_uuid,
                "text": _node_get(node, "text", ""),
                "index": _node_get(node, "index", 0),
                "doc_uuid": _node_get(node, "doc_uuid"),
            }
        )
    return normalized

def _write_promotion_to_cache(
    sess: Session,
    entity: Dict[str, Any],
    documents: List[Dict[str, Any]],
    paragraphs: List[Dict[str, Any]],
    exp_ts: int,
) -> None:
    result = sess.run(
        CACHE_PROMOTION_QUERY,
        entity=entity,
        documents=documents,
        paragraphs=paragraphs,
        exp=exp_ts,
        promote_documents=PROMOTE_DOCUMENT_NODES,
    )
    result.consume()  # surface server-side errors now

def _calculate_expiration(now_ms: int, ttl_ms: int | None) -> int:
    if ttl_ms is None:
        ttl_ms = DEFAULT_PROMOTION_TTL_MS
    if ttl_ms <= 0:
        return 0
    return now_ms + ttl_ms

def _promote_entity(
    name: str,
    label: str,
    long_sess: Session,
    short_sess: Session,
    now_ms: int,
    exp_ts: int,
) -> bool:
    records = list(long_sess.run(PROMOTION_QUERY, name=name, label=label, now=now_ms))
    if not records:
        return False

    entity_payload = _normalize_entity(_rec_get(records[0], "entity"))
    if not entity_payload:
        return False

    documents: Dict[str, Dict[str, Any]] = {}
    paragraphs: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        doc_payload = _normalize_document(_rec_get(rec, "doc"))
        if doc_payload:
            documents[doc_payload["doc_uuid"]] = doc_payload

        for para in _normalize_paragraphs(_rec_get(rec, "paras", [])):
            key = para["para_uuid"]
            if key not in paragraphs:
                paragraphs[key] = para

    _write_promotion_to_cache(
        short_sess,
        entity_payload,
        list(documents.values()),
        list(paragraphs.values()),
        exp_ts,
    )
    return True

def _promote_entities(entity_pairs: List[Tuple[str, str]], ttl_ms: int | None) -> int:
    if not entity_pairs:
        return 0

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for name, label in entity_pairs:
        key = (name.strip().lower(), label.strip().upper())
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            deduped.append(key)

    if not deduped:
        return 0

    now_ms = int(time.time() * 1000)
    exp_ts = _calculate_expiration(now_ms, ttl_ms)

    promoted = 0
    with _long_driver().session() as long_sess, _short_driver().session() as short_sess:
        for name, label in deduped:
            try:
                if _promote_entity(name, label, long_sess, short_sess, now_ms, exp_ts):
                    promoted += 1
            except Neo4jError as exc:
                print(f"[neo4j error promoting ({name}, {label})] {exc}")
                raise
    return promoted

@app.route("/ner", methods=["POST"])
def ner():
    data = request.get_json(silent=True)
    if not data or "text" not in data or not isinstance(data["text"], str):
        return (
            jsonify(
                {
                    "error": "Invalid request",
                    "detail": "Expected JSON with a 'text' string field.",
                }
            ),
            400,
        )

    text: str = data["text"]
    labels_field = data.get("labels")
    if isinstance(labels_field, list) and labels_field:
        allowed = {str(l).upper() for l in labels_field}
    else:
        allowed = DEFAULT_INTERESTING_ENTITY_TYPES

    promote_flag = _as_bool(data.get("promote"), True)

    ttl_field = data.get("ttl_ms")
    if ttl_field is None:
        ttl_ms: int | None = DEFAULT_PROMOTION_TTL_MS
    else:
        try:
            ttl_ms = int(ttl_field)
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "error": "invalid_ttl",
                        "detail": "ttl_ms must be an integer",
                    }
                ),
                400,
            )

    entity_pairs = _extract_entities(nlp, text, allowed)
    normalized_entities: List[str] = []
    seen_names = set()
    for name, _label in entity_pairs:
        if name not in seen_names:
            seen_names.add(name)
            normalized_entities.append(name)

    print(
        f"[{datetime.now(timezone.utc).isoformat()}] /ner called, {len(text)} chars, {len(normalized_entities)} entities"
    )
    print(f"[entities discovered] {normalized_entities}\n")

    request_id = str(uuid.uuid4())
    status_code = 200
    promotion_info: Dict[str, Any] = {
        "enabled": promote_flag,
        "ttl_ms": ttl_ms if promote_flag else None,
        "promoted": 0,
    }

    if promote_flag:
        try:
            promotion_info["promoted"] = _promote_entities(entity_pairs, ttl_ms)
        except Neo4jError as exc:
            status_code = 500
            promotion_info["error"] = "neo4j_error"
            promotion_info["detail"] = str(exc)
        except Exception as exc:  # pragma: no cover
            status_code = 500
            promotion_info["error"] = type(exc).__name__
            promotion_info["detail"] = str(exc)

    payload = {
        "text": text,
        "model": SPACY_MODEL,
        "entities": normalized_entities,
        "entity_pairs": [{"name": name, "label": label} for name, label in entity_pairs],
        "promotion": promotion_info,
        "request_id": request_id,
    }

    return jsonify(payload), status_code

# ---------------------------
# Local Dev Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)
