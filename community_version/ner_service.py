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
          "promote": true,                      # optional, defaults to true
          "ttl_ms": 86400000                    # optional TTL override for promotion
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

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


@dataclass
class PromotionAudit:
    """Records what was copied into the cache for a single entity promotion."""

    entity_uuid: str | None = None
    entity_name: str | None = None
    entity_label: str | None = None
    documents: Set[str] = field(default_factory=set)
    paragraphs: Set[str] = field(default_factory=set)
    doc_mentions: Set[str] = field(default_factory=set)
    para_mentions: Set[str] = field(default_factory=set)
    part_of_edges: Set[Tuple[str, str]] = field(default_factory=set)
    cleared_mentions: bool = False
    skipped_reason: str | None = None


def _log_promotion_audit(audit: PromotionAudit) -> None:
    """Emit a concise, human-readable summary of a promotion audit."""

    if not audit.entity_uuid:
        reason = audit.skipped_reason or "unknown"
        print(f"[promotion audit] skipped entity promotion: {reason}")
        return

    doc_count = len(audit.documents)
    para_count = len(audit.paragraphs)
    doc_mentions = len(audit.doc_mentions)
    para_mentions = len(audit.para_mentions)
    part_of = len(audit.part_of_edges)

    print(
        "[promotion audit] entity=%s label=%s docs=%d paras=%d doc_mentions=%d "
        "para_mentions=%d part_of_edges=%d cleared_mentions=%s"
        % (
            audit.entity_name or audit.entity_uuid,
            audit.entity_label or "?",
            doc_count,
            para_count,
            doc_mentions,
            para_mentions,
            part_of,
            audit.cleared_mentions,
        )
    )

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

def _extract_normalized_entities(nlp_obj: spacy.Language, text: str, allowed_labels: set[str]) -> List[str]:
    ent_pairs = _extract_entities(nlp_obj, text, allowed_labels)

    # Normalize: lowercase, dedupe while preserving order
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

# @app.before_first_request
# def _startup():
#     _ensure_worker_started()

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

PROMOTION_QUERY = """
MATCH (e:Entity {name:$name, label:$label})-[m:MENTIONS]->(t)
WHERE  m.expiration IS NULL
   OR  m.expiration = 0
   OR  m.expiration > $now
OPTIONAL MATCH (t)-[:PART_OF]->(d:Document)
WITH e, COALESCE(d, t) AS doc
OPTIONAL MATCH (doc)<-[:PART_OF]-(p:Paragraph)
WITH e, doc, collect(DISTINCT p) AS paras
RETURN e, doc, paras
"""


def _merge_entity(
    sess: Session, ent_uuid: str, name: str, label: str, exp_ts: int
) -> None:
    sess.run(
        """
        MERGE (e:Entity {ent_uuid:$uuid})
        ON CREATE SET e.name=$name,
                      e.label=$label,
                      e.expiration=$exp
        SET e.expiration=$exp
        """,
        uuid=ent_uuid,
        name=name,
        label=label,
        exp=exp_ts,
    )


def _merge_paragraph(sess: Session, para_node, exp_ts: int) -> None:
    sess.run(
        """
        MERGE (p:Paragraph {para_uuid:$uuid})
        ON CREATE SET p.text=$text,
                      p.index=$idx,
                      p.doc_uuid=$doc_uuid,
                      p.expiration=$exp
        SET p.expiration=$exp
        """,
        uuid=_node_get(para_node, "para_uuid"),
        text=_node_get(para_node, "text", ""),
        idx=_node_get(para_node, "index", 0),
        doc_uuid=_node_get(para_node, "doc_uuid"),
        exp=exp_ts,
    )


def _merge_document(sess: Session, doc_node, exp_ts: int) -> None:
    sess.run(
        """
        MERGE (d:Document {doc_uuid:$uuid})
        ON CREATE SET d.title=$title,
                      d.content=$content,
                      d.category=$category,
                      d.expiration=$exp
        SET d.expiration=$exp
        """,
        uuid=_node_get(doc_node, "doc_uuid"),
        title=_node_get(doc_node, "title", ""),
        content=_node_get(doc_node, "content", ""),
        category=_node_get(doc_node, "category", ""),
        exp=exp_ts,
    )


def _merge_part_of(sess: Session, para_uuid: str, doc_uuid: str) -> None:
    sess.run(
        """
        MATCH (p:Paragraph {para_uuid:$p}), (d:Document {doc_uuid:$d})
        MERGE (p)-[:PART_OF]->(d)
        """,
        p=para_uuid,
        d=doc_uuid,
    )


def _merge_mentions(
    sess: Session,
    ent_uuid: str,
    target_label: str,
    target_id_name: str,
    target_id_value: str,
    exp_ts: int,
) -> None:
    sess.run(
        f"""
        MATCH (e:Entity {{ent_uuid:$e_uuid}}),
              (t:{target_label} {{{target_id_name}:$tid}})
        MERGE (e)-[m:MENTIONS]->(t)
        SET   m.expiration=$exp
        """,
        e_uuid=ent_uuid,
        tid=target_id_value,
        exp=exp_ts,
    )


def _clear_mentions(sess: Session, ent_uuid: str, include_documents: bool) -> None:
    if include_documents:
        sess.run(
            """
            MATCH (e:Entity {ent_uuid:$uuid})-[m:MENTIONS]->(t)
            WHERE t:Paragraph OR t:Document
            DELETE m
            """,
            uuid=ent_uuid,
        )
    else:
        sess.run(
            """
            MATCH (e:Entity {ent_uuid:$uuid})-[m:MENTIONS]->(t:Paragraph)
            DELETE m
            """,
            uuid=ent_uuid,
        )


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
) -> PromotionAudit:
    audit = PromotionAudit()

    records = list(long_sess.run(PROMOTION_QUERY, name=name, label=label, now=now_ms))
    if not records:
        audit.skipped_reason = "no_records"
        _log_promotion_audit(audit)
        return audit

    primary = records[0]["e"]
    ent_uuid = _node_get(primary, "ent_uuid")
    if not ent_uuid:
        audit.skipped_reason = "missing_ent_uuid"
        _log_promotion_audit(audit)
        return audit

    ent_name = _node_get(primary, "name", name)
    ent_label = _node_get(primary, "label", label)

    audit.entity_uuid = ent_uuid
    audit.entity_name = ent_name
    audit.entity_label = ent_label

    _merge_entity(short_sess, ent_uuid, ent_name, ent_label, exp_ts)
    _clear_mentions(short_sess, ent_uuid, include_documents=PROMOTE_DOCUMENT_NODES)
    audit.cleared_mentions = True

    seen_docs: Set[str] = set()
    seen_doc_mentions: Set[str] = set()
    seen_paragraphs: Set[str] = set()
    seen_para_mentions: Set[str] = set()
    part_of_edges: Set[Tuple[str, str]] = set()

    for rec in records:
        doc_node = rec["doc"]
        para_nodes = rec["paras"] or []

        if PROMOTE_DOCUMENT_NODES and doc_node:
            doc_uuid = _node_get(doc_node, "doc_uuid")
            if doc_uuid:
                if doc_uuid not in seen_docs:
                    _merge_document(short_sess, doc_node, exp_ts)
                    seen_docs.add(doc_uuid)
                    audit.documents.add(doc_uuid)
                if doc_uuid not in seen_doc_mentions:
                    _merge_mentions(
                        short_sess,
                        ent_uuid,
                        "Document",
                        "doc_uuid",
                        doc_uuid,
                        exp_ts,
                    )
                    seen_doc_mentions.add(doc_uuid)
                    audit.doc_mentions.add(doc_uuid)

        for para_node in para_nodes:
            para_uuid = _node_get(para_node, "para_uuid")
            if not para_uuid:
                continue

            if para_uuid not in seen_paragraphs:
                _merge_paragraph(short_sess, para_node, exp_ts)
                seen_paragraphs.add(para_uuid)
                audit.paragraphs.add(para_uuid)

            if PROMOTE_DOCUMENT_NODES:
                doc_uuid = _node_get(para_node, "doc_uuid")
                if doc_uuid and (para_uuid, doc_uuid) not in part_of_edges:
                    _merge_part_of(short_sess, para_uuid, doc_uuid)
                    part_of_edges.add((para_uuid, doc_uuid))
                    audit.part_of_edges.add((para_uuid, doc_uuid))

            if para_uuid not in seen_para_mentions:
                _merge_mentions(
                    short_sess,
                    ent_uuid,
                    "Paragraph",
                    "para_uuid",
                    para_uuid,
                    exp_ts,
                )
                seen_para_mentions.add(para_uuid)
                audit.para_mentions.add(para_uuid)

    _log_promotion_audit(audit)
    return audit


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

    with _long_driver().session() as long_sess, _short_driver().session() as short_sess:
        for name, label in deduped:
            _promote_entity(name, label, long_sess, short_sess, now_ms, exp_ts)

    return len(deduped)


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
        except Exception as exc:  # pragma: no cover - unexpected
            status_code = 500
            promotion_info["error"] = type(exc).__name__
            promotion_info["detail"] = str(exc)

    payload = {
        "text": text,
        "model": SPACY_MODEL,
        "entities": normalized_entities,
        "entity_pairs": [
            {"name": name, "label": label}
            for name, label in entity_pairs
        ],
        "promotion": promotion_info,
        "request_id": request_id,
    }

    return jsonify(payload), status_code

# ---------------------------
# Local Dev Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)
