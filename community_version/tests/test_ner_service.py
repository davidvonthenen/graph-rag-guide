"""Unit tests for key promotion helpers in ner_service."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure repository root is importable when tests run from project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

with patch("spacy.load", return_value=Mock()):  # Load a lightweight stub model
    from community_version import ner_service  # noqa: E402


def _record(ent_uuid: str, doc_uuid: str, para_uuid: str) -> dict:
    return {
        "e": {"ent_uuid": ent_uuid, "name": "google", "label": "ORG"},
        "doc": {"doc_uuid": doc_uuid, "title": doc_uuid.title()},
        "paras": [
            {"para_uuid": para_uuid, "doc_uuid": doc_uuid, "text": f"Text {para_uuid}"},
        ],
    }


def test_promote_entity_relinks_documents_and_paragraphs():
    long_session = Mock()
    long_session.run.return_value = [
        _record("e1", "doc1", "para1"),
        _record("e1", "doc2", "para2"),
    ]
    short_session = Mock()

    with (
        patch.object(ner_service, "_merge_entity") as mock_merge_entity,
        patch.object(ner_service, "_clear_mentions") as mock_clear,
        patch.object(ner_service, "_merge_document") as mock_merge_doc,
        patch.object(ner_service, "_merge_paragraph") as mock_merge_para,
        patch.object(ner_service, "_merge_part_of") as mock_merge_part,
        patch.object(ner_service, "_merge_mentions") as mock_merge_mentions,
        patch.object(ner_service, "_log_promotion_audit") as mock_log,
    ):
        audit = ner_service._promote_entity(
            name="google",
            label="ORG",
            long_sess=long_session,
            short_sess=short_session,
            now_ms=0,
            exp_ts=500,
        )

    mock_merge_entity.assert_called_once_with(short_session, "e1", "google", "ORG", 500)
    mock_clear.assert_called_once_with(short_session, "e1", include_documents=True)
    mock_log.assert_called_once()

    promoted_docs = {c.args[1]["doc_uuid"] for c in mock_merge_doc.call_args_list}
    assert promoted_docs == {"doc1", "doc2"}

    promoted_paras = {c.args[1]["para_uuid"] for c in mock_merge_para.call_args_list}
    assert promoted_paras == {"para1", "para2"}

    part_of_pairs = {(c.args[1], c.args[2]) for c in mock_merge_part.call_args_list}
    assert part_of_pairs == {("para1", "doc1"), ("para2", "doc2")}

    doc_mentions = {c.args[4] for c in mock_merge_mentions.call_args_list if c.args[2] == "Document"}
    para_mentions = {c.args[4] for c in mock_merge_mentions.call_args_list if c.args[2] == "Paragraph"}
    assert doc_mentions == {"doc1", "doc2"}
    assert para_mentions == {"para1", "para2"}

    assert audit.entity_uuid == "e1"
    assert audit.documents == {"doc1", "doc2"}
    assert audit.paragraphs == {"para1", "para2"}
    assert audit.doc_mentions == {"doc1", "doc2"}
    assert audit.para_mentions == {"para1", "para2"}
    assert audit.part_of_edges == {("para1", "doc1"), ("para2", "doc2")}
    assert audit.cleared_mentions is True


def test_promote_entity_skips_document_links_when_disabled():
    long_session = Mock()
    long_session.run.return_value = [_record("e2", "doc3", "para3")]
    short_session = Mock()

    with (
        patch.object(ner_service, "PROMOTE_DOCUMENT_NODES", False),
        patch.object(ner_service, "_merge_entity") as mock_merge_entity,
        patch.object(ner_service, "_clear_mentions") as mock_clear,
        patch.object(ner_service, "_merge_document") as mock_merge_doc,
        patch.object(ner_service, "_merge_paragraph") as mock_merge_para,
        patch.object(ner_service, "_merge_part_of") as mock_merge_part,
        patch.object(ner_service, "_merge_mentions") as mock_merge_mentions,
        patch.object(ner_service, "_log_promotion_audit") as mock_log,
    ):
        audit = ner_service._promote_entity(
            name="google",
            label="ORG",
            long_sess=long_session,
            short_sess=short_session,
            now_ms=0,
            exp_ts=900,
        )

    mock_merge_entity.assert_called_once_with(short_session, "e2", "google", "ORG", 900)
    mock_clear.assert_called_once_with(short_session, "e2", include_documents=False)
    mock_merge_doc.assert_not_called()
    mock_merge_part.assert_not_called()

    para_mentions = [c.args[4] for c in mock_merge_mentions.call_args_list if c.args[2] == "Paragraph"]
    assert para_mentions == ["para3"]
    doc_mentions = [c for c in mock_merge_mentions.call_args_list if c.args[2] == "Document"]
    assert doc_mentions == []

    promoted_paras = {c.args[1]["para_uuid"] for c in mock_merge_para.call_args_list}
    assert promoted_paras == {"para3"}
    mock_log.assert_called_once()

    assert audit.documents == set()
    assert audit.paragraphs == {"para3"}
    assert audit.doc_mentions == set()
    assert audit.para_mentions == {"para3"}
    assert audit.part_of_edges == set()
