"""Unit tests for ner_service helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Ensure the repository root is importable when tests run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

with patch("spacy.load", return_value=Mock()):  # noqa: SIM117 - explicit context keeps scope tight
    from community_version import ner_service  # noqa: E402


def test_fetch_promotion_records_falls_back_when_label_mismatch():
    session = Mock()
    session.run.side_effect = [
        [],
        [{"e": {"ent_uuid": "123"}}],
    ]

    records = ner_service._fetch_promotion_records(session, "windsurf", "ORG", now_ms=42)

    assert records == [{"e": {"ent_uuid": "123"}}]
    assert session.run.call_count == 2
    assert session.run.call_args_list[0].kwargs["label"] == "ORG"
    assert session.run.call_args_list[1].kwargs["label"] is None


def test_fetch_promotion_records_respects_initial_matches():
    session = Mock()
    session.run.return_value = [{"e": {"ent_uuid": "999"}}]

    records = ner_service._fetch_promotion_records(session, "windsurf", "ORG", now_ms=99)

    assert records == [{"e": {"ent_uuid": "999"}}]
    session.run.assert_called_once()
    assert session.run.call_args.kwargs["label"] == "ORG"
