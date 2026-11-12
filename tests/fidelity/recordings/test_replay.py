"""Replay of recorded fidelity fixtures against the stored article (local data only)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from artigo_env.core.paths import ENV_VAR, data_path
from tools.calibration.fixtures import load_fixtures

VERSION = "cal-v1"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

_FIXTURES = load_fixtures(VERSION, FIXTURES_DIR)
_PARAMS = [pytest.param(item, id=str(item["source_session"])) for item in _FIXTURES] or [
    pytest.param(None, id="no-fixtures")
]


@pytest.mark.local
@pytest.mark.req("REQ-CAL-003", "REQ-TST-003")
@pytest.mark.parametrize("fixture", _PARAMS)
def test_fixture_replay(fixture: dict[str, Any] | None) -> None:
    if fixture is None:
        pytest.skip("no fixtures recorded yet")
    if not os.environ.get(ENV_VAR, "").strip():
        pytest.skip(f"{ENV_VAR} is not set; the article text lives outside the repository")
    sha256 = fixture["article_ref"]["sha256"]
    article_file = data_path("calibration", VERSION, "articles", f"{sha256}.json")
    if not article_file.is_file():
        pytest.skip(f"article {sha256} not found under the {VERSION} calibration data")
    article = json.loads(article_file.read_text(encoding="utf-8"))
    assert hashlib.sha256(article["text"].encode("utf-8")).hexdigest() == sha256
    # TODO(REQ-ENG-013, phase F2): replay fixture["guesses"] through the engine and compare
    # counted/count/revealed_token_indices/message with the recorded expectations.
