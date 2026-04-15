"""Hermetic unit tests for ``tests.eval.llm_judge.LLMJudge``.

No live Anthropic API calls are made — every test either constructs
the judge with an injected mock client or patches ``os.environ`` to
hide the API key entirely. Phase 6 is responsible for the real
live-API smoke test; Sprint 3 only pins the parsing + error contracts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Make the Fusion MCP repo root importable so ``tests.eval.*`` resolves
# when pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval.llm_judge import DEFAULT_MODEL, LLMJudge  # noqa: E402


def _fake_response(text: str) -> Any:
    """Build a minimal object that mimics ``anthropic`` response shape."""
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block])


def _mock_client_returning(text: str) -> MagicMock:
    """Return a mock anthropic client whose .messages.create -> text."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_response(text)
    return mock_client


# ---------------------------------------------------------------------------
# Constructor contracts
# ---------------------------------------------------------------------------


def test_judge_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMJudge must raise RuntimeError if ANTHROPIC_API_KEY is not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        LLMJudge()
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_judge_accepts_injected_client_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injecting a client bypasses the API-key constructor check.

    This is the unit-test escape hatch: tests never hit the real
    anthropic SDK, so the constructor should not demand a key when a
    client is supplied explicitly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = MagicMock()
    judge = LLMJudge(client=client)
    assert judge.config.model == DEFAULT_MODEL
    assert judge.config.temperature == 0.0


def test_judge_defaults_are_pinned() -> None:
    """Pin the defaults so a drift in the class shows up loudly here."""
    client = MagicMock()
    judge = LLMJudge(client=client)
    assert judge.config.model == "claude-sonnet-4-6"
    assert judge.config.temperature == 0.0
    assert judge.config.max_tokens == 256


# ---------------------------------------------------------------------------
# judge_relevance happy path
# ---------------------------------------------------------------------------


def test_judge_relevance_parses_valid_json() -> None:
    """A clean JSON response parses into the standard return shape."""
    response_text = json.dumps(
        {"score": 0.8, "reasoning": "Partial topical overlap on strategy baseline."}
    )
    client = _mock_client_returning(response_text)
    judge = LLMJudge(client=client)

    result = judge.judge_relevance(
        query="What is the NovaTrade strategy baseline?",
        candidate_memory={
            "id": "fx-01",
            "content": "Decision: NovaTrade baseline is Rob Hoffman IRB v5.",
        },
    )

    assert result["score"] == 0.8
    assert result["reasoning"] == "Partial topical overlap on strategy baseline."
    assert result["model"] == DEFAULT_MODEL
    assert isinstance(result["timestamp"], str) and "T" in result["timestamp"]

    # Exactly one prompt built and sent.
    assert client.messages.create.call_count == 1
    call = client.messages.create.call_args
    assert call.kwargs["model"] == DEFAULT_MODEL
    assert call.kwargs["temperature"] == 0.0
    assert call.kwargs["max_tokens"] == 256
    prompt = call.kwargs["messages"][0]["content"]
    assert "What is the NovaTrade strategy baseline?" in prompt
    assert "Rob Hoffman IRB v5" in prompt


def test_judge_relevance_extracts_text_field_fallback() -> None:
    """If candidate has 'text' but not 'content', it is still extracted."""
    client = _mock_client_returning(
        json.dumps({"score": 0.5, "reasoning": "tangential"})
    )
    judge = LLMJudge(client=client)

    judge.judge_relevance(
        query="test query",
        candidate_memory={"id": "x", "text": "fallback-text-payload"},
    )

    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "fallback-text-payload" in prompt


def test_judge_relevance_accepts_score_at_boundaries() -> None:
    """Scores of exactly 0.0 and 1.0 are valid."""
    client = _mock_client_returning(
        json.dumps({"score": 0.0, "reasoning": "irrelevant"})
    )
    judge = LLMJudge(client=client)
    result = judge.judge_relevance("q", {"id": "x", "content": "c"})
    assert result["score"] == 0.0

    client2 = _mock_client_returning(
        json.dumps({"score": 1.0, "reasoning": "perfect"})
    )
    judge2 = LLMJudge(client=client2)
    result2 = judge2.judge_relevance("q", {"id": "x", "content": "c"})
    assert result2["score"] == 1.0


# ---------------------------------------------------------------------------
# judge_relevance error contracts
# ---------------------------------------------------------------------------


def test_judge_relevance_raises_on_invalid_json() -> None:
    """A non-JSON response must raise ValueError with the raw text."""
    client = _mock_client_returning("this is not json at all")
    judge = LLMJudge(client=client)

    with pytest.raises(ValueError) as excinfo:
        judge.judge_relevance("q", {"id": "x", "content": "c"})
    # Raw text is surfaced in the error message for debugging.
    assert "this is not json at all" in str(excinfo.value)


def test_judge_relevance_raises_on_missing_score_key() -> None:
    """A JSON object without a 'score' key must raise ValueError."""
    client = _mock_client_returning(json.dumps({"reasoning": "no score!"}))
    judge = LLMJudge(client=client)

    with pytest.raises(ValueError) as excinfo:
        judge.judge_relevance("q", {"id": "x", "content": "c"})
    assert "score" in str(excinfo.value).lower()


def test_judge_relevance_raises_on_non_numeric_score() -> None:
    """A non-numeric score must raise ValueError."""
    client = _mock_client_returning(
        json.dumps({"score": "high", "reasoning": "oops"})
    )
    judge = LLMJudge(client=client)

    with pytest.raises(ValueError):
        judge.judge_relevance("q", {"id": "x", "content": "c"})


def test_judge_relevance_raises_on_out_of_range_score() -> None:
    """A score > 1.0 or < 0.0 must raise ValueError."""
    client = _mock_client_returning(
        json.dumps({"score": 1.5, "reasoning": "off the chart"})
    )
    judge = LLMJudge(client=client)
    with pytest.raises(ValueError):
        judge.judge_relevance("q", {"id": "x", "content": "c"})

    client2 = _mock_client_returning(
        json.dumps({"score": -0.1, "reasoning": "negative"})
    )
    judge2 = LLMJudge(client=client2)
    with pytest.raises(ValueError):
        judge2.judge_relevance("q", {"id": "x", "content": "c"})


def test_judge_relevance_raises_on_non_object_json() -> None:
    """A bare JSON number or array must raise — we expect an object."""
    client = _mock_client_returning("0.8")
    judge = LLMJudge(client=client)
    with pytest.raises(ValueError):
        judge.judge_relevance("q", {"id": "x", "content": "c"})


# ---------------------------------------------------------------------------
# Content truncation + extraction
# ---------------------------------------------------------------------------


def test_candidate_content_is_truncated_to_1000_chars() -> None:
    """Long candidate content is clipped before being sent to the judge."""
    client = _mock_client_returning(
        json.dumps({"score": 0.5, "reasoning": "ok"})
    )
    judge = LLMJudge(client=client)

    long_content = "A" * 5000
    judge.judge_relevance("q", {"id": "x", "content": long_content})

    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    # The full 5000-char block is NOT in the prompt.
    assert "A" * 5000 not in prompt
    # But the first 1000 chars are.
    assert "A" * 1000 in prompt
