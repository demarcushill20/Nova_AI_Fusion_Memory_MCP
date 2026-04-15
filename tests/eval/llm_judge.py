"""LLM-as-judge wrapper for PLAN-0759 Phase 6 associative-recall eval.

Design
------
This is the Sprint 3 scaffolding for the LLM-as-judge decision locked in
``/home/nova/nova-core/MEMORY/plans/PLAN-0759/eval_ground_truth_design.md``.
The judge grades ``(query, candidate_memory)`` pairs with a rubric and
returns a numeric relevance score in ``[0, 1]`` plus a short reasoning
string.

Pinning & reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~
- Model is pinned via the constructor's ``model`` argument (default
  ``"claude-sonnet-4-6"``). The ``judge_relevance`` return value echoes
  the model id so every saved score is traceable to the exact model
  version that produced it.
- Temperature is pinned to 0 by default — any drift should then be
  attributable to model revisions, not sampling noise.
- The wrapper does **not** hardcode any API key. The constructor raises
  ``RuntimeError`` if ``ANTHROPIC_API_KEY`` is not set in the
  environment, so a missing key surfaces early rather than silently
  producing an anthropic client that will fail on first call.

Sprint 3 scope
~~~~~~~~~~~~~~
No live judge calls are made during the Sprint 3 test run — the unit
tests in ``tests/eval/test_llm_judge.py`` mock the anthropic client
entirely. Phase 6 will drive the first real judge run against the
initial query set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256

# Candidate memory content is truncated before being shown to the judge
# so a single runaway long memory can't blow past the per-call token
# budget. 1000 characters is roughly 250 tokens, well under the 256
# max_tokens response budget on top of the (query + rubric) overhead.
_MAX_CANDIDATE_CHARS = 1000

_RUBRIC = """You are an objective relevance judge for a memory-retrieval system.

Given a query and a candidate memory, rate how relevant the candidate is
for answering the query, using a single float score in the range [0, 1]:

  1.0 = directly answers the query, unambiguous fit
  0.7 = partially answers; strong topical overlap
  0.5 = tangentially related; could be useful context
  0.3 = weak topical overlap; probably not useful
  0.0 = irrelevant; unrelated topic

Return ONLY a JSON object with exactly two keys:

  {"score": <float in [0, 1]>, "reasoning": "<one short sentence>"}

Do not return any other text, markdown, code fences, or commentary.
"""


@dataclass
class JudgeConfig:
    """Configuration snapshot echoed into every judgement for provenance."""

    model: str
    temperature: float
    max_tokens: int


class LLMJudge:
    """LLM-as-judge relevance scorer.

    Example
    -------
    >>> judge = LLMJudge()  # requires ANTHROPIC_API_KEY set
    >>> result = judge.judge_relevance(
    ...     query="What is the NovaTrade baseline strategy?",
    ...     candidate_memory={"content": "Decision: ...", "memory_type": "decision"},
    ... )
    >>> result["score"]
    0.9
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the judge and eagerly construct the anthropic client.

        Args:
            model: Pinned anthropic model id. Default is
                ``"claude-sonnet-4-6"`` (PLAN-0759 Step 0.4 pin).
            temperature: Pinned sampling temperature. Default is ``0.0``.
            max_tokens: Max response tokens. Default is ``256`` — enough
                for a short JSON response, small enough to keep the
                eval-run cost bounded.
            client: Optional preconstructed client (for unit tests to
                inject a mock). If ``None``, a real ``anthropic.Anthropic``
                client is constructed and the constructor enforces that
                ``ANTHROPIC_API_KEY`` is set.

        Raises:
            RuntimeError: If ``client`` is ``None`` and
                ``ANTHROPIC_API_KEY`` is not set in the environment.
        """
        self.config = JudgeConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if client is not None:
            self._client = client
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLMJudge requires ANTHROPIC_API_KEY to be set in the "
                "environment. Export it before constructing the judge: "
                "`export ANTHROPIC_API_KEY=sk-ant-...`. The Sprint 3 "
                "scaffolding will not hardcode a key."
            )

        # Import lazily so a test that never constructs a real judge
        # (i.e. every Sprint 3 unit test, which injects a mock client)
        # can run in environments where the anthropic package is not
        # installed.
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — env sanity
            raise RuntimeError(
                "LLMJudge requires the 'anthropic' package. Install via "
                "`pip install anthropic>=0.40.0` or use the project "
                "requirements.txt."
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def judge_relevance(
        self,
        query: str,
        candidate_memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score how relevant ``candidate_memory`` is for ``query``.

        Args:
            query: The user-style query string.
            candidate_memory: The candidate memory dict. The judge reads
                ``content`` or ``text`` (whichever is present) as the
                primary payload; other metadata fields are ignored at
                the prompt layer to keep the rubric focused on content.

        Returns:
            A dict with keys ``score`` (float in [0, 1]), ``reasoning``
            (str), ``model`` (str), and ``timestamp`` (ISO-8601 UTC).

        Raises:
            ValueError: If the judge response cannot be parsed as JSON
                or is missing the required ``score`` key. The raw
                response text is included in the error message so a
                reviewer can debug the parse failure.
            Exception: Any anthropic SDK error is re-raised as-is.
        """
        content = self._extract_content(candidate_memory)
        prompt = self._build_prompt(query=query, content=content)

        # Delegate to the injected (or real) anthropic client. Errors
        # propagate; the caller is expected to retry or abort at the
        # run_eval layer, not here.
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = self._extract_response_text(response)
        parsed = self._parse_judge_response(raw_text)

        return {
            "score": parsed["score"],
            "reasoning": parsed.get("reasoning", ""),
            "model": self.config.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers (exposed for unit tests)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_content(candidate_memory: Dict[str, Any]) -> str:
        """Pick the primary text payload from a candidate memory dict."""
        for key in ("content", "text"):
            value = candidate_memory.get(key)
            if isinstance(value, str) and value.strip():
                return value[:_MAX_CANDIDATE_CHARS]
        # Fall back to a JSON dump so the judge at least sees *something*
        # instead of being handed an empty string.
        return json.dumps(candidate_memory, sort_keys=True)[:_MAX_CANDIDATE_CHARS]

    @staticmethod
    def _build_prompt(query: str, content: str) -> str:
        """Assemble the rubric prompt for a single pair."""
        return (
            f"{_RUBRIC}\n\n"
            f"Query:\n{query.strip()}\n\n"
            f"Candidate memory:\n{content.strip()}\n"
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Pull the plain text out of an anthropic Messages API response.

        This mirrors the shape returned by ``anthropic.Anthropic.messages.create``:
        ``response.content`` is a list of content blocks, and the first
        block with a ``text`` attribute carries the model's output.
        """
        content = getattr(response, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"]
        if isinstance(content, str):
            return content
        raise ValueError(
            "Unable to extract text from anthropic response: "
            f"content={content!r}"
        )

    @staticmethod
    def _parse_judge_response(raw_text: str) -> Dict[str, Any]:
        """Parse the judge's JSON response and validate the ``score`` key.

        Raises:
            ValueError: On unparseable JSON, missing ``score``, non-numeric
                ``score``, or score outside ``[0, 1]``. The raw text is
                included in the error so a reviewer can debug drift.
        """
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLMJudge could not parse response as JSON. "
                f"Raw response:\n{raw_text!r}\n(parse error: {exc})"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLMJudge expected a JSON object, got {type(parsed).__name__}. "
                f"Raw response:\n{raw_text!r}"
            )

        if "score" not in parsed:
            raise ValueError(
                "LLMJudge response is missing required 'score' key. "
                f"Raw response:\n{raw_text!r}"
            )

        score_raw = parsed["score"]
        try:
            score = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"LLMJudge 'score' is not numeric: {score_raw!r}. "
                f"Raw response:\n{raw_text!r}"
            ) from exc

        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"LLMJudge 'score' must be in [0, 1], got {score}. "
                f"Raw response:\n{raw_text!r}"
            )

        parsed["score"] = score
        return parsed
