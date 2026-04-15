"""Claude Code CLI judge — subscription-auth alternative to LLMJudge.

Shells out to ``claude -p --model claude-opus-4-6 --effort high --output-format json``
so the Phase 6 eval can run against the operator's Claude Code subscription
instead of a console API key. Duck-types into the same ``judge_relevance``
contract that ``run_eval`` in ``associative_recall_eval.py`` expects.

Batched scoring
---------------
``judge_batch(query, candidates)`` scores an entire top-k list in one CLI
call — ~10x fewer subprocess invocations and far less cache overhead than
per-candidate calls. ``judge_relevance`` is a thin single-candidate
wrapper over ``judge_batch`` so existing callers stay compatible.

The rubric mirrors ``llm_judge.LLMJudge`` exactly so scores are comparable
between the two backends.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence


DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_EFFORT = "high"
DEFAULT_TIMEOUT_SEC = 300

# Candidate memory content is truncated before being shown to the judge so
# a single runaway memory can't blow past the prompt budget. Matches
# llm_judge._MAX_CANDIDATE_CHARS so scores are comparable.
_MAX_CANDIDATE_CHARS = 1000

_RUBRIC_HEADER = """You are an objective relevance judge for a memory-retrieval system.

For each candidate below, rate how relevant the candidate is for answering
the query, using a single float score in [0, 1]:

  1.0 = directly answers the query, unambiguous fit
  0.7 = partially answers; strong topical overlap
  0.5 = tangentially related; could be useful context
  0.3 = weak topical overlap; probably not useful
  0.0 = irrelevant; unrelated topic

Return ONLY a JSON array with one object per candidate, in the SAME ORDER
as the candidates below. Each object has exactly two keys:

  {"score": <float in [0, 1]>, "reasoning": "<one short sentence>"}

Do not return any text, markdown, or code fences outside the JSON array.
"""


@dataclass
class JudgeConfig:
    model: str
    effort: str
    timeout_sec: int


class ClaudeCLIJudge:
    """LLM-as-judge that shells out to the Claude Code CLI.

    The CLI honors the operator's subscription auth, so no
    ``ANTHROPIC_API_KEY`` is required. Suitable for running Phase 6 evals
    on a Max/Pro subscription rather than console credits.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.config = JudgeConfig(
            model=model, effort=effort, timeout_sec=timeout_sec
        )

    def judge_relevance(
        self, query: str, candidate_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single-candidate score. Prefer ``judge_batch`` when scoring a
        full top-k list — it amortizes the CLI startup cost."""
        results = self.judge_batch(query, [candidate_memory])
        return results[0]

    def judge_batch(
        self, query: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score a list of candidates in one CLI call.

        Returns one judgement dict per input candidate, in the same order,
        each with keys ``score``, ``reasoning``, ``model``, ``timestamp``.
        If the CLI returns fewer scores than candidates (model skipped
        some), the tail is padded with ``score=0.0`` and a reasoning note
        so the eval harness always gets a dense list.
        """
        if not candidates:
            return []

        prompt = self._build_batch_prompt(query, candidates)
        raw_result = self._invoke_cli(prompt)
        parsed = self._parse_judge_response(raw_result)
        parsed = self._coerce_length(parsed, len(candidates))

        now = datetime.now(timezone.utc).isoformat()
        out: List[Dict[str, Any]] = []
        for entry in parsed:
            out.append(
                {
                    "score": float(entry["score"]),
                    "reasoning": str(entry.get("reasoning", "")),
                    "model": self.config.model,
                    "timestamp": now,
                }
            )
        return out

    def _build_batch_prompt(
        self, query: str, candidates: Sequence[Dict[str, Any]]
    ) -> str:
        parts = [_RUBRIC_HEADER, "", f"Query:\n{query.strip()}", "", "Candidates:"]
        for i, cand in enumerate(candidates, start=1):
            content = self._extract_content(cand)
            parts.append(f"\n--- Candidate {i} ---")
            parts.append(content)
        parts.append("")
        parts.append(
            f"Return a JSON array of exactly {len(candidates)} objects, in the"
            " order the candidates are listed above."
        )
        return "\n".join(parts)

    @staticmethod
    def _extract_content(candidate_memory: Dict[str, Any]) -> str:
        for key in ("content", "text"):
            value = candidate_memory.get(key)
            if isinstance(value, str) and value.strip():
                return value[:_MAX_CANDIDATE_CHARS]
        return json.dumps(candidate_memory, sort_keys=True)[:_MAX_CANDIDATE_CHARS]

    def _invoke_cli(self, prompt: str) -> str:
        """Invoke ``claude -p`` and return the raw model ``result`` string.

        ``claude -p --output-format json`` emits a single JSON envelope on
        stdout with a ``result`` field containing the model response. We
        return the ``result`` field verbatim — the caller is responsible
        for stripping markdown fences and parsing the inner JSON.
        """
        cmd = [
            "claude",
            "-p",
            "--model",
            self.config.model,
            "--effort",
            self.config.effort,
            "--output-format",
            "json",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ClaudeCLIJudge: 'claude -p' timed out after"
                f" {self.config.timeout_sec}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"ClaudeCLIJudge: 'claude -p' exited {proc.returncode}.\n"
                f"stderr:\n{proc.stderr}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "ClaudeCLIJudge: could not parse CLI envelope JSON.\n"
                f"stdout:\n{proc.stdout[:2000]}"
            ) from exc

        if envelope.get("is_error"):
            raise RuntimeError(
                "ClaudeCLIJudge: CLI envelope reported is_error=true. "
                f"result={envelope.get('result')!r}"
            )

        result = envelope.get("result")
        if not isinstance(result, str):
            raise RuntimeError(
                "ClaudeCLIJudge: CLI envelope missing 'result' string. "
                f"envelope keys={list(envelope.keys())}"
            )
        return result

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Strip ```json ... ``` or ``` ... ``` fences that Opus sometimes
        wraps structured output in despite 'no fences' instructions."""
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    def _parse_judge_response(self, raw: str) -> List[Dict[str, Any]]:
        stripped = self._strip_fences(raw)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # Sometimes the model emits a single object when k=1 — try to
            # unwrap into a one-element list before giving up.
            raise ValueError(
                f"ClaudeCLIJudge could not parse judge response as JSON.\n"
                f"Stripped:\n{stripped[:2000]}\n(parse error: {exc})"
            ) from exc

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError(
                "ClaudeCLIJudge expected a JSON array, got "
                f"{type(parsed).__name__}. Raw:\n{stripped[:2000]}"
            )

        out: List[Dict[str, Any]] = []
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"ClaudeCLIJudge: candidate {i} is not a dict: {entry!r}"
                )
            if "score" not in entry:
                raise ValueError(
                    f"ClaudeCLIJudge: candidate {i} missing 'score'. "
                    f"Got: {entry!r}"
                )
            try:
                score = float(entry["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ClaudeCLIJudge: candidate {i} 'score' not numeric: "
                    f"{entry['score']!r}"
                ) from exc
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"ClaudeCLIJudge: candidate {i} 'score' out of [0,1]: "
                    f"{score}"
                )
            out.append({"score": score, "reasoning": entry.get("reasoning", "")})
        return out

    @staticmethod
    def _coerce_length(
        parsed: List[Dict[str, Any]], expected: int
    ) -> List[Dict[str, Any]]:
        """Make sure the parsed list is exactly `expected` long.

        The eval harness treats missing candidates as irrelevant, so if
        the model returns fewer, we pad with score=0.0. If the model
        returns more, we truncate.
        """
        if len(parsed) == expected:
            return parsed
        if len(parsed) > expected:
            return parsed[:expected]
        pad_count = expected - len(parsed)
        padding = [
            {
                "score": 0.0,
                "reasoning": "judge returned fewer scores than candidates; padded",
            }
            for _ in range(pad_count)
        ]
        return parsed + padding
