"""Heuristic entity extraction + canonicalization for PLAN-0759 Phase 3 (Sprint 8).

Overview
--------

This module is a **pure-Python, I/O-free utility** that Sprint 9's (future)
``EntityLinker`` will call on the ingestion path to turn a memory's raw
``content`` string into a small, deterministic list of "entity" candidates that
the linker then upserts as ``(:Entity)`` nodes with ``[:MENTIONS]`` edges from
the source ``(:base)`` memory.

Per ADR-0759 §4/§5 the linker hook lives *inside* ``perform_upsert()`` and
runs fire-and-forget after the memory is durably persisted. Sprint 8 is
library code only — it does not create any ``:Entity`` nodes, does not open a
Neo4j session, does not read the ``ASSOC_ENTITY_WRITE_ENABLED`` feature flag,
and must not import from any other ``app/services/associations/`` module.

Source contract (v2 plan Step 3.0)
----------------------------------

The downstream linker resolves entities with a two-tier priority:

- **Tier A — caller-provided**: if a memory's ``metadata["entities"]`` is a
  non-empty list, the linker uses those values verbatim (still passed through
  :func:`canon_entity` for key generation). This module is **not** responsible
  for that branch; the linker decides.
- **Tier B — heuristic**: if Tier A is absent, the linker falls back to
  :func:`extract_entities` on the memory's ``content`` string. This is the
  surface the present module implements.

No LLM-based NER is invoked by either tier in Phase 3. A richer extractor
(spaCy / Anthropic NER / domain-tagged gazetteer) is deferred to a later
phase; the heuristic extractor is deliberately conservative so that
pathological over-extraction cannot flood the graph.

Normalization spec (v2 plan Step 3.1)
-------------------------------------

:func:`canon_entity` applies the following pipeline in order. The result is
the **canonical string** used as the Entity node's ``name`` component of the
``(project, name)`` primary key.

1. Strip leading/trailing whitespace.
2. Lowercase (via :meth:`str.lower`).
3. Collapse internal runs of whitespace into a single ASCII space.
4. If the input looks like a filesystem path (contains ``/`` or ``\\``):
   - Normalize separators: convert every ``\\`` to ``/``.
   - Strip a leading ``./`` (relative-path marker).
   - Strip trailing ``/``.
5. Apply the :data:`ALIAS_TABLE` lookup on the result; if a match exists, the
   alias value replaces the normalized string.
6. Return the canonical string. Empty input (after stripping) raises
   :class:`ValueError`.

The function is **idempotent** — ``canon_entity(canon_entity(x)) ==
canon_entity(x)`` for any valid input — because every step either leaves an
already-normalized string unchanged or converges in a single additional pass.

Alias table policy
------------------

:data:`ALIAS_TABLE` is a small, hand-maintained dict of short-form aliases for
the project names that appear most frequently in operator speech. Any
additions to this table must go through a PR with operator review —
auto-growing the alias table from usage statistics is explicitly out of
scope. The table applies **after** lowercase and whitespace normalization,
so entries are stored with their normalized key form.

Ranking rules (v2 plan Step 3.2)
--------------------------------

When :func:`extract_entities` finds more candidates than the
``max_entities`` cap (default 20 per :data:`MAX_ENTITIES_PER_MEMORY`), it
selects the top N by the following deterministic rule:

1. **Earliest first-occurrence position wins** — a candidate that appears
   earlier in document order outranks one that appears later.
2. **Tie-break by length, longer first** — if two candidates happen to share
   the same first-occurrence position (which the extractor in practice never
   produces, because positions are byte offsets, but the helper function
   handles it anyway for completeness), the longer identifier wins on the
   theory that longer strings carry more specificity.
3. Take the first ``max_entities`` after sorting.

:func:`rank_and_truncate` is the public helper that encapsulates the rule.

Unicode and content-size notes
------------------------------

- **Unicode input** is accepted and :meth:`str.lower` is applied, so the
  canonicalization step is safe for non-ASCII characters. However, the
  **regex extractor patterns are ASCII-only**: non-ASCII file paths,
  identifiers with diacritics, or CJK backticked strings will simply not
  match. This is a known limitation; a future extractor upgrade can add
  Unicode-aware patterns without changing :func:`canon_entity`.
- **Content size cap** — :func:`extract_entities` truncates content larger
  than :data:`MAX_CONTENT_BYTES` (100 KB) before running regex matches. This
  is a belt-and-braces guard against pathological linker input; the linker
  itself also enforces per-memory size limits, but this module must not
  explode on a 10 MB blob. Truncation is logged at DEBUG level.

Module invariants (verified by tests)
-------------------------------------

- Zero imports from other ``app/services/associations/`` modules.
- Zero imports of backend clients or the Pydantic settings object.
- Zero imports of ``asyncio``. Every public function is synchronous.
- Regex patterns are compiled once at module load (:class:`re.Pattern`
  instances), not on every call.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


#: Maximum number of entities surfaced per memory. Above this, the ranking
#: rules in :func:`rank_and_truncate` pick the top N deterministically.
#: Kept in sync with the v2 plan Step 3.2 fan-out cap.
MAX_ENTITIES_PER_MEMORY: int = 20


#: Hard cap on content size the extractor will scan. Anything longer is
#: truncated to the first ``MAX_CONTENT_BYTES`` bytes (UTF-8) before regex
#: matching. 100 KB is the documented bound from Sprint 8's design notes.
MAX_CONTENT_BYTES: int = 100 * 1024


#: Maximum number of bytes of a backticked span the extractor will keep.
#: Kept identical to the regex cap for consistency.
MAX_BACKTICK_LENGTH: int = 60


#: Hand-maintained alias table mapping short-form project names to their
#: canonical project IDs. Applied **after** lowercase + whitespace
#: normalization by :func:`canon_entity`, so the keys are stored in their
#: already-normalized form.
#:
#: Any additions must go through a PR with operator review — this table is
#: deliberately small and not auto-grown from usage statistics.
ALIAS_TABLE: Dict[str, str] = {
    "nc": "nova-core",
    "nt": "novatrade",
}


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------


#: The whitelisted file extensions. **Order matters**: longer extensions
#: must come before shorter prefixes (e.g. ``tsx`` before ``ts``, ``yaml``
#: before ``yml``, ``jsx`` before ``js``) so that regex alternation picks
#: the longest match first.
_EXTENSIONS: str = "tsx|jsx|yaml|py|md|ts|json|yml|sh|toml|js|rs|go"


#: File paths with at least one directory component and a whitelisted
#: extension. Matches e.g. ``agents/memory_router.py``,
#: ``docs/guides/README.md``, ``configs/prod/app.yml``.
#:
#: The character classes are bounded (``{1,200}``) rather than unbounded
#: (``+``) so that pathological inputs (e.g. 100 KB of filename-legal
#: characters with no path separator) cannot trigger quadratic backtracking.
#: 200 is far above the realistic longest filename on any supported OS.
_PATH_PATTERN: re.Pattern[str] = re.compile(
    rf"(?:(?:[a-zA-Z0-9_.-]{{1,200}}/){{1,10}}[a-zA-Z0-9_.-]{{1,200}}\.(?:{_EXTENSIONS})\b)"
)


#: Bare file-like tokens (no directory component). Matches e.g. ``watcher.py``,
#: ``README.md``. Note that the tighter :data:`_PATH_PATTERN` is tried first;
#: :func:`extract_entities` filters out overlapping matches so ``a/b.py`` is
#: not also reported as ``b.py``.
#:
#: The identifier portion uses ``[a-zA-Z0-9_-]`` (no literal dot) so that a
#: run of identifier chars cannot be greedily consumed through a nearby dot
#: and then backtracked across. The ``\b`` boundaries ensure the engine
#: can skip non-word start positions cheaply.
_BARE_FILE_PATTERN: re.Pattern[str] = re.compile(
    rf"\b[a-zA-Z0-9_-]{{1,200}}\.(?:{_EXTENSIONS})\b"
)


#: CamelCase identifiers with at least TWO uppercase letters. This rules out
#: plain capitalized words (``Claude``, ``Tuesday``) while still catching
#: ``CamelCase``, ``NovaCore``, ``MemoryEdge``, ``LLMJudge``,
#: ``HTTPRequest``.
_CAMELCASE_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"
)


#: Backticked spans. Non-greedy, single-line only (explicit ``[^`\n]``),
#: capped at :data:`MAX_BACKTICK_LENGTH` characters between the backticks.
#: The capture group holds the inner span, not the backticks themselves.
_BACKTICK_PATTERN: re.Pattern[str] = re.compile(
    r"`([^`\n]{1,60})`"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canon_entity(raw: str) -> str:
    """Return the canonical form of a raw entity string.

    Parameters
    ----------
    raw:
        The raw entity string as produced by the caller (either Tier A
        metadata or Tier B heuristic extraction). Must be a non-empty
        string after whitespace stripping.

    Returns
    -------
    str
        The canonical string used as the ``name`` component of the
        ``(project, name)`` Entity node key.

    Raises
    ------
    ValueError
        If ``raw`` is empty or contains only whitespace.

    Examples
    --------
    >>> canon_entity("Neo4j")
    'neo4j'
    >>> canon_entity("  Foo   Bar  ")
    'foo bar'
    >>> canon_entity("./agents/memory_router.py")
    'agents/memory_router.py'
    >>> canon_entity("agents\\\\memory_router.py")
    'agents/memory_router.py'
    >>> canon_entity("NC")
    'nova-core'
    """
    if not isinstance(raw, str):
        raise ValueError(f"canon_entity expected str, got {type(raw).__name__}")

    # Step 1 + 2: strip and lowercase.
    s = raw.strip().lower()
    if not s:
        raise ValueError("canon_entity refuses empty/whitespace-only input")

    # Step 3: collapse runs of internal whitespace into a single ASCII space.
    # This runs before path normalization so that e.g. ``"  ./foo  / bar  "``
    # becomes ``"./foo / bar"`` before we try to strip a trailing ``/``.
    s = re.sub(r"\s+", " ", s)

    # Step 4: path-flavoured normalization. We treat a string as "path-like"
    # if it contains any path separator. Non-path strings (bare identifiers,
    # backticked fragments) skip this block and fall straight through to the
    # alias lookup.
    if "/" in s or "\\" in s:
        # Normalize backslashes to forward slashes so cross-platform paths
        # produce the same canonical string regardless of which separator
        # the author typed.
        s = s.replace("\\", "/")
        # Strip a leading ``./`` (relative-path marker).
        if s.startswith("./"):
            s = s[2:]
        # Strip a single trailing ``/`` if present (but keep a lone ``/``
        # itself intact — an edge case that can't collide with any real
        # entity name and is trivially idempotent).
        if len(s) > 1 and s.endswith("/"):
            s = s[:-1]

    # Step 5: alias table lookup applied on the final normalized form.
    if s in ALIAS_TABLE:
        s = ALIAS_TABLE[s]

    return s


def rank_and_truncate(
    candidates: List[Tuple[str, int]], max_entities: int
) -> List[str]:
    """Rank candidate entities and return the top ``max_entities`` raw strings.

    Ranking rules (v2 plan Step 3.2):

    1. Earliest first-occurrence position wins (ascending ``position``).
    2. Tie-break by longer ``raw`` string first.
    3. Take the first ``max_entities`` from the sorted list.

    Parameters
    ----------
    candidates:
        List of ``(raw, position)`` tuples as produced by
        :func:`extract_entities` during its scan. ``position`` is a byte
        offset into the content (any monotonic integer will do as long as
        smaller means "earlier"). ``raw`` is the pre-canonicalized surface
        form.
    max_entities:
        Maximum number of candidates to return. Must be non-negative. A
        value of zero returns an empty list.

    Returns
    -------
    list[str]
        The top ``max_entities`` raw strings, in ranked order (earliest
        first, longer on ties).
    """
    if max_entities <= 0:
        return []
    # ``sorted`` is stable, so the primary key (position ASC) fires first and
    # the secondary key (-length, i.e. longer first) resolves ties. We use a
    # tuple key rather than two sort passes for determinism.
    ranked = sorted(candidates, key=lambda pair: (pair[1], -len(pair[0])))
    return [raw for raw, _pos in ranked[:max_entities]]


def extract_entities(
    content: str, max_entities: int = MAX_ENTITIES_PER_MEMORY
) -> List[str]:
    """Extract heuristic entity candidates from memory content.

    This is the Tier B path of the source contract. It runs four regex
    patterns (file paths, bare file names, CamelCase identifiers, backticked
    spans), deduplicates by canonical form, ranks by first-occurrence
    position, and truncates to ``max_entities``.

    The returned list holds the **raw** surface forms (as they appeared in
    the content) — not the canonicalized names. The downstream linker is
    responsible for running each raw string through :func:`canon_entity`
    again when it constructs the ``(project, name)`` key. This mirrors the
    Tier A contract: the linker always runs canonicalization as a final
    step regardless of which tier provided the raw list.

    Parameters
    ----------
    content:
        The memory's full text content. If longer than
        :data:`MAX_CONTENT_BYTES`, it is truncated to the first 100 KB and
        a DEBUG log is emitted. Non-string input raises ``ValueError``.
    max_entities:
        Maximum entities to return. Defaults to
        :data:`MAX_ENTITIES_PER_MEMORY` (20).

    Returns
    -------
    list[str]
        Up to ``max_entities`` raw entity strings, in ranked order.
        Returns an empty list for empty, whitespace-only, or match-free
        content. Never raises on empty input.
    """
    if not isinstance(content, str):
        raise ValueError(
            f"extract_entities expected str, got {type(content).__name__}"
        )

    # Empty / whitespace-only content is legitimate (many memories have
    # very short bodies); return an empty list without running the regex
    # engine on the empty string.
    if not content or not content.strip():
        return []

    # 100 KB cap — truncate first. We measure UTF-8 byte length (encode +
    # slice + decode) so that multi-byte characters don't push us past the
    # intended budget. Using errors="ignore" drops a possibly-incomplete
    # trailing multibyte sequence rather than raising.
    encoded = content.encode("utf-8", errors="ignore")
    if len(encoded) > MAX_CONTENT_BYTES:
        logger.debug(
            "entity_extractor.truncated original_bytes=%d kept_bytes=%d",
            len(encoded),
            MAX_CONTENT_BYTES,
        )
        content = encoded[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")

    # Collect ``(raw, position)`` tuples across all four patterns. We keep
    # raw surface forms (not canonicalized) so the ranker can tie-break by
    # length on the author's original spelling.
    raw_hits: List[Tuple[str, int]] = []

    # Track the span of every full-path match so that a later bare-file
    # pass can skip characters already claimed by a fuller path match.
    # Example: ``"see agents/memory_router.py"`` should produce ONE
    # candidate (``agents/memory_router.py``), not also ``memory_router.py``.
    path_spans: List[Tuple[int, int]] = []
    for m in _PATH_PATTERN.finditer(content):
        raw_hits.append((m.group(0), m.start()))
        path_spans.append(m.span())

    def _in_path_span(pos: int) -> bool:
        for start, end in path_spans:
            if start <= pos < end:
                return True
        return False

    for m in _BARE_FILE_PATTERN.finditer(content):
        if _in_path_span(m.start()):
            continue
        raw_hits.append((m.group(0), m.start()))

    # CamelCase matches that fall inside a path span are almost always
    # capitalized filename stems (e.g. ``README`` inside ``docs/README.md``)
    # and are redundant with the path hit. Suppress them so the linker
    # gets one clean entity per conceptual token.
    for m in _CAMELCASE_PATTERN.finditer(content):
        if _in_path_span(m.start()):
            continue
        raw_hits.append((m.group(0), m.start()))

    for m in _BACKTICK_PATTERN.finditer(content):
        # The captured group is the inside of the backticks. Position is
        # the start of the inside span, not the backtick itself — that's
        # fine for ranking (it's still monotonically increasing in the
        # content).
        inner = m.group(1)
        # Regex already bounds the inner length to 60; defence-in-depth
        # skip anything that somehow slipped through.
        if len(inner) > MAX_BACKTICK_LENGTH:
            continue
        raw_hits.append((inner, m.start(1)))

    if not raw_hits:
        return []

    # Deduplicate by canonical form, keeping the FIRST occurrence of each
    # canonical. Iteration order of raw_hits is pattern-group order (paths,
    # bare files, camelcase, backticks), which does not correspond to
    # document order. We must sort by position BEFORE the dedup pass so
    # "first occurrence" means "earliest in document order" rather than
    # "first pattern that fired".
    raw_hits.sort(key=lambda pair: pair[1])

    seen_canonical: Dict[str, Tuple[str, int]] = {}
    for raw, pos in raw_hits:
        try:
            canon = canon_entity(raw)
        except ValueError:
            # Defensive: regex shouldn't produce empty strings, but an
            # all-whitespace backtick span could in theory. Skip silently.
            continue
        if canon not in seen_canonical:
            seen_canonical[canon] = (raw, pos)

    deduped: List[Tuple[str, int]] = list(seen_canonical.values())

    return rank_and_truncate(deduped, max_entities)


__all__ = [
    "ALIAS_TABLE",
    "MAX_ENTITIES_PER_MEMORY",
    "MAX_CONTENT_BYTES",
    "canon_entity",
    "extract_entities",
    "rank_and_truncate",
]
