"""Unit tests for the Phase 3 / Sprint 8 entity extractor.

These tests are **fully hermetic**. No Neo4j, no Pinecone, no Anthropic, no
async. Every function under test in ``app/services/associations/entity_extractor``
is pure Python and I/O-free, so the test surface is stdlib + pytest only.

The suite covers:

1. ``canon_entity`` — whitespace, lowercase, path normalization, alias
   table, ValueError on empty input, idempotency.
2. ``extract_entities`` — the four regex patterns (paths, bare files,
   CamelCase identifiers, backticked spans), document-order ranking,
   length tie-break, deduplication by canonical form, ``MAX_ENTITIES_PER_MEMORY``
   enforcement, 100 KB content cap, determinism.
3. ``rank_and_truncate`` — exercised directly for the tie-break rule.
4. Structural invariants — compiled regex patterns at module load time,
   no forbidden imports.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pytest

# Make the ``app`` package importable without pulling in app.config.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.associations import entity_extractor as ee
from app.services.associations.entity_extractor import (
    ALIAS_TABLE,
    MAX_CONTENT_BYTES,
    MAX_ENTITIES_PER_MEMORY,
    canon_entity,
    extract_entities,
    rank_and_truncate,
)


# ---------------------------------------------------------------------------
# 1. canon_entity — normalization pipeline
# ---------------------------------------------------------------------------


def test_canon_simple_lowercase() -> None:
    """Upper-case tokens are lowercased."""
    assert canon_entity("Neo4j") == "neo4j"


def test_canon_strips_whitespace() -> None:
    """Leading and trailing whitespace is removed."""
    assert canon_entity("  Neo4j  ") == "neo4j"


def test_canon_collapses_internal_whitespace() -> None:
    """Runs of internal whitespace collapse to a single ASCII space."""
    assert canon_entity("foo   bar") == "foo bar"


def test_canon_collapses_tabs_and_newlines() -> None:
    """Mixed tab/newline whitespace also collapses."""
    assert canon_entity("foo\t\nbar") == "foo bar"


def test_canon_path_separator_normalize() -> None:
    """Windows-style backslashes become forward slashes."""
    assert canon_entity("agents\\memory_router.py") == "agents/memory_router.py"


def test_canon_leading_dot_slash_stripped() -> None:
    """``./`` relative-path marker is removed."""
    assert canon_entity("./watcher.py") == "watcher.py"


def test_canon_trailing_slash_stripped() -> None:
    """A trailing ``/`` on a path is removed."""
    assert canon_entity("agents/") == "agents"


def test_canon_trailing_slash_on_nested_path() -> None:
    """Trailing slash on a multi-segment path is also removed."""
    assert canon_entity("agents/sub/") == "agents/sub"


def test_canon_path_normalization_preserves_slashes_in_middle() -> None:
    """Internal slashes are kept intact during path normalization."""
    assert canon_entity("./a/b/c.py") == "a/b/c.py"


def test_canon_alias_lookup_uppercase() -> None:
    """Alias lookup runs after lowercasing so ``NC`` matches ``nc``."""
    assert canon_entity("NC") == "nova-core"


def test_canon_alias_lookup_with_padding() -> None:
    """Whitespace + uppercase both normalize before alias lookup."""
    assert canon_entity("  nc  ") == "nova-core"


def test_canon_alias_nt_lookup() -> None:
    """The second alias table entry also resolves."""
    assert canon_entity("NT") == "novatrade"


def test_canon_no_alias_match_passthrough() -> None:
    """Strings that do not appear in the alias table pass through."""
    assert canon_entity("unknown-alias") == "unknown-alias"


def test_canon_empty_raises() -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError):
        canon_entity("")


def test_canon_whitespace_only_raises() -> None:
    """Whitespace-only input raises ValueError."""
    with pytest.raises(ValueError):
        canon_entity("   \t\n  ")


def test_canon_non_string_raises() -> None:
    """Non-string input raises ValueError (not TypeError)."""
    with pytest.raises(ValueError):
        canon_entity(None)  # type: ignore[arg-type]


def test_canon_unicode_preserved_but_lowercased() -> None:
    """Unicode input is accepted and lowercased."""
    assert canon_entity("Datei") == "datei"


def test_canon_is_idempotent_across_varied_inputs() -> None:
    """canon(canon(x)) == canon(x) for varied inputs."""
    samples = [
        "Neo4j",
        "  FooBar  ",
        "./agents/memory_router.py",
        "agents\\subdir\\file.py",
        "NC",
        "nt",
        "MyClassName",
        "foo   bar",
        "README.md",
        "docs/guides/",
    ]
    for raw in samples:
        first = canon_entity(raw)
        second = canon_entity(first)
        assert first == second, (
            f"canon_entity is not idempotent on {raw!r}: "
            f"first={first!r} second={second!r}"
        )


def test_canon_alias_table_exact_entries() -> None:
    """Alias table has exactly the Sprint 8 approved entries."""
    assert ALIAS_TABLE == {"nc": "nova-core", "nt": "novatrade"}


# ---------------------------------------------------------------------------
# 2. extract_entities — path and file heuristics
# ---------------------------------------------------------------------------


def test_extract_python_file_path() -> None:
    """A single Python file path is extracted."""
    out = extract_entities("see agents/memory_router.py for details")
    assert out == ["agents/memory_router.py"]


def test_extract_markdown_file_path() -> None:
    """Markdown file paths are extracted."""
    out = extract_entities("refer to docs/README.md")
    assert out == ["docs/README.md"]


def test_extract_bare_file_name() -> None:
    """A bare file name without a directory is extracted."""
    out = extract_entities("check watcher.py")
    assert out == ["watcher.py"]


def test_extract_multiple_file_paths_in_document_order() -> None:
    """Multiple file paths appear in the order they were found."""
    out = extract_entities("a.py then b.md")
    assert out == ["a.py", "b.md"]


def test_extract_full_path_over_bare_filename() -> None:
    """A full path match suppresses a duplicate bare-file match inside it."""
    # The content contains ``agents/memory_router.py`` which should be
    # extracted once, not also as the bare ``memory_router.py`` substring.
    out = extract_entities("look at agents/memory_router.py please")
    assert out == ["agents/memory_router.py"]


def test_extract_whitelisted_extensions_variety() -> None:
    """Every whitelisted extension produces a hit."""
    content = "a.py b.md c.ts d.tsx e.json f.yml g.yaml h.sh i.toml j.js k.jsx l.rs m.go"
    out = extract_entities(content)
    expected = [
        "a.py",
        "b.md",
        "c.ts",
        "d.tsx",
        "e.json",
        "f.yml",
        "g.yaml",
        "h.sh",
        "i.toml",
        "j.js",
        "k.jsx",
        "l.rs",
        "m.go",
    ]
    assert out == expected


def test_extract_non_whitelisted_extension_ignored() -> None:
    """Files with unknown extensions are not extracted."""
    out = extract_entities("look at readme.txt or data.csv")
    assert out == []


# ---------------------------------------------------------------------------
# 3. extract_entities — CamelCase identifiers
# ---------------------------------------------------------------------------


def test_extract_camelcase_identifier() -> None:
    """A simple CamelCase identifier is extracted."""
    out = extract_entities("the CamelCase class")
    assert out == ["CamelCase"]


def test_extract_novacore() -> None:
    """``NovaCore`` has two uppercase letters and qualifies."""
    out = extract_entities("working on NovaCore tonight")
    assert out == ["NovaCore"]


def test_extract_single_capital_not_camelcase() -> None:
    """``Claude`` has only one capital so it is NOT extracted."""
    out = extract_entities("Claude is helpful")
    assert out == []


def test_extract_all_caps_two_letters() -> None:
    """Two-letter all-caps identifiers do qualify (they have 2 uppers)."""
    # ``HTTPRequest`` has 5 uppercase letters total.
    out = extract_entities("the HTTPRequest handler")
    assert out == ["HTTPRequest"]


def test_extract_llm_judge_multi_case() -> None:
    """``LLMJudge`` pattern matches."""
    out = extract_entities("use LLMJudge here")
    assert out == ["LLMJudge"]


# ---------------------------------------------------------------------------
# 4. extract_entities — backticked spans
# ---------------------------------------------------------------------------


def test_extract_backticked_identifier() -> None:
    """A backticked identifier is extracted verbatim."""
    out = extract_entities("we use `settings.ASSOC_FOO` here")
    assert out == ["settings.ASSOC_FOO"]


def test_extract_backtick_with_spaces() -> None:
    """Spaces inside backticks are fine."""
    out = extract_entities("use `the thing`")
    assert out == ["the thing"]


def test_extract_backtick_over_60_chars_ignored() -> None:
    """Backticks longer than 60 chars do not match."""
    content = "`" + ("x" * 70) + "`"
    out = extract_entities(content)
    # The pattern requires 1-60 chars between backticks, so this misses.
    # There is also no CamelCase / file match in the content.
    assert out == []


def test_extract_multiline_backtick_ignored() -> None:
    """Backticks spanning a newline are rejected by the pattern."""
    out = extract_entities("`a\nb`")
    # No qualifying backtick match; also no camelcase or file hits.
    assert out == []


def test_extract_backtick_exact_60_chars_allowed() -> None:
    """Boundary: exactly 60 chars inside backticks is allowed."""
    inner = "y" * 60
    out = extract_entities(f"`{inner}`")
    assert out == [inner]


# ---------------------------------------------------------------------------
# 5. Deduplication and ranking
# ---------------------------------------------------------------------------


def test_extract_dedup_case_insensitive_via_canonicalization() -> None:
    """``Neo4j`` and ``neo4j`` collapse to a single canonical entry."""
    # Both ``Neo4j`` and ``neo4j`` would appear — they canonicalize to
    # the same ``neo4j`` string. Only one entry should be returned.
    # Both need to actually MATCH a pattern first; we wrap them in backticks
    # so the backtick pattern picks them up.
    out = extract_entities("`Neo4j` and `neo4j`")
    assert len(out) == 1
    # The first occurrence's raw form wins.
    assert out[0] == "Neo4j"


def test_extract_ranking_by_document_position() -> None:
    """Earlier-position candidates rank first."""
    # Two CamelCase entities: ``SecondThing`` appears before ``FirstThing``
    # in the content, so ``SecondThing`` ranks first.
    out = extract_entities("SecondThing and then FirstThing")
    assert out == ["SecondThing", "FirstThing"]


def test_extract_ranking_tie_break_by_length() -> None:
    """When positions tie, the longer raw string wins."""
    # Synthetic: rank_and_truncate handles this directly since
    # extract_entities' positions are monotonically increasing and will
    # never actually tie in practice.
    candidates = [("short", 0), ("muchlongerstring", 0)]
    assert rank_and_truncate(candidates, 2) == ["muchlongerstring", "short"]


def test_rank_and_truncate_respects_max_entities() -> None:
    """rank_and_truncate drops candidates past max_entities."""
    candidates = [("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
    assert rank_and_truncate(candidates, 3) == ["a", "b", "c"]


def test_rank_and_truncate_zero_returns_empty() -> None:
    """A zero cap returns an empty list."""
    assert rank_and_truncate([("a", 0)], 0) == []


def test_rank_and_truncate_negative_returns_empty() -> None:
    """A negative cap returns an empty list."""
    assert rank_and_truncate([("a", 0)], -5) == []


def test_extract_max_entities_enforcement() -> None:
    """extract_entities truncates at MAX_ENTITIES_PER_MEMORY (20)."""
    # Synthesize 30 unique CamelCase entities to guarantee we exceed the cap.
    camels = [f"Aa{i:02d}Bb" for i in range(30)]
    # Each of these has two uppercase letters (``A`` and ``B``) so all
    # 30 must match the CamelCase pattern.
    content = " ".join(camels)
    out = extract_entities(content)
    assert len(out) == MAX_ENTITIES_PER_MEMORY == 20
    # Earliest-first ranking: the first 20 in document order wins.
    assert out == camels[:20]


def test_extract_max_entities_respects_explicit_cap() -> None:
    """An explicit ``max_entities`` arg overrides the default."""
    camels = [f"Aa{i:02d}Bb" for i in range(10)]
    content = " ".join(camels)
    out = extract_entities(content, max_entities=5)
    assert out == camels[:5]


# ---------------------------------------------------------------------------
# 6. Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_extract_empty_content() -> None:
    """Empty content returns an empty list without raising."""
    assert extract_entities("") == []


def test_extract_whitespace_only_content() -> None:
    """Whitespace-only content returns an empty list."""
    assert extract_entities("   \n\t  ") == []


def test_extract_content_with_no_matches() -> None:
    """Plain prose with no qualifying tokens returns empty."""
    out = extract_entities("just a simple sentence with nothing special")
    assert out == []


def test_extract_non_string_raises() -> None:
    """Non-string input raises ValueError."""
    with pytest.raises(ValueError):
        extract_entities(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. 100 KB content cap
# ---------------------------------------------------------------------------


def test_extract_content_larger_than_100kb_is_truncated() -> None:
    """200 KB input runs quickly and only the first 100 KB produces matches.

    The content is crafted so that:
    - ``ClassOne`` and ``thing`` appear in the first 35 bytes (well inside
      the 100 KB window),
    - the middle is padded with ``x`` bytes to push the 100 KB boundary
      past the end of the padding,
    - ``ClassTwoNotSeen`` appears only *after* the 100 KB boundary, so the
      extractor must drop it.
    """
    head = "start ClassOne middle `thing` tail "
    padding = "x" * (MAX_CONTENT_BYTES + 100)  # pushes ClassTwoNotSeen out
    tail = " ClassTwoNotSeen " + ("y" * 1024)
    content = head + padding + tail
    assert len(content.encode("utf-8")) > MAX_CONTENT_BYTES

    t0 = time.perf_counter()
    out = extract_entities(content)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"extraction took {elapsed:.3f}s (expected < 1s)"

    # ClassOne and `thing` are in the first 100 KB, ClassTwoNotSeen is not.
    assert "ClassOne" in out
    assert "thing" in out
    assert "ClassTwoNotSeen" not in out


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_extract_deterministic_across_repeated_calls() -> None:
    """Same input → identical output list on every call."""
    content = (
        "look at agents/memory_router.py and `NovaCore` and ClassFoo "
        "plus docs/README.md for MoreInfo"
    )
    first = extract_entities(content)
    for _ in range(9):
        assert extract_entities(content) == first


def test_extract_alias_integration_dedup() -> None:
    """NC and nova-core canonicalize to the same key; only one entry kept.

    The memory content contains ``NC`` (an alias for ``nova-core``) and
    ``nova-core`` (the canonical form). Because neither is a file path or
    CamelCase match, both must be wrapped in backticks for the backtick
    pattern to pick them up. The first-occurrence raw form wins, so the
    surviving entry is the earlier one.
    """
    out = extract_entities("The `NC` repo is the same as `nova-core`")
    # Both forms canonicalize to ``nova-core`` via the alias table, so
    # there is exactly one entry in the result.
    assert len(out) == 1
    # First-occurrence wins: ``NC`` appeared first in the content.
    assert out[0] == "NC"
    # And it canonicalizes to the target alias for the downstream linker.
    assert canon_entity(out[0]) == "nova-core"


# ---------------------------------------------------------------------------
# 9. Structural invariants (module load time)
# ---------------------------------------------------------------------------


def test_patterns_are_compiled_at_module_load() -> None:
    """Every regex pattern is a compiled re.Pattern, not a str."""
    assert isinstance(ee._PATH_PATTERN, re.Pattern)
    assert isinstance(ee._BARE_FILE_PATTERN, re.Pattern)
    assert isinstance(ee._CAMELCASE_PATTERN, re.Pattern)
    assert isinstance(ee._BACKTICK_PATTERN, re.Pattern)


def test_module_has_no_neo4j_or_pinecone_imports() -> None:
    """entity_extractor must not import any backend clients or config."""
    module_source = Path(ee.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import neo4j",
        "from neo4j",
        "import pinecone",
        "from pinecone",
        "import anthropic",
        "from anthropic",
        "import asyncio",
        "from asyncio",
        "app.config",
    ):
        assert forbidden not in module_source, (
            f"entity_extractor.py must not contain {forbidden!r}"
        )


def test_module_has_no_associations_sibling_imports() -> None:
    """entity_extractor must be a standalone utility within associations/."""
    module_source = Path(ee.__file__).read_text(encoding="utf-8")
    # No imports of other associations/ siblings.
    for sibling in (
        "memory_edges",
        "edge_cypher",
        "edge_service",
        "similarity_linker",
    ):
        assert f"from .{sibling}" not in module_source
        assert f"from app.services.associations.{sibling}" not in module_source


def test_max_entities_constant_value() -> None:
    """MAX_ENTITIES_PER_MEMORY is pinned at 20 (v2 plan Step 3.2)."""
    assert MAX_ENTITIES_PER_MEMORY == 20


def test_max_content_bytes_constant_value() -> None:
    """MAX_CONTENT_BYTES is pinned at 100 KB."""
    assert MAX_CONTENT_BYTES == 100 * 1024


def test_module_exports_public_api() -> None:
    """__all__ lists the public symbols Sprint 9's linker will import."""
    assert set(ee.__all__) == {
        "ALIAS_TABLE",
        "MAX_ENTITIES_PER_MEMORY",
        "MAX_CONTENT_BYTES",
        "canon_entity",
        "extract_entities",
        "rank_and_truncate",
    }
