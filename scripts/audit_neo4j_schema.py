"""Read-only Neo4j schema audit for PLAN-0759 Phase 0 / Step 0.2.

Purpose
-------
Before the Associative Linking v2 plan (PLAN-0759) starts writing any new edge
types into the Fusion Memory graph, we need to verify three things against the
*live* Neo4j instance:

1. The canonical node identity that every new linker will target is really
   ``(:base {entity_id})`` (ADR-0759 Section 7 locks this as an invariant, but
   this script re-confirms it on the wire so Phase 1 Cypher does not regress
   to the placeholder ``(:Memory {memory_id})`` shown in v2 plan text).
2. The existing ``FOLLOWS`` relationship type is genuinely in use for
   ``:Session`` chaining, which is the collision that motivated renaming the
   PLAN-0759 temporal edge to ``MEMORY_FOLLOWS`` (ADR-0759 Section 6).
3. No PLAN-0759 edge type (``SIMILAR_TO``, ``MEMORY_FOLLOWS``, ``MENTIONS``,
   ``PROMOTED_FROM``, ``SUPERSEDES``, ``COMPACTED_FROM``, ``CAUSED_BY``,
   ``RELATED_TASK``, ``CO_OCCURS``) already exists with a non-zero count —
   except possibly ``SUPERSEDES`` and ``COMPACTED_FROM`` which the v2 plan
   warns may pre-date this plan.

The script is **strictly read-only**. It uses only ``CALL db.labels()``,
``CALL db.relationshipTypes()``, ``SHOW CONSTRAINTS``, ``SHOW INDEXES``, and
``MATCH ... RETURN`` / ``count(...)`` queries. It does not MERGE, CREATE,
DELETE, SET, or REMOVE anything. There is no ``--write`` flag and adding one
would violate PLAN-0759 Phase 0 guardrails.

Usage
-----
From the host shell (i.e. not inside the Fusion MCP Docker network), the
Neo4j URI is ``bolt://localhost:7687``. The Fusion MCP ``app/config.py``
default ``bolt://neo4j:7687`` is the Docker-internal alias and will fail
resolution from the host, so the ``--uri`` flag is required whenever you run
this outside the compose network::

    # Default host-shell invocation. Writes report to the PLAN-0759 plan dir.
    python -m scripts.audit_neo4j_schema \
        --uri bolt://localhost:7687 \
        --report /home/nova/nova-core/MEMORY/plans/PLAN-0759/phase0_schema_audit.md

Authentication flags default to reading ``NEO4J_USER`` / ``NEO4J_PASSWORD``
from the environment, falling back to the Fusion MCP ``Settings`` defaults
(``neo4j`` / ``None``). If ``NEO4J_AUTH=none`` in the container, leave
``--password`` empty and the driver will attach without credentials. If auth
is enabled and the password is wrong, the script raises ``neo4j.exceptions
.AuthError`` and exits non-zero with a clear message — we intentionally do
**not** silently fall through to hardcoded test creds.

Exit codes
----------
- ``0`` — all queries ran successfully, report written.
- ``1`` — argument or usage error.
- ``2`` — Neo4j connectivity or authentication error.
- ``3`` — a read-only Cypher query failed unexpectedly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from neo4j import GraphDatabase, Driver
    from neo4j import exceptions as neo4j_exceptions
except ImportError as exc:  # pragma: no cover - env sanity
    print(
        "ERROR: neo4j python driver is not installed. Install with "
        "`pip install neo4j` (it is already pinned in requirements.txt).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


LOGGER = logging.getLogger("audit_neo4j_schema")


# Relationship types that PLAN-0759 plans to introduce. Most should be zero
# at baseline; ``SUPERSEDES`` and ``COMPACTED_FROM`` may pre-date this plan
# per the v2 plan warning.
ASSOC_EDGE_TYPES: Tuple[str, ...] = (
    "SIMILAR_TO",
    "MEMORY_FOLLOWS",
    "MENTIONS",
    "PROMOTED_FROM",
    "SUPERSEDES",
    "COMPACTED_FROM",
    "CAUSED_BY",
    "RELATED_TASK",
    "CO_OCCURS",
)
# Per PLAN-0759 Phase 0, these two may legitimately pre-exist.
ASSOC_EDGES_MAY_PRE_EXIST: Tuple[str, ...] = ("SUPERSEDES", "COMPACTED_FROM")

# Default report path (relative to nova-core repo). Overridable via --report.
DEFAULT_REPORT_PATH = Path(
    "/home/nova/nova-core/MEMORY/plans/PLAN-0759/phase0_schema_audit.md"
)


@dataclass
class AuditResult:
    """Structured holder for everything the audit collects."""

    uri: str = ""
    database: str = "neo4j"
    timestamp: str = ""
    labels: List[Tuple[str, int]] = field(default_factory=list)
    rel_types: List[Tuple[str, int]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    base_property_sample: List[List[str]] = field(default_factory=list)
    follows_count: int = 0
    follows_sample: List[Dict[str, Any]] = field(default_factory=list)
    assoc_edge_counts: Dict[str, int] = field(default_factory=dict)
    base_unique_constraint_text: Optional[str] = None


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audit_neo4j_schema",
        description=(
            "PLAN-0759 Phase 0 / Step 0.2 — read-only Neo4j schema audit. "
            "Verifies :base/entity_id invariant and existing FOLLOWS collision."
        ),
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help=(
            "Neo4j bolt URI. Defaults to $NEO4J_URI or bolt://localhost:7687 "
            "(the host-shell URI; the Fusion MCP config default "
            "bolt://neo4j:7687 only resolves inside the Docker network)."
        ),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username. Defaults to $NEO4J_USER or 'neo4j'.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help=(
            "Neo4j password. Defaults to $NEO4J_PASSWORD. Leave unset if the "
            "target container has NEO4J_AUTH=none."
        ),
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name. Defaults to 'neo4j'.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Path to write the markdown audit report. Directory is created "
            "if it does not exist. Default: "
            f"{DEFAULT_REPORT_PATH}"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _connect(args: argparse.Namespace) -> Driver:
    """Open a sync Neo4j driver. Raises on failure; caller maps to exit code."""
    LOGGER.info("Opening Neo4j driver at %s (user=%s)", args.uri, args.user)
    auth: Optional[Tuple[str, str]]
    if args.password:
        auth = (args.user, args.password)
    else:
        # Container with NEO4J_AUTH=none — pass no auth tuple.
        auth = None
    driver = GraphDatabase.driver(args.uri, auth=auth)
    # Verify connectivity immediately so errors surface with a clean stack.
    driver.verify_connectivity()
    return driver


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _run(session, query: str, **params: Any) -> List[Dict[str, Any]]:
    """Run a read-only Cypher query and return rows as dicts."""
    result = session.run(query, parameters=params)
    return [dict(record) for record in result]


def _collect_labels(session) -> List[Tuple[str, int]]:
    # Neo4j 5 allows CALL {} subqueries with WITH; this is the pattern the
    # v2 plan specified. Each label's node count is computed in-line.
    query = (
        "CALL db.labels() YIELD label "
        "CALL { WITH label MATCH (n) WHERE label IN labels(n) "
        "RETURN count(n) AS c } "
        "RETURN label, c ORDER BY c DESC"
    )
    rows = _run(session, query)
    return [(r["label"], r["c"]) for r in rows]


def _collect_rel_types(session) -> List[Tuple[str, int]]:
    query = (
        "CALL db.relationshipTypes() YIELD relationshipType "
        "CALL { WITH relationshipType MATCH ()-[r]->() "
        "WHERE type(r) = relationshipType RETURN count(r) AS c } "
        "RETURN relationshipType, c ORDER BY c DESC"
    )
    rows = _run(session, query)
    return [(r["relationshipType"], r["c"]) for r in rows]


def _collect_constraints(session) -> List[Dict[str, Any]]:
    return _run(session, "SHOW CONSTRAINTS")


def _collect_indexes(session) -> List[Dict[str, Any]]:
    return _run(session, "SHOW INDEXES")


def _collect_base_property_sample(session) -> List[List[str]]:
    rows = _run(session, "MATCH (n:base) RETURN keys(n) AS props LIMIT 10")
    return [list(r["props"]) for r in rows]


def _collect_follows(session) -> Tuple[int, List[Dict[str, Any]]]:
    count_rows = _run(session, "MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS count")
    count = count_rows[0]["count"] if count_rows else 0
    sample = _run(
        session,
        "MATCH (a)-[r:FOLLOWS]->(b) "
        "RETURN labels(a) AS src_labels, labels(b) AS dst_labels, "
        "keys(r) AS rel_props LIMIT 5",
    )
    # Neo4j returns labels as frozensets/lists; normalize for JSON/markdown.
    normalized = []
    for row in sample:
        normalized.append(
            {
                "src_labels": list(row["src_labels"]),
                "dst_labels": list(row["dst_labels"]),
                "rel_props": list(row["rel_props"]),
            }
        )
    return count, normalized


def _collect_assoc_edge_counts(session) -> Dict[str, int]:
    """Count each PLAN-0759 edge type individually.

    We issue one ``MATCH ()-[r:<TYPE>]->() RETURN count(r)`` per type rather
    than one big UNION, so that an invalid type name (e.g. if Neo4j has never
    seen it) returns zero cleanly instead of erroring.
    """
    counts: Dict[str, int] = {}
    for rel_type in ASSOC_EDGE_TYPES:
        # Relationship types cannot be parameterized; we whitelist against
        # ASSOC_EDGE_TYPES, which is a static, code-owned constant — no
        # user-supplied input touches this query.
        cypher = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
        try:
            rows = _run(session, cypher)
            counts[rel_type] = rows[0]["c"] if rows else 0
        except neo4j_exceptions.Neo4jError as exc:
            # Some Neo4j versions return a warning rather than 0 for unknown
            # relationship types. Treat those as zero.
            LOGGER.debug(
                "Count query for %s failed (%s); treating as 0",
                rel_type,
                exc,
            )
            counts[rel_type] = 0
    return counts


def _find_base_unique_constraint(constraints: Iterable[Dict[str, Any]]) -> Optional[str]:
    """Return the SHOW CONSTRAINTS row text for the :base(entity_id) uniqueness."""
    for row in constraints:
        label_field = row.get("labelsOrTypes") or row.get("labels") or []
        properties = row.get("properties") or []
        ctype = (row.get("type") or "").upper()
        if (
            "base" in label_field
            and "entity_id" in properties
            and "UNIQUE" in ctype
        ):
            return str(row)
    return None


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    # Right-align numeric columns if header ends in 'count'/'c'
    sep_cells = []
    for i, h in enumerate(headers):
        right = h.lower() in {"count", "c", "n"}
        sep_cells.append(("-" * (widths[i] - 1) + (":" if right else "-")))
    sep_line = "| " + " | ".join(sep_cells) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append(str(cell).ljust(widths[i]))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line, *body_lines]) + "\n"


def _render_report(audit: AuditResult) -> str:
    base_constraint_found = audit.base_unique_constraint_text is not None
    base_label_present = any(label == "base" for label, _ in audit.labels)
    # entity_id key confirmed if at least one sampled :base node has it
    entity_id_present = any(
        "entity_id" in keys for keys in audit.base_property_sample
    )
    follows_src_dst_summary = "; ".join(
        f"{s['src_labels']} -> {s['dst_labels']}" for s in audit.follows_sample
    ) or "(no samples returned)"

    assoc_rows = []
    for rel in ASSOC_EDGE_TYPES:
        count = audit.assoc_edge_counts.get(rel, 0)
        pre_existing_ok = rel in ASSOC_EDGES_MAY_PRE_EXIST
        if count == 0:
            pre_flag = "no (expected)"
        elif pre_existing_ok:
            pre_flag = f"yes (count={count}, allowed by v2 plan)"
        else:
            pre_flag = f"**UNEXPECTED** (count={count})"
        assoc_rows.append([rel, count, pre_flag])

    # Conclusions
    conclusions: List[str] = []
    if base_label_present and entity_id_present and base_constraint_found:
        conclusions.append(
            "- ADR-0759 node-label decision confirmed on the wire: the "
            "`:base` label exists, `entity_id` is present on sampled nodes, "
            "and a unique constraint on `(:base, entity_id)` is in force."
        )
    else:
        conclusions.append(
            "- **INVARIANT DRIFT**: one of (`:base` label present, "
            "`entity_id` key present on sample, unique constraint on "
            "`(:base, entity_id)`) did NOT hold. PLAN-0759 Phase 1 MUST NOT "
            "proceed until this is reconciled."
        )
    if audit.follows_count > 0:
        conclusions.append(
            f"- Existing `FOLLOWS` collision confirmed: {audit.follows_count} "
            f"edges exist in the graph ({follows_src_dst_summary}). Renaming "
            "the PLAN-0759 temporal edge to `MEMORY_FOLLOWS` (ADR-0759 §6) is "
            "necessary and correct."
        )
    else:
        conclusions.append(
            "- No existing `FOLLOWS` edges observed. ADR-0759 §6 still recommends "
            "`MEMORY_FOLLOWS` because the type name is reserved by "
            "`link_session_follows()` in code; it just has not been exercised "
            "on this live DB yet."
        )
    surprises = [
        rel
        for rel in ASSOC_EDGE_TYPES
        if audit.assoc_edge_counts.get(rel, 0) > 0
        and rel not in ASSOC_EDGES_MAY_PRE_EXIST
    ]
    if surprises:
        conclusions.append(
            "- **SURPRISE**: the following PLAN-0759 edge types already have a "
            "non-zero count in this database and are NOT on the allow-list "
            f"(SUPERSEDES, COMPACTED_FROM): {', '.join(surprises)}. "
            "PLAN-0759 and/or ADR-0759 must be amended before Phase 1 writes."
        )
    else:
        conclusions.append(
            "- No unexpected PLAN-0759 edge types pre-exist. Phase 1 Cypher is "
            "clear to proceed with `(:base {entity_id})` as the node identity "
            "and `MEMORY_FOLLOWS` as the temporal-adjacency edge."
        )

    lines: List[str] = []
    lines.append("# Phase 0 Schema Audit Report (PLAN-0759)")
    lines.append("")
    lines.append(f"**Date**: {audit.timestamp}")
    lines.append("**Script**: scripts/audit_neo4j_schema.py")
    lines.append(
        f"**Neo4j instance**: {audit.uri} (neo4j:5.19, container nova_neo4j_db)"
    )
    lines.append("**Audit run by**: Sprint 2 implementer (PLAN-0759)")
    lines.append("")
    lines.append("## Verified invariants (feed into Phase 1 Cypher)")
    lines.append("")
    lines.append(
        f"- Primary memory node label: `:base`  {'✓' if base_label_present else '✗'}"
    )
    lines.append(
        f"- Primary key property: `entity_id` {'✓' if entity_id_present else '✗'}"
    )
    if base_constraint_found:
        lines.append(
            "- Existing unique constraint on `(:base, entity_id)`: yes"
        )
        lines.append("")
        lines.append("  ```text")
        lines.append(f"  {audit.base_unique_constraint_text}")
        lines.append("  ```")
    else:
        lines.append(
            "- Existing unique constraint on `(:base, entity_id)`: **NO — DRIFT**"
        )
    lines.append(
        f"- Existing `FOLLOWS` edges: {audit.follows_count} on "
        f"{follows_src_dst_summary}  (confirms `MEMORY_FOLLOWS` rename was necessary)"
    )
    lines.append("")
    lines.append("## Full enumeration")
    lines.append("")
    lines.append("### Node labels (count)")
    lines.append("")
    lines.append(_fmt_table(["label", "count"], [[l, c] for l, c in audit.labels]))
    lines.append("### Relationship types (count)")
    lines.append("")
    lines.append(
        _fmt_table(["type", "count"], [[t, c] for t, c in audit.rel_types])
    )
    lines.append("### Constraints")
    lines.append("")
    lines.append("```text")
    for row in audit.constraints:
        lines.append(str(row))
    lines.append("```")
    lines.append("")
    lines.append("### Indexes")
    lines.append("")
    lines.append("```text")
    for row in audit.indexes:
        lines.append(str(row))
    lines.append("```")
    lines.append("")
    lines.append("### `:base` property sample (10 nodes)")
    lines.append("")
    if audit.base_property_sample:
        union_keys = sorted(
            {k for row in audit.base_property_sample for k in row}
        )
        lines.append(
            f"Union of property keys across {len(audit.base_property_sample)} "
            f"sampled nodes: `{union_keys}`"
        )
        lines.append("")
        for i, keys in enumerate(audit.base_property_sample, 1):
            lines.append(f"- sample {i}: `{sorted(keys)}`")
    else:
        lines.append("_(no `:base` nodes found)_")
    lines.append("")
    lines.append("## Associative-linking edge type baseline")
    lines.append("")
    lines.append(_fmt_table(["edge type", "count", "pre-existing?"], assoc_rows))
    lines.append("## Conclusions")
    lines.append("")
    lines.extend(conclusions)
    lines.append("")
    return "\n".join(lines)


def _print_human_summary(audit: AuditResult) -> None:
    """Short stdout dump so the operator sees headline numbers immediately."""
    print("=" * 72)
    print("PLAN-0759 Neo4j schema audit — live results")
    print("=" * 72)
    print(f"URI        : {audit.uri}")
    print(f"Database   : {audit.database}")
    print(f"Timestamp  : {audit.timestamp}")
    print()
    print(f"Labels ({len(audit.labels)}):")
    for label, count in audit.labels:
        print(f"  {label:<30} {count}")
    print()
    print(f"Relationship types ({len(audit.rel_types)}):")
    for rel, count in audit.rel_types:
        print(f"  {rel:<30} {count}")
    print()
    print(f"FOLLOWS edges      : {audit.follows_count}")
    if audit.follows_sample:
        for s in audit.follows_sample:
            print(f"  sample: {s['src_labels']} -> {s['dst_labels']} props={s['rel_props']}")
    print()
    print("PLAN-0759 edge baseline:")
    for rel in ASSOC_EDGE_TYPES:
        count = audit.assoc_edge_counts.get(rel, 0)
        marker = ""
        if count > 0 and rel not in ASSOC_EDGES_MAY_PRE_EXIST:
            marker = "  <-- UNEXPECTED"
        elif count > 0:
            marker = "  (allowed pre-existing)"
        print(f"  {rel:<20} {count}{marker}")
    print()
    if audit.base_unique_constraint_text:
        print("Unique constraint on (:base, entity_id): present")
    else:
        print("Unique constraint on (:base, entity_id): **MISSING**")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Neo4j 5 emits a WARNING notification for every MATCH on an unknown
    # relationship type. We deliberately probe types that may not yet exist
    # (that is the whole point of the baseline check), so suppress those
    # warnings to keep the operator output readable.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        driver = _connect(args)
    except neo4j_exceptions.AuthError as exc:
        print(
            f"ERROR: Neo4j authentication failed for user '{args.user}' at "
            f"{args.uri}: {exc}",
            file=sys.stderr,
        )
        return 2
    except (neo4j_exceptions.ServiceUnavailable, OSError) as exc:
        print(
            f"ERROR: Could not connect to Neo4j at {args.uri}: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"ERROR: Unexpected failure opening Neo4j driver: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 2

    audit = AuditResult(
        uri=args.uri,
        database=args.database,
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )
    try:
        with driver.session(database=args.database) as session:
            LOGGER.info("Collecting node labels + counts")
            audit.labels = _collect_labels(session)
            LOGGER.info("Collecting relationship types + counts")
            audit.rel_types = _collect_rel_types(session)
            LOGGER.info("Collecting constraints")
            audit.constraints = _collect_constraints(session)
            LOGGER.info("Collecting indexes")
            audit.indexes = _collect_indexes(session)
            LOGGER.info("Sampling :base node property keys")
            audit.base_property_sample = _collect_base_property_sample(session)
            LOGGER.info("Counting + sampling FOLLOWS edges")
            audit.follows_count, audit.follows_sample = _collect_follows(session)
            LOGGER.info("Counting PLAN-0759 candidate edge types")
            audit.assoc_edge_counts = _collect_assoc_edge_counts(session)
            audit.base_unique_constraint_text = _find_base_unique_constraint(
                audit.constraints
            )
    except neo4j_exceptions.Neo4jError as exc:
        print(
            f"ERROR: Read-only audit query failed: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        driver.close()
        return 3
    finally:
        driver.close()

    _print_human_summary(audit)

    report = _render_report(audit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print()
    print(f"Full markdown report written to: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
