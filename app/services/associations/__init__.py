"""Associative linking components for PLAN-0759.

See ADR-0759 (``/home/nova/nova-core/10-adrs/ADR-0759-assoc-linking-location.md``)
for the architecture decision record: why this package lives in Fusion Memory
MCP (not ``nova-core/agents``), the exact ``perform_upsert()`` hook window, the
``MEMORY_FOLLOWS`` (not ``FOLLOWS``) edge-type rename, and the verified
``(:base {entity_id})`` node schema.

Sprint 4 landed the bedrock data model and Cypher templates only
(``memory_edges``, ``edge_cypher``). No live Neo4j writes occur from this
package yet — Sprint 5 adds the ``edge_service`` executor, and Phase 2+ wires
linker components under their respective ``ASSOC_*`` feature flags.
"""
