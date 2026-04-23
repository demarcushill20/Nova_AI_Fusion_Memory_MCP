"""Graph-expanded recall for associative linking (Phase 6, PLAN-0759).

Overview
--------

This module implements :class:`AssociativeRecall`, the read-path graph
expansion component. Given a set of seed results (from Pinecone semantic
search, temporal retrieval, etc.), it walks the associative edge graph up
to :data:`MAX_HOPS` hops to discover related memories that the vector
store alone might not surface.

Design principles
~~~~~~~~~~~~~~~~~

1. **Seeds pass through unchanged.** The caller's ranked results are
   never modified — expansion candidates are *appended* with decayed
   scores so downstream re-ranking can integrate them.

2. **Intent-aware edge selection.** Different recall intents (temporal,
   procedural, decision, debug, …) prioritize different edge types via
   :data:`INTENT_EDGE_FILTER`. The caller supplies the intent; there
   is no auto-classifier.

3. **Bounded fan-out.** Per-seed neighbor limit (5 symmetric + 5 directed)
   × seed count × 2 hops keeps the traversal manageable.

4. **Graceful degradation.** Per-seed and per-neighbor errors are caught
   and logged so one bad edge does not abort the entire expansion. Only
   catastrophic errors propagate to the caller.

5. **Feature-flag agnosticism.** This module does not read
   ``app.config.settings``. The caller checks
   ``ASSOC_GRAPH_RECALL_ENABLED`` before instantiating or calling.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Optional

from app.observability.metrics import record_graph_expansion

logger = logging.getLogger(__name__)

# Allow-list of edge types per recall intent (not a priority ordering).
INTENT_EDGE_FILTER: dict[str, list[str]] = {
    "temporal_recall": ["MEMORY_FOLLOWS", "SIMILAR_TO"],
    "procedural_recall": ["PROMOTED_FROM", "SIMILAR_TO", "MENTIONS"],
    "decision_recall": ["SUPERSEDES", "SIMILAR_TO", "PROMOTED_FROM"],
    "debug_recall": ["CAUSED_BY", "SIMILAR_TO", "MENTIONS"],
    "entity_recall": ["MENTIONS", "CO_OCCURS"],
    "provenance_recall": ["PROMOTED_FROM", "COMPACTED_FROM", "SUPERSEDES"],
    "general": ["SIMILAR_TO", "MENTIONS", "MEMORY_FOLLOWS"],
}

DIRECTED_EDGE_DIRECTION: dict[str, str] = {
    "MEMORY_FOLLOWS": "out",
    "SUPERSEDES": "out",
    "PROMOTED_FROM": "out",
    "CAUSED_BY": "out",
    "COMPACTED_FROM": "out",
}


class AssociativeRecall:
    """Graph-expanded recall: walk edges from seed results to find related memories.

    Parameters
    ----------
    edge_service:
        A :class:`~app.services.associations.edge_service.MemoryEdgeService`
        instance (or any object with an async ``get_neighbors`` method
        matching its signature).
    content_fetcher:
        Optional async callable ``(ids: list[str]) -> list[dict]`` that
        returns full memory content by IDs. Each returned dict should have
        at minimum ``id`` and ``metadata`` (with ``text`` key). If
        ``None``, expansion candidates are returned with IDs only (no
        content).
    """

    MAX_HOPS: int = 2
    MAX_EXPANSION: int = 20
    MIN_EDGE_WEIGHT: float = 0.5
    # Path-1 tuning (2026-04-15 session 2): was 0.7 — lowered to 0.5 to
    # reduce displacement of high-scoring seeds by expansion candidates.
    # At 0.5, hop-1 candidates cap at seed_score * edge_weight * 0.5,
    # landing at the bottom of the typical seed distribution rather than
    # mid-pack. Apparent improvement at this setting was recall_delta
    # -0.002 → +0.000, mrr_delta -0.037 → -0.023 (vs DECAY=0.7); a
    # follow-on DECAY=0.3 run regressed (-0.036 recall) but with a
    # 2.8pp shift in the supposedly-deterministic baseline, indicating
    # the judge variance is at the edge of what tuning at this
    # granularity can resolve. DECAY=0.5 is the best-known config.
    DECAY_PER_HOP: float = 0.5

    def __init__(self, edge_service: Any, content_fetcher: Any = None) -> None:
        self._edge_service = edge_service
        self._content_fetcher = content_fetcher

    async def expand(
        self,
        seed_results: list[dict[str, Any]],
        *,
        intent: str = "general",
        max_expansion: int | None = None,
    ) -> list[dict[str, Any]]:
        """Expand seed results via graph traversal.

        Returns seeds + up to *max_expansion* additional candidates with
        decayed scores. Seeds are returned unchanged.

        Parameters
        ----------
        seed_results:
            List of result dicts from upstream retrieval. Each must have
            an ``id`` (or ``entity_id``) key and a score key (``rrf_score``,
            ``score``, or ``composite_score``).
        intent:
            Recall intent that determines which edge types are traversed.
            Must be a key of :data:`INTENT_EDGE_FILTER` or ``"general"``.
        max_expansion:
            Cap on the number of expansion candidates. Defaults to
            :data:`MAX_EXPANSION`.
        """
        expand_t0 = time.perf_counter()
        if not isinstance(seed_results, list):
            record_graph_expansion(intent, time.perf_counter() - expand_t0, 0)
            return []
        if not seed_results:
            record_graph_expansion(intent, time.perf_counter() - expand_t0, 0)
            return []

        cap = max_expansion if max_expansion is not None else self.MAX_EXPANSION
        edge_types = INTENT_EDGE_FILTER.get(
            intent, INTENT_EDGE_FILTER["general"]
        )

        # --- Extract seed IDs and scores --------------------------------- #
        seed_ids: set[str] = set()
        seed_scores: dict[str, float] = {}
        for result in seed_results:
            sid = result.get("id")
            if sid is None:
                sid = result.get("entity_id")
            if sid is not None and sid != "":
                seed_ids.add(sid)
                # Pick the best available score key (explicit None checks
                # so that a legitimate 0.0 is not treated as missing).
                # Prefer composite_score (includes temporal weighting) over
                # raw rrf_score.
                score = result.get("composite_score")
                if score is None:
                    score = result.get("rrf_score")
                if score is None:
                    score = result.get("score")
                if score is None:
                    score = 0.0
                seed_scores[sid] = float(score)

        if not seed_ids:
            record_graph_expansion(intent, time.perf_counter() - expand_t0, 0)
            return list(seed_results)

        # --- Hop 1: seed → neighbors (parallel fan-out) ------------------- #
        # candidates: list of (node_id, score, edge_type)
        hop1_best: dict[str, tuple[float, str]] = {}

        async def _fetch_hop1(sid: str) -> tuple[str, list[dict]]:
            # MENTIONS edges go (:base)->(:Entity), so they are invisible
            # to get_neighbors (which pins both endpoints to :base). Walk
            # them through the entity-mediated 2-hop helper instead, and
            # strip MENTIONS from the regular get_neighbors call to avoid
            # a wasted query.
            wants_mentions = "MENTIONS" in edge_types
            non_mentions = [et for et in edge_types if et != "MENTIONS"]
            symmetric = [et for et in non_mentions if et not in DIRECTED_EDGE_DIRECTION]
            directed = [et for et in non_mentions if et in DIRECTED_EDGE_DIRECTION]
            all_neighbors: list[dict] = []
            if symmetric:
                try:
                    all_neighbors.extend(await self._edge_service.get_neighbors(
                        node_id=sid, edge_types=symmetric, direction="both",
                        min_weight=self.MIN_EDGE_WEIGHT, limit=5,
                    ))
                except Exception:
                    logger.debug("symmetric get_neighbors failed for seed %s", sid)
            if directed:
                try:
                    all_neighbors.extend(await self._edge_service.get_neighbors(
                        node_id=sid, edge_types=directed, direction="out",
                        min_weight=self.MIN_EDGE_WEIGHT, limit=5,
                    ))
                except Exception:
                    logger.debug("directed get_neighbors failed for seed %s", sid)
            if wants_mentions:
                try:
                    fn = getattr(
                        self._edge_service,
                        "get_memory_neighbors_via_mentions",
                        None,
                    )
                    if fn is not None:
                        all_neighbors.extend(
                            await fn(node_id=sid, hub_threshold=50, limit=5)
                        )
                except Exception:
                    logger.debug("mentions traversal failed for seed %s", sid)
            return sid, all_neighbors

        hop1_results = await asyncio.gather(
            *[_fetch_hop1(sid) for sid in seed_ids]
        )

        for sid, neighbors in hop1_results:
            for neighbor in neighbors:
                try:
                    nid = neighbor.get("node_id", "")
                    if not nid or nid in seed_ids:
                        continue
                    edge_weight = float(neighbor.get("weight", 0.0))
                    if not math.isfinite(edge_weight) or edge_weight < 0.0 or edge_weight > 1.0:
                        logger.debug("Invalid edge weight %.4f for %s, skipping", edge_weight, nid)
                        continue
                    decayed = seed_scores.get(sid, 0.0) * edge_weight * self.DECAY_PER_HOP
                    existing = hop1_best.get(nid)
                    if existing is None or decayed > existing[0]:
                        hop1_best[nid] = (decayed, neighbor.get("edge_type", ""))
                except Exception:
                    logger.debug("Skipping malformed neighbor for seed %s", sid)
                    continue

        visited = set(seed_ids) | set(hop1_best.keys())
        hop1_candidates = [(nid, score, etype) for nid, (score, etype) in hop1_best.items()]

        # --- Hop 2: hop-1 candidates → their neighbors (parallel) --------- #
        hop2_candidates: list[tuple[str, float, str]] = []

        if self.MAX_HOPS >= 2 and hop1_candidates:
            hop1_score_map = {nid: h1_score for nid, h1_score, _ in hop1_candidates}

            async def _fetch_hop2(nid: str) -> tuple[str, list[dict]]:
                # Same MENTIONS handling as _fetch_hop1: bypass
                # get_neighbors for MENTIONS and walk via the entity-
                # mediated helper instead.
                wants_mentions = "MENTIONS" in edge_types
                non_mentions = [et for et in edge_types if et != "MENTIONS"]
                symmetric = [
                    et for et in non_mentions if et not in DIRECTED_EDGE_DIRECTION
                ]
                directed = [
                    et for et in non_mentions if et in DIRECTED_EDGE_DIRECTION
                ]
                all_neighbors: list[dict] = []
                if symmetric:
                    try:
                        all_neighbors.extend(await self._edge_service.get_neighbors(
                            node_id=nid, edge_types=symmetric, direction="both",
                            min_weight=self.MIN_EDGE_WEIGHT, limit=5,
                        ))
                    except Exception:
                        logger.debug("symmetric get_neighbors failed for hop-1 node %s", nid)
                if directed:
                    try:
                        all_neighbors.extend(await self._edge_service.get_neighbors(
                            node_id=nid, edge_types=directed, direction="out",
                            min_weight=self.MIN_EDGE_WEIGHT, limit=5,
                        ))
                    except Exception:
                        logger.debug("directed get_neighbors failed for hop-1 node %s", nid)
                if wants_mentions:
                    try:
                        fn = getattr(
                            self._edge_service,
                            "get_memory_neighbors_via_mentions",
                            None,
                        )
                        if fn is not None:
                            all_neighbors.extend(
                                await fn(node_id=nid, hub_threshold=50, limit=5)
                            )
                    except Exception:
                        logger.debug(
                            "mentions traversal failed for hop-1 node %s", nid
                        )
                return nid, all_neighbors

            hop2_results = await asyncio.gather(
                *[_fetch_hop2(nid) for nid, _, _ in hop1_candidates]
            )

            hop2_best: dict[str, tuple[float, str]] = {}
            for nid, neighbors in hop2_results:
                h1_score = hop1_score_map[nid]
                for neighbor in neighbors:
                    try:
                        n2id = neighbor.get("node_id", "")
                        if not n2id or n2id in visited:
                            continue
                        edge_weight = float(neighbor.get("weight", 0.0))
                        if not math.isfinite(edge_weight) or edge_weight < 0.0 or edge_weight > 1.0:
                            logger.debug("Invalid edge weight %.4f for %s, skipping", edge_weight, n2id)
                            continue
                        decayed = h1_score * edge_weight * self.DECAY_PER_HOP
                        existing = hop2_best.get(n2id)
                        if existing is None or decayed > existing[0]:
                            hop2_best[n2id] = (decayed, neighbor.get("edge_type", ""))
                    except Exception:
                        logger.debug("Skipping malformed neighbor for hop-1 node %s", nid)
                        continue

            hop2_candidates = [(nid, score, etype) for nid, (score, etype) in hop2_best.items()]

        # --- Merge, sort, cap -------------------------------------------- #
        all_candidates: list[tuple[str, float, int, str]] = []
        for nid, score, etype in hop1_candidates:
            all_candidates.append((nid, score, 1, etype))
        for nid, score, etype in hop2_candidates:
            all_candidates.append((nid, score, 2, etype))

        # Sort by score DESC
        all_candidates.sort(key=lambda c: -c[1])
        top = all_candidates[:cap]

        if not top:
            record_graph_expansion(intent, time.perf_counter() - expand_t0, 0)
            return list(seed_results)

        # --- Optionally fetch full content ------------------------------- #
        content_map: dict[str, dict] = {}
        if self._content_fetcher is not None:
            try:
                fetch_ids = [c[0] for c in top]
                fetched = await self._content_fetcher(fetch_ids)
                for item in fetched:
                    item_id = item.get("id", "")
                    if item_id:
                        content_map[item_id] = item
            except Exception:
                logger.debug("Content fetcher failed, returning expansion IDs only", exc_info=True)

        # --- Build expansion result dicts -------------------------------- #
        # The decayed score lives in [0, 1] because seed_scores are read
        # from composite_score (also [0, 1]). Write it into composite_score
        # so the downstream sort in memory_service.perform_query compares
        # seeds and expansion candidates in the same score domain. Writing
        # it into rrf_score as well would inflate expansion well above the
        # ~0.01–0.05 RRF range and displace every seed (the bug Phase 6
        # eval surfaced on 2026-04-15).
        expansion_results: list[dict[str, Any]] = []
        for nid, score, hop, etype in top:
            if nid in content_map:
                base = dict(content_map[nid])
                base["score"] = score
                base["composite_score"] = score
                base["source"] = "graph_expansion"
                base["expansion_score"] = score
                base["expansion_hop"] = hop
                base["expansion_edge_type"] = etype
            else:
                base = {
                    "id": nid,
                    "metadata": {},
                    "score": score,
                    "composite_score": score,
                    "source": "graph_expansion",
                    "expansion_score": score,
                    "expansion_hop": hop,
                    "expansion_edge_type": etype,
                }
            expansion_results.append(base)

        logger.info(
            "Graph expansion: %d seeds -> %d candidates (intent=%s, hops=%d)",
            len(seed_ids),
            len(expansion_results),
            intent,
            self.MAX_HOPS,
        )

        record_graph_expansion(
            intent, time.perf_counter() - expand_t0, len(expansion_results)
        )
        return list(seed_results) + expansion_results
