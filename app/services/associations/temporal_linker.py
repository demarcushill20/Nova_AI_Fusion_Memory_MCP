"""Write-time temporal linker for PLAN-0759 Phase 4 (Sprint 10).

Overview
--------

This module lands :class:`TemporalLinker` — the third write-time associative
linker in PLAN-0759. When ``ASSOC_TEMPORAL_WRITE_ENABLED`` is True, the hook
inside :meth:`MemoryService.perform_upsert` (wired in Sprint 11) calls
:meth:`TemporalLinker.enqueue_link` after the memory is durably persisted.
The linker fires off a fire-and-forget background task that:

1. Acquires a per-session :class:`asyncio.Lock` so two concurrent stores
   for the same session do not race each other for the predecessor slot.
2. Looks up the immediate predecessor of the newly-stored memory: the
   ``:base`` node with the largest ``event_seq`` strictly less than the
   current memory's ``event_seq`` **in the same session**.
3. If a predecessor exists, persists a directed
   ``(current)-[:MEMORY_FOLLOWS]->(predecessor)`` edge via the injected
   :class:`MemoryEdgeService`. If no predecessor exists (first memory in
   the session), logs ``temporal_link.no_predecessor`` and returns 0.

Edge direction rationale
------------------------

``MEMORY_FOLLOWS`` is directed (``:base`` → ``:base``), not bidirectional —
see :data:`memory_edges.VALID_EDGE_TYPES` (present) and
:data:`memory_edges.BIDIRECTIONAL_EDGE_TYPES` (absent). The edge points
**from the later memory to the earlier one** so that a Cypher traversal
that says "walk back in time from this memory" follows outgoing edges.
Reading ``(current)-[:MEMORY_FOLLOWS]->(predecessor)`` aloud is: "current
memory follows predecessor" — which matches natural English semantics
("A follows B" means A came after B).

Concurrency model
-----------------

Mirrors Sprints 6 and 9's semaphore-bounded task-per-call pattern, with
an additional per-session layer:

- **Global semaphore** (``BACKGROUND_MAX_IN_FLIGHT = 32``) bounds the
  total number of simultaneously in-flight linker tasks so a session-
  heavy burst can never exhaust the event loop. Acquired FIRST.
- **Per-session lock** bounds the number of simultaneously in-flight
  tasks *for the same session* to exactly one. Acquired SECOND.

**Acquire order is load-bearing.** If the per-session lock were acquired
first, a single hot session could starve the global semaphore pool while
N-1 slots sat idle: each hot-session task would grab its lock, then
queue on the semaphore, while slower sessions waited behind them.
Acquiring the semaphore first means every hot-session task contends for
the same global budget and a per-session-serialized task cannot hold a
semaphore slot while waiting for its session lock.

Out-of-order arrival
--------------------

``event_seq`` is monotonically increasing under
:class:`SequenceService.next_seq` (the chronological upgrade from Phase
1 already addresses clock drift), so the *assigned* sequence is
strictly ordered. But ``perform_upsert()`` dispatch is async, so the
*arrival order* at the linker can diverge from the sequence order:
memory A (``event_seq = 3``) may call :meth:`enqueue_link` **after**
memory B (``event_seq = 5``) if they were dispatched on different
``asyncio.create_task`` slots and B happened to cross the hook window
first.

Sprint 10 handles this case with the **simpler self-healing MERGE**
strategy:

- When B arrives first with ``event_seq = 5``, its predecessor lookup
  finds nothing with ``event_seq < 5`` in the session (A is not yet
  stored) and writes no edge.
- When A arrives later with ``event_seq = 3``, its predecessor lookup
  *also* finds nothing (no memory in the session has ``event_seq < 3``)
  and writes no edge.
- When C arrives with ``event_seq = 7``, its predecessor lookup finds B
  (largest seq < 7) and writes ``C -> B``.

The net result of this specific out-of-order sequence is that B has no
MEMORY_FOLLOWS edge pointing to A even though A's seq is strictly less
than B's. The missing ``B -> A`` edge is a **known deferred gap**:
full out-of-order fix-up (a backward-scan on arrival that also detects
"are there later seqs in this session that should point to me?") is
deferred to Sprint 11's coverage monitor or a later sprint. The
simpler strategy is sufficient for the common case (in-order arrival
within a single session), is idempotent under MERGE (repeated runs
over the same triple produce one stored edge), and does not require
adding a second read query per call.

Predecessor-lookup injection
----------------------------

The linker needs to query Neo4j for the predecessor candidate, but the
:class:`MemoryEdgeService` does not expose this query and Sprint 10 is
forbidden from modifying Sprint 4–9 files. The linker therefore takes
**two optional constructor parameters**:

- ``predecessor_lookup`` — an explicit async callable with signature
  ``(session_id, event_seq, exclude_id) -> tuple[str, int] | None``.
  Tests use this to bypass Neo4j entirely with a controllable async
  mock; production code can pass an injected callable too if that is
  simpler than reaching into the driver.
- ``driver`` — a ``neo4j.AsyncDriver``. If ``predecessor_lookup`` is
  not supplied, the linker constructs an internal driver-based lookup
  that issues a small read-only Cypher query against the injected
  driver.

**Exactly one of the two must be non-None.** Constructing a linker with
both raises no error — the explicit ``predecessor_lookup`` wins (this
priority is documented and pinned by a unit test). Constructing a
linker with neither raises ``ValueError`` at ``__init__`` time so
misconfiguration fails loud rather than fails late under an
``enqueue_link`` call.

Cypher template
---------------

The driver-based lookup uses inline Cypher because no Sprint 4 template
matches a read-only "find predecessor in session" query:

.. code-block:: cypher

   MATCH (m:base)
   WHERE m.session_id = $session_id
     AND m.event_seq < $event_seq
     AND m.entity_id <> $memory_id
   RETURN m.entity_id AS predecessor_id, m.event_seq AS predecessor_seq
   ORDER BY m.event_seq DESC
   LIMIT 1

Both endpoints (there is only one, ``m``) are pinned to ``:base`` so
the query cannot walk into ``:Session`` chains. Every value is a
``$param`` placeholder (no f-string interpolation of user-controlled
values). Self-exclusion (``m.entity_id <> $memory_id``) is essential
for the re-store / backfill case where the current memory is already
in the graph and would otherwise match itself if its own ``event_seq``
happened to be strictly less than the queried value (it never is, but
the self-exclusion is defensive against the re-store case).

Metadata=None contract
----------------------

The edge is constructed with ``metadata=None`` because Neo4j 5 rejects
Map-valued relationship properties (Sprint 6 finding, repeated in
Sprint 9). Attribution is fully captured by ``created_by``, ``run_id``,
and the derivable ``event_seq`` of the two endpoint nodes. The linker
deliberately does **not** store ``event_seq`` as an edge property — it
is already on both endpoint nodes and any consumer can read it from
there.

Exception containment
---------------------

:meth:`_link_one_safe` wraps :meth:`_link_one` in ``asyncio.wait_for``
with a :data:`BACKGROUND_TIMEOUT` deadline. Every failure path
(timeout, predecessor-lookup error, edge-service error) is caught,
logged as a structured ``temporal_link.failed`` event, and swallowed —
the task returns ``0`` instead of raising. This is the contract that
protects ``perform_upsert()``.

Module invariants (verified by tests)
-------------------------------------

- Does **not** import ``app.config.settings``. The feature flag check
  lives in Sprint 11's hook, not in the linker itself.
- Does **not** create a Neo4j driver at import or construct time. The
  driver (when supplied) is injected.
- Does **not** store ``event_seq`` as an edge property — it is
  derivable from the endpoint nodes.
- Does **not** pass Map-valued metadata onto the relationship. Every
  write sets ``edge.metadata = None``.
- Session locks grow unboundedly in ``_session_locks`` — documented
  as a future cleanup task. Not a correctness issue in Sprint 10.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from .memory_edges import MemoryEdge

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from neo4j import AsyncDriver

    from .edge_service import MemoryEdgeService

logger = logging.getLogger(__name__)


# Default Neo4j database name — matches Sprint 5's ``MemoryEdgeService``
# default. Tests may override via the constructor keyword.
_DEFAULT_DATABASE: str = "neo4j"


# Read-only predecessor lookup Cypher. Pinned to ``:base`` so the query
# cannot walk into ``:Session`` chains or any other label. All values
# are bound via ``$param`` placeholders.
_PREDECESSOR_LOOKUP_CYPHER: str = (
    "MATCH (m:base)\n"
    "WHERE m.session_id = $session_id\n"
    "  AND m.event_seq < $event_seq\n"
    "  AND m.entity_id <> $memory_id\n"
    "RETURN m.entity_id AS predecessor_id, m.event_seq AS predecessor_seq\n"
    "ORDER BY m.event_seq DESC\n"
    "LIMIT 1"
)


#: Type alias for the predecessor-lookup callable contract. Returns the
#: ``(predecessor_id, predecessor_seq)`` tuple of the immediate
#: predecessor, or ``None`` if no predecessor exists in the session.
PredecessorLookup = Callable[
    [str, int, str],
    Awaitable[Optional[tuple[str, int]]],
]


class TemporalLinker:
    """Fire-and-forget write-time temporal linker.

    See module docstring for the full design rationale. The hot path is
    :meth:`enqueue_link`; every other method is internal plumbing that
    tests call directly.
    """

    #: Per-call upper bound on total linker work including predecessor
    #: lookup and Neo4j edge write. If exceeded, the task is cancelled
    #: and a ``temporal_link.failed`` event is logged.
    BACKGROUND_TIMEOUT: float = 30.0

    #: Maximum number of simultaneously in-flight linker tasks. Past
    #: this we drop new requests (with a logged warning) rather than
    #: block the caller or unbounded-queue them. Matches Sprint 6/9.
    BACKGROUND_MAX_IN_FLIGHT: int = 32

    def __init__(
        self,
        edge_service: "MemoryEdgeService",
        predecessor_lookup: Optional[PredecessorLookup] = None,
        *,
        database: str = _DEFAULT_DATABASE,
        driver: Optional["AsyncDriver"] = None,
    ) -> None:
        """Store injected dependencies.

        Parameters
        ----------
        edge_service:
            A fully-constructed :class:`MemoryEdgeService` bound to an
            async Neo4j driver. The linker does not create its own
            driver and does not manage the service's lifecycle.
        predecessor_lookup:
            An explicit async callable used to fetch the immediate
            predecessor. Signature:
            ``(session_id, event_seq, exclude_id) -> tuple[str, int] | None``.
            If supplied, takes priority over the driver-based default
            (documented and pinned by a unit test). Tests construct the
            linker with this set to a :class:`~unittest.mock.AsyncMock`
            so the Neo4j round-trip is fully hermetic.
        database:
            Neo4j database name for the driver-based lookup. Defaults
            to ``"neo4j"``. Ignored if ``predecessor_lookup`` is given.
        driver:
            An optional ``neo4j.AsyncDriver`` to use for the built-in
            predecessor-lookup query. Required if ``predecessor_lookup``
            is ``None``; if both are ``None`` a ``ValueError`` is raised
            at ``__init__`` time rather than under the first call to
            :meth:`enqueue_link`.

        Raises
        ------
        ValueError
            If neither ``predecessor_lookup`` nor ``driver`` is supplied.
        """
        if predecessor_lookup is None and driver is None:
            raise ValueError(
                "TemporalLinker requires either an explicit "
                "`predecessor_lookup` callable or a `driver` for the "
                "built-in driver-based lookup; both are None."
            )

        self._edge_service = edge_service
        self._database = database
        self._driver = driver

        # Resolve the effective lookup callable. Explicit callable wins
        # over driver — this priority is pinned by test 20.
        self._predecessor_lookup: PredecessorLookup
        if predecessor_lookup is not None:
            self._predecessor_lookup = predecessor_lookup
        else:
            # ``driver`` is guaranteed non-None by the guard above; the
            # assertion is defence-in-depth and also narrows the type
            # for static checkers.
            assert driver is not None
            self._predecessor_lookup = self._default_predecessor_lookup

        self._semaphore = asyncio.Semaphore(self.BACKGROUND_MAX_IN_FLIGHT)

        # Per-session lock map. Each session_id gets its own lock; the
        # dict itself is protected by a coarser lock so two concurrent
        # first-time accesses to the same session do not create two
        # distinct Lock instances. Documented: this dict grows
        # unboundedly over time — an LRU-cap cleanup is a future task.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_lock = asyncio.Lock()

        # Strong references to in-flight tasks so the asyncio loop does
        # not GC them mid-flight (classic create_task footgun). The set
        # is pruned in the done-callback.
        self._inflight: set[asyncio.Task[int]] = set()

    # ------------------------------------------------------------------ #
    # Default driver-based predecessor lookup                            #
    # ------------------------------------------------------------------ #

    async def _default_predecessor_lookup(
        self,
        session_id: str,
        event_seq: int,
        exclude_id: str,
    ) -> Optional[tuple[str, int]]:
        """Query Neo4j for the immediate predecessor via the injected driver.

        This is the fallback that runs when the caller did not supply
        an explicit ``predecessor_lookup`` callable. It opens a short-
        lived session on the injected driver, issues
        :data:`_PREDECESSOR_LOOKUP_CYPHER`, and returns the single
        ``(predecessor_id, predecessor_seq)`` tuple or ``None``.
        """
        assert self._driver is not None, (
            "default predecessor lookup invoked without a driver"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _PREDECESSOR_LOOKUP_CYPHER,
                {
                    "session_id": session_id,
                    "event_seq": int(event_seq),
                    "memory_id": exclude_id,
                },
            )
            record = await result.single()
            await result.consume()
            if record is None:
                return None
            return (record["predecessor_id"], int(record["predecessor_seq"]))

    # ------------------------------------------------------------------ #
    # Per-session lock management                                        #
    # ------------------------------------------------------------------ #

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the lock unique to this ``session_id``, creating on first access.

        The outer ``_session_locks_lock`` is held for O(1) time — just
        the dict lookup and possibly one assignment. This is a coarse
        but cheap guard that prevents the classic "two concurrent
        first-time accesses create two separate Lock objects" race.
        """
        async with self._session_locks_lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    async def enqueue_link(
        self,
        memory_id: str,
        session_id: Optional[str],
        thread_id: Optional[str],
        event_seq: Optional[int],
        project: Optional[str],
    ) -> None:
        """Dispatch a background temporal-linking task and return.

        This method does **not** block the caller on any real work. It:

        1. Skips and logs ``temporal_link.no_session`` if ``session_id``
           is ``None`` or ``event_seq`` is ``None``. Either missing
           field means we cannot scope the predecessor lookup or order
           the chain, so no edge can be written.
        2. Skips and logs ``temporal_link.dropped_semaphore_full`` if
           the global concurrency budget is exhausted.
        3. Otherwise ``asyncio.create_task(self._link_one_safe(...))``
           and return immediately.

        The caller's ``perform_upsert()`` return value is never affected
        by anything downstream of this method — all structured logging
        happens either at this dispatch point or inside the background
        task.

        Parameters
        ----------
        memory_id:
            ``entity_id`` of the freshly-persisted ``:base`` memory.
        session_id:
            Session identifier from the memory metadata. ``None`` skips.
        thread_id:
            Optional thread identifier. Currently informational — the
            Sprint 10 predecessor query is scoped by ``session_id`` only.
            Passed through for logging so operator debug output can
            correlate temporal edges with the thread they belong to.
        event_seq:
            Monotonic sequence number assigned by
            :class:`SequenceService`. ``None`` skips.
        project:
            Project scoping tag. Informational for Sprint 10 — the
            predecessor query is scoped by ``session_id`` not
            ``project``, because two sessions in different projects
            with the same ``session_id`` is impossible by construction.
            Logged for symmetry with Sprint 6/9.
        """
        if session_id is None or event_seq is None:
            logger.info(
                "temporal_link.no_session memory_id=%s session_id=%s "
                "event_seq=%s skipping=True",
                memory_id,
                session_id,
                event_seq,
            )
            return

        # Saturation guard: a non-blocking check on the semaphore. If
        # BACKGROUND_MAX_IN_FLIGHT tasks are already running we drop
        # the new request rather than queue unboundedly. The caller
        # still returns immediately.
        if self._semaphore.locked():
            if getattr(self._semaphore, "_value", 0) <= 0:
                logger.warning(
                    "temporal_link.dropped_semaphore_full "
                    "memory_id=%s session_id=%s max_in_flight=%d",
                    memory_id,
                    session_id,
                    self.BACKGROUND_MAX_IN_FLIGHT,
                )
                return

        # Per-call run_id — used by ``scripts/assoc_rollback`` (which
        # deletes by ``r.run_id`` regardless of type) for surgical
        # rollback if this particular linker invocation creates bad
        # edges. The ``wt-temporal-`` prefix distinguishes temporal-
        # linker runs from Sprint 6's ``wt-link-`` similarity runs and
        # Sprint 9's ``wt-entity-`` entity runs.
        run_id = f"wt-temporal-{uuid.uuid4().hex[:8]}"

        logger.info(
            "temporal_link.queued memory_id=%s run_id=%s session_id=%s "
            "thread_id=%s event_seq=%d project=%s",
            memory_id,
            run_id,
            session_id,
            thread_id,
            int(event_seq),
            project,
        )

        task = asyncio.create_task(
            self._link_one_safe(
                memory_id=memory_id,
                session_id=session_id,
                event_seq=int(event_seq),
                project=project,
                run_id=run_id,
            )
        )
        # Strong reference so the task isn't GC'd mid-flight.
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # ------------------------------------------------------------------ #
    # Internal plumbing                                                  #
    # ------------------------------------------------------------------ #

    async def _link_one_safe(
        self,
        memory_id: str,
        session_id: str,
        event_seq: int,
        project: Optional[str],
        run_id: str,
    ) -> int:
        """Exception-containment wrapper around :meth:`_link_one`.

        This is the function actually scheduled onto the event loop.
        Any exception thrown by :meth:`_link_one` (predecessor-lookup
        failed, Neo4j down, bad edge, cancelled, timeout, ...) is
        caught here, logged as ``temporal_link.failed``, and swallowed —
        the task returns ``0`` instead of raising. This is the contract
        that protects ``perform_upsert()``.
        """
        try:
            return await asyncio.wait_for(
                self._link_one(
                    memory_id=memory_id,
                    session_id=session_id,
                    event_seq=event_seq,
                    project=project,
                    run_id=run_id,
                ),
                timeout=self.BACKGROUND_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "temporal_link.failed memory_id=%s run_id=%s session_id=%s "
                "reason=timeout timeout_s=%s",
                memory_id,
                run_id,
                session_id,
                self.BACKGROUND_TIMEOUT,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "temporal_link.failed memory_id=%s run_id=%s session_id=%s "
                "reason=%s error=%s",
                memory_id,
                run_id,
                session_id,
                type(exc).__name__,
                exc,
            )
            return 0

    async def _link_one(
        self,
        memory_id: str,
        session_id: str,
        event_seq: int,
        project: Optional[str],
        run_id: str,
    ) -> int:
        """Do the actual temporal-linking work for one memory.

        Returns the number of ``MEMORY_FOLLOWS`` edges that were
        created (0 or 1). The ordering of the two locks is
        ``semaphore → session_lock``; see module docstring for the
        deadlock-free rationale.
        """
        t0 = time.perf_counter()

        async with self._semaphore:
            session_lock = await self._get_session_lock(session_id)
            async with session_lock:
                # Step 1: find the immediate predecessor. The callable
                # contract is ``(session_id, event_seq, exclude_id) ->
                # (pred_id, pred_seq) | None``. Self-exclusion is
                # delegated to the callable itself (the default Cypher
                # enforces it via ``m.entity_id <> $memory_id`` and an
                # explicit test-supplied callable is free to ignore
                # the ``exclude_id`` parameter if it already controls
                # the candidate set).
                pred = await self._predecessor_lookup(
                    session_id,
                    event_seq,
                    memory_id,
                )

                if pred is None:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    logger.info(
                        "temporal_link.no_predecessor memory_id=%s "
                        "run_id=%s session_id=%s event_seq=%d "
                        "elapsed_ms=%.3f",
                        memory_id,
                        run_id,
                        session_id,
                        event_seq,
                        elapsed_ms,
                    )
                    return 0

                predecessor_id, predecessor_seq = pred

                # Step 2: build the directed edge. Direction is
                # ``(current)-[:MEMORY_FOLLOWS]->(predecessor)`` — see
                # module docstring for the natural-English rationale.
                now_iso = datetime.now(tz=timezone.utc).isoformat()
                edge = MemoryEdge(
                    source_id=memory_id,
                    target_id=predecessor_id,
                    edge_type="MEMORY_FOLLOWS",
                    weight=1.0,
                    created_at=now_iso,
                    last_seen_at=now_iso,
                    created_by="temporal_linker",
                    run_id=run_id,
                    # metadata is LITERALLY None, not an empty dict.
                    # Neo4j 5 refuses Map-valued relationship props.
                    # Attribution is fully captured by created_by +
                    # run_id, and event_seq is derivable from the
                    # endpoint nodes (not stored on the edge).
                    metadata=None,
                )

                # Step 3: create the edge via the edge service. The
                # service handles MERGE idempotency: two concurrent
                # stores that resolve to the same
                # ``(source, MEMORY_FOLLOWS, target)`` triple land one
                # edge. The service returns a bool indicating whether
                # both endpoints matched; we surface 1/0 accordingly
                # so the caller's log line reflects reality.
                created = await self._edge_service.create_edge(edge)
                edges_count = 1 if created else 0

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "temporal_link.completed memory_id=%s run_id=%s "
                    "session_id=%s event_seq=%d predecessor_id=%s "
                    "predecessor_seq=%d edges_created=%d elapsed_ms=%.3f",
                    memory_id,
                    run_id,
                    session_id,
                    event_seq,
                    predecessor_id,
                    predecessor_seq,
                    edges_count,
                    elapsed_ms,
                )
                return edges_count


__all__ = ["TemporalLinker", "PredecessorLookup"]
