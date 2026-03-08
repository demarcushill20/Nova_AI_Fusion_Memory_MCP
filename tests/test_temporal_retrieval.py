"""Tests for Phase 3 temporal retrieval.

Tests the get_recent_events sorting, filtering, and clamping logic
in isolation — no openai/pinecone/neo4j dependencies needed.

Also tests the Neo4j query_recent_events Cypher generation logic.
"""

import pytest


# --- Isolated get_recent_events logic replica ---


def _sort_and_slice(raw_results, n):
    """Replica of the client-side sort+slice in MemoryService.get_recent_events."""
    if not raw_results:
        return []
    raw_results.sort(
        key=lambda r: r.get("metadata", {}).get("event_seq", 0),
        reverse=True,
    )
    return raw_results[:n]


def _build_pinecone_filter(
    project=None, thread_id=None, memory_type=None,
    since_seq=None, since_time=None,
):
    """Replica of the filter-building logic in MemoryService.get_recent_events."""
    filter_dict = {}
    if project:
        filter_dict["project"] = {"$eq": project}
    if thread_id:
        filter_dict["thread_id"] = {"$eq": thread_id}
    if memory_type:
        filter_dict["memory_type"] = {"$eq": memory_type}
    if since_seq is not None:
        filter_dict["event_seq"] = {"$gte": since_seq}
    if since_time:
        filter_dict["event_time"] = {"$gte": since_time}
    return filter_dict if filter_dict else None


def _clamp_n(n):
    """Replica of n clamping logic."""
    return max(1, min(n, 200))


def _apply_filters(results, project=None, thread_id=None, memory_type=None,
                   since_seq=None):
    """Client-side filter simulation (mirrors what Pinecone does server-side)."""
    filtered = list(results)
    if project:
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("project") == project]
    if thread_id:
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("thread_id") == thread_id]
    if memory_type:
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("memory_type") == memory_type]
    if since_seq is not None:
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("event_seq", 0) >= since_seq]
    return filtered


# --- Test Helpers ---


def _make_event(event_seq, project=None, thread_id=None, memory_type="scratch"):
    """Create a fake Pinecone-style result."""
    metadata = {
        "event_seq": event_seq,
        "event_time": f"2026-03-08T{10 + event_seq % 12:02d}:00:00Z",
        "memory_type": memory_type,
        "text": f"Event at seq {event_seq}",
    }
    if project:
        metadata["project"] = project
    if thread_id:
        metadata["thread_id"] = thread_id
    return {
        "id": f"id_{event_seq}",
        "score": 0.0,
        "metadata": metadata,
    }


# --- Sorting Tests ---


class TestTemporalSorting:
    """Results are sorted by event_seq descending."""

    def test_already_sorted(self):
        events = [_make_event(i) for i in [50, 40, 30, 20, 10]]
        result = _sort_and_slice(events, 5)
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [50, 40, 30, 20, 10]

    def test_unsorted_input(self):
        events = [_make_event(i) for i in [10, 50, 30, 20, 40]]
        result = _sort_and_slice(events, 5)
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [50, 40, 30, 20, 10]

    def test_reverse_sorted_input(self):
        events = [_make_event(i) for i in [1, 2, 3, 4, 5]]
        result = _sort_and_slice(events, 5)
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [5, 4, 3, 2, 1]

    def test_slice_top_n(self):
        events = [_make_event(i) for i in range(1, 51)]
        result = _sort_and_slice(events, 10)
        assert len(result) == 10
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [50, 49, 48, 47, 46, 45, 44, 43, 42, 41]

    def test_n_larger_than_results(self):
        events = [_make_event(i) for i in [1, 2, 3]]
        result = _sort_and_slice(events, 100)
        assert len(result) == 3
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [3, 2, 1]

    def test_empty_results(self):
        result = _sort_and_slice([], 10)
        assert result == []

    def test_single_result(self):
        events = [_make_event(42)]
        result = _sort_and_slice(events, 10)
        assert len(result) == 1
        assert result[0]["metadata"]["event_seq"] == 42

    def test_duplicate_event_seq(self):
        """Duplicate event_seq values are handled gracefully."""
        events = [_make_event(10), _make_event(10), _make_event(5)]
        result = _sort_and_slice(events, 5)
        assert len(result) == 3
        assert result[0]["metadata"]["event_seq"] == 10
        assert result[2]["metadata"]["event_seq"] == 5


# --- Filter Building Tests ---


class TestFilterBuilding:
    """Pinecone metadata filter construction."""

    def test_no_filters(self):
        assert _build_pinecone_filter() is None

    def test_project_filter(self):
        f = _build_pinecone_filter(project="NovaTrade")
        assert f == {"project": {"$eq": "NovaTrade"}}

    def test_thread_id_filter(self):
        f = _build_pinecone_filter(thread_id="thread_001")
        assert f == {"thread_id": {"$eq": "thread_001"}}

    def test_memory_type_filter(self):
        f = _build_pinecone_filter(memory_type="decision")
        assert f == {"memory_type": {"$eq": "decision"}}

    def test_since_seq_filter(self):
        f = _build_pinecone_filter(since_seq=42)
        assert f == {"event_seq": {"$gte": 42}}

    def test_since_seq_zero(self):
        """since_seq=0 should still generate a filter."""
        f = _build_pinecone_filter(since_seq=0)
        assert f == {"event_seq": {"$gte": 0}}

    def test_since_time_filter(self):
        f = _build_pinecone_filter(since_time="2026-03-08T00:00:00Z")
        assert f == {"event_time": {"$gte": "2026-03-08T00:00:00Z"}}

    def test_multiple_filters(self):
        f = _build_pinecone_filter(
            project="NovaTrade",
            memory_type="decision",
            since_seq=10,
        )
        assert f == {
            "project": {"$eq": "NovaTrade"},
            "memory_type": {"$eq": "decision"},
            "event_seq": {"$gte": 10},
        }

    def test_all_filters(self):
        f = _build_pinecone_filter(
            project="P1",
            thread_id="t1",
            memory_type="checkpoint",
            since_seq=5,
            since_time="2026-01-01T00:00:00Z",
        )
        assert len(f) == 5


# --- N Clamping Tests ---


class TestNClamping:
    """n is clamped to [1, 200]."""

    def test_normal_n(self):
        assert _clamp_n(20) == 20

    def test_n_zero(self):
        assert _clamp_n(0) == 1

    def test_n_negative(self):
        assert _clamp_n(-5) == 1

    def test_n_over_max(self):
        assert _clamp_n(500) == 200

    def test_n_at_max(self):
        assert _clamp_n(200) == 200

    def test_n_one(self):
        assert _clamp_n(1) == 1


# --- Client-Side Filter Simulation Tests ---


class TestFilterApplication:
    """Simulates Pinecone server-side filtering for end-to-end logic validation."""

    def test_filter_by_project(self):
        events = [
            _make_event(10, project="NovaTrade"),
            _make_event(20, project="NovaSHIFT"),
            _make_event(30, project="NovaTrade"),
        ]
        filtered = _apply_filters(events, project="NovaTrade")
        result = _sort_and_slice(filtered, 10)
        assert len(result) == 2
        assert result[0]["metadata"]["event_seq"] == 30
        assert result[1]["metadata"]["event_seq"] == 10

    def test_filter_by_thread_id(self):
        events = [
            _make_event(10, thread_id="t1"),
            _make_event(20, thread_id="t2"),
            _make_event(30, thread_id="t1"),
        ]
        filtered = _apply_filters(events, thread_id="t1")
        result = _sort_and_slice(filtered, 10)
        assert len(result) == 2

    def test_filter_by_memory_type(self):
        events = [
            _make_event(10, memory_type="scratch"),
            _make_event(20, memory_type="decision"),
            _make_event(30, memory_type="scratch"),
        ]
        filtered = _apply_filters(events, memory_type="decision")
        result = _sort_and_slice(filtered, 10)
        assert len(result) == 1
        assert result[0]["metadata"]["event_seq"] == 20

    def test_filter_by_since_seq(self):
        events = [_make_event(i) for i in range(1, 11)]
        filtered = _apply_filters(events, since_seq=7)
        result = _sort_and_slice(filtered, 10)
        assert len(result) == 4
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [10, 9, 8, 7]

    def test_combined_filters(self):
        events = [
            _make_event(10, project="P1", memory_type="scratch"),
            _make_event(20, project="P1", memory_type="decision"),
            _make_event(30, project="P2", memory_type="decision"),
            _make_event(40, project="P1", memory_type="decision"),
        ]
        filtered = _apply_filters(events, project="P1", memory_type="decision")
        result = _sort_and_slice(filtered, 10)
        assert len(result) == 2
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == [40, 20]

    def test_no_matches(self):
        events = [_make_event(i, project="P1") for i in range(1, 6)]
        filtered = _apply_filters(events, project="NonExistent")
        result = _sort_and_slice(filtered, 10)
        assert result == []

    def test_50_events_get_top_10(self):
        """Insert 50 events, get_recent_events(n=10) returns highest 10 event_seq."""
        events = [_make_event(i) for i in range(1, 51)]
        result = _sort_and_slice(events, 10)
        assert len(result) == 10
        seqs = [r["metadata"]["event_seq"] for r in result]
        assert seqs == list(range(50, 40, -1))

    def test_strict_descending_order(self):
        """Results are in strict event_seq descending order."""
        events = [_make_event(i) for i in [5, 15, 25, 35, 45, 10, 20, 30, 40, 50]]
        result = _sort_and_slice(events, 10)
        seqs = [r["metadata"]["event_seq"] for r in result]
        for i in range(len(seqs) - 1):
            assert seqs[i] > seqs[i + 1], f"Not strictly descending at index {i}"


# --- Neo4j Cypher Generation Tests ---


class TestNeo4jCypherGeneration:
    """Tests the Cypher WHERE clause building logic for query_recent_events."""

    def _build_where_and_params(self, n=20, filters=None):
        """Replica of the WHERE clause + params logic from GraphClient.query_recent_events."""
        where_clauses = []
        params = {"limit": n}

        if filters:
            if filters.get("project"):
                where_clauses.append("n.project = $project")
                params["project"] = filters["project"]
            if filters.get("thread_id"):
                where_clauses.append("n.thread_id = $thread_id")
                params["thread_id"] = filters["thread_id"]
            if filters.get("memory_type"):
                where_clauses.append("n.memory_type = $memory_type")
                params["memory_type"] = filters["memory_type"]
            if filters.get("since_seq") is not None:
                where_clauses.append("n.event_seq >= $since_seq")
                params["since_seq"] = filters["since_seq"]

        return where_clauses, params

    def test_no_filters(self):
        clauses, params = self._build_where_and_params()
        assert clauses == []
        assert params == {"limit": 20}

    def test_project_filter(self):
        clauses, params = self._build_where_and_params(
            filters={"project": "NovaTrade"}
        )
        assert "n.project = $project" in clauses
        assert params["project"] == "NovaTrade"

    def test_thread_id_filter(self):
        clauses, params = self._build_where_and_params(
            filters={"thread_id": "t1"}
        )
        assert "n.thread_id = $thread_id" in clauses
        assert params["thread_id"] == "t1"

    def test_memory_type_filter(self):
        clauses, params = self._build_where_and_params(
            filters={"memory_type": "decision"}
        )
        assert "n.memory_type = $memory_type" in clauses
        assert params["memory_type"] == "decision"

    def test_since_seq_filter(self):
        clauses, params = self._build_where_and_params(
            filters={"since_seq": 42}
        )
        assert "n.event_seq >= $since_seq" in clauses
        assert params["since_seq"] == 42

    def test_multiple_filters(self):
        clauses, params = self._build_where_and_params(
            n=10,
            filters={"project": "P1", "memory_type": "decision", "since_seq": 5},
        )
        assert len(clauses) == 3
        assert params["limit"] == 10
        assert params["project"] == "P1"
        assert params["memory_type"] == "decision"
        assert params["since_seq"] == 5

    def test_custom_n(self):
        _, params = self._build_where_and_params(n=50)
        assert params["limit"] == 50
