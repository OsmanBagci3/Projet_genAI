"""Tests for the memory system."""

from core.memory import LongTermMemory, ShortTermMemory


def test_short_term_add_and_context():
    """Test adding entries and getting context."""
    mem = ShortTermMemory(max_items=3)
    mem.add("query1", "SELECT 1", True)
    mem.add("query2", "SELECT 2", False)

    context = mem.get_context()
    assert "query1" in context
    assert "OK" in context
    assert "FAIL" in context


def test_short_term_max_items():
    """Test max items limit."""
    mem = ShortTermMemory(max_items=2)
    mem.add("q1", "s1", True)
    mem.add("q2", "s2", True)
    mem.add("q3", "s3", True)

    assert len(mem.history) == 2
    assert mem.history[0]["query"] == "q2"


def test_short_term_clear():
    """Test clearing memory."""
    mem = ShortTermMemory()
    mem.add("q1", "s1", True)
    mem.clear()
    assert len(mem.history) == 0
    assert mem.get_context() == "Aucun historique."


def test_long_term_save_and_history(tmp_path):
    """Test saving and retrieving."""
    db = str(tmp_path / "test.db")
    mem = LongTermMemory(db_path=db)

    mem.save(
        query="Test query",
        route="SQL_QUERY",
        generated_sql="SELECT 1",
        is_valid=True,
        num_rows=5,
        provider="claude",
        attempts=1,
        duration_seconds=2.5,
    )

    history = mem.get_history(limit=5)
    assert len(history) == 1
    assert history[0]["query"] == "Test query"
    assert history[0]["is_valid"] is True


def test_long_term_empty(tmp_path):
    """Test empty history."""
    db = str(tmp_path / "empty.db")
    mem = LongTermMemory(db_path=db)
    assert mem.get_history() == []
