"""Memory system with short-term and long-term storage."""

import sqlite3
from datetime import datetime


class ShortTermMemory:
    """In-memory buffer for current session."""

    def __init__(self, max_items: int = 10):
        """Initialize short-term memory.

        Args:
            max_items: Maximum number of items to retain.
        """
        self.max_items = max_items
        self.history: list[dict] = []

    def add(self, query: str, sql: str, success: bool) -> None:
        """Add an entry to short-term memory.

        Args:
            query: The user query.
            sql: Generated SQL.
            success: Whether execution succeeded.
        """
        self.history.append(
            {
                "query": query,
                "sql": sql,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self.history) > self.max_items:
            self.history.pop(0)

    def get_context(self) -> str:
        """Get formatted context from recent history.

        Returns:
            Formatted string of recent queries.
        """
        if not self.history:
            return "Aucun historique."
        lines = []
        for item in self.history:
            status = "OK" if item["success"] else "FAIL"
            lines.append(f"- [{status}] {item['query']}: {item['sql'][:80]}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all short-term memory."""
        self.history = []


class LongTermMemory:
    """SQLite-backed persistent memory."""

    def __init__(self, db_path: str = "nlq_memory.db"):
        """Initialize long-term memory.

        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                route TEXT,
                generated_sql TEXT,
                is_valid INTEGER,
                num_rows INTEGER,
                provider TEXT,
                attempts INTEGER,
                duration_seconds REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)
        conn.commit()
        conn.close()

    def save(
        self,
        query: str,
        route: str,
        generated_sql: str,
        is_valid: bool,
        num_rows: int,
        provider: str,
        attempts: int,
        duration_seconds: float,
    ) -> None:
        """Save a query session to long-term memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO query_history
            (query, route, generated_sql, is_valid, num_rows,
             provider, attempts, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                route,
                generated_sql,
                1 if is_valid else 0,
                num_rows,
                provider,
                attempts,
                duration_seconds,
            ),
        )
        conn.commit()
        conn.close()

    def get_history(self, limit: int = 10) -> list[dict]:
        """Retrieve recent query history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT query, generated_sql, is_valid, provider, created_at
            FROM query_history
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "query": row[0],
                "sql": row[1],
                "is_valid": bool(row[2]),
                "provider": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
