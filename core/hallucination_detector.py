"""Hallucination detector for generated SQL queries."""

import re
import sqlite3


class HallucinationDetector:
    """Detect hallucinations in generated SQL by comparing
    against the actual database schema and data."""

    def __init__(self, db_path: str):
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = db_path
        self.tables = {}
        self.valid_values = {}
        self._load_schema()
        self._load_sample_values()

    def _load_schema(self) -> None:
        """Load all tables and columns from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        for (table,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table});")
            cols = {row[1].lower() for row in cursor.fetchall()}
            self.tables[table.lower()] = cols
        conn.close()

    def _load_sample_values(self) -> None:
        """Load distinct values for key columns to detect
        value hallucinations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        check_cols = {
            "departments": ["department_name"],
            "appointments": ["status"],
            "prescriptions": ["medication_name"],
        }
        for table, cols in check_cols.items():
            if table in self.tables:
                for col in cols:
                    try:
                        cursor.execute(f"SELECT DISTINCT {col} FROM {table};")
                        values = {str(r[0]).lower() for r in cursor.fetchall()}
                        key = f"{table}.{col}"
                        self.valid_values[key] = values
                    except Exception:
                        pass
        conn.close()

    def detect(self, sql: str) -> dict:
        """Detect all hallucinations in a SQL query.

        Args:
            sql: The generated SQL query.

        Returns:
            Dict with hallucination details and score.
        """
        hallucinations = []

        # 1. Ghost tables
        ghost_tables = self._detect_ghost_tables(sql)
        for t in ghost_tables:
            hallucinations.append(
                {
                    "type": "ghost_table",
                    "detail": f"Table '{t}' does not exist",
                    "severity": "critical",
                }
            )

        # 2. Ghost columns
        ghost_cols = self._detect_ghost_columns(sql)
        for t, c in ghost_cols:
            hallucinations.append(
                {
                    "type": "ghost_column",
                    "detail": f"Column '{t}.{c}' does not exist",
                    "severity": "critical",
                }
            )

        # 3. Value hallucinations
        value_halls = self._detect_value_hallucinations(sql)
        for col, val in value_halls:
            hallucinations.append(
                {
                    "type": "value_hallucination",
                    "detail": (f"Value '{val}' not found in {col}"),
                    "severity": "warning",
                }
            )

        # 4. Common LLM inventions
        inventions = self._detect_common_inventions(sql)
        for inv in inventions:
            hallucinations.append(
                {
                    "type": "common_invention",
                    "detail": f"LLM invented '{inv}'",
                    "severity": "critical",
                }
            )

        # Score: 1.0 = no hallucination, 0.0 = severe
        critical = sum(1 for h in hallucinations if h["severity"] == "critical")
        warnings = sum(1 for h in hallucinations if h["severity"] == "warning")
        score = max(0.0, 1.0 - (critical * 0.3) - (warnings * 0.1))

        return {
            "sql": sql,
            "hallucinations": hallucinations,
            "total_hallucinations": len(hallucinations),
            "critical_count": critical,
            "warning_count": warnings,
            "hallucination_score": round(score, 4),
        }

    def _detect_ghost_tables(self, sql: str) -> list:
        """Find tables referenced but not in schema."""
        sql_kw = {
            "select",
            "from",
            "join",
            "left",
            "right",
            "inner",
            "outer",
            "on",
            "where",
            "group",
            "by",
            "order",
            "having",
            "limit",
            "as",
            "distinct",
            "count",
            "sum",
            "avg",
            "min",
            "max",
            "and",
            "or",
            "not",
            "in",
            "like",
            "is",
            "null",
            "asc",
            "desc",
            "case",
            "when",
            "then",
            "else",
            "end",
            "between",
            "exists",
            "union",
            "all",
            "insert",
            "update",
            "delete",
            "into",
            "values",
            "set",
            "create",
            "drop",
            "alter",
            "table",
            "index",
            "view",
        }
        matches = re.findall(
            r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            sql,
            flags=re.IGNORECASE,
        )
        ghost = []
        for t in matches:
            tl = t.lower()
            if tl not in self.tables and tl not in sql_kw:
                ghost.append(t)
        return ghost

    def _detect_ghost_columns(self, sql: str) -> list:
        """Find qualified columns that do not exist."""
        matches = re.findall(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)" r"\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
            sql,
        )
        # Build alias to table mapping
        alias_map = {}
        alias_matches = re.findall(
            r"\b(?:from|join)\s+"
            r"([a-zA-Z_][a-zA-Z0-9_]*)"
            r"\s+(?:as\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b",
            sql,
            flags=re.IGNORECASE,
        )
        for table, alias in alias_matches:
            alias_map[alias.lower()] = table.lower()

        ghost = []
        for prefix, col in matches:
            pl = prefix.lower()
            cl = col.lower()
            # Resolve alias to table
            table = alias_map.get(pl, pl)
            if table in self.tables:
                if cl not in self.tables[table]:
                    ghost.append((table, cl))
        return ghost

    def _detect_value_hallucinations(self, sql: str) -> list:
        """Find WHERE values that do not exist in the DB."""
        halls = []
        patterns = re.findall(
            r"(\w+)\s*=\s*'([^']+)'",
            sql,
            flags=re.IGNORECASE,
        )
        for col, val in patterns:
            cl = col.lower()
            vl = val.lower()
            for key, valid in self.valid_values.items():
                table_name, col_name = key.split(".")
                if cl == col_name and vl not in valid:
                    halls.append((key, val))
        return halls

    def _detect_common_inventions(self, sql: str) -> list:
        """Detect commonly hallucinated column names."""
        known_inventions = [
            "doctor_name",
            "patient_name",
            "user_name",
            "appointment_time",
            "phone_number",
            "doctor_specialty",
            "patient_city",
            "medical_history",
            "treatment",
            "consultation",
            "age",
        ]
        sql_lower = sql.lower()
        found = []
        for inv in known_inventions:
            if inv in sql_lower:
                found.append(inv)
        return found
