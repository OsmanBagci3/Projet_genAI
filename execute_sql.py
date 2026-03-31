from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from langfuse import Langfuse
    _langfuse: Optional[Langfuse] = Langfuse()
except Exception:
    _langfuse = None


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "hospital.db"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


class ChainOfThoughtPlanner:
    """
    Étape de raisonnement avant génération SQL.
    Demande au LLM d'identifier tables/colonnes/filtres AVANT d'écrire le SQL.
    Réduit les hallucinations en forçant une réflexion structurée.
    """

    def __init__(self, generator_func) -> None:
        self._generate = generator_func

    def plan(self, question: str, schema_summary: str) -> str:
        planning_prompt = (
            "You are a SQL analyst. Before writing any SQL, analyze the question carefully.\n\n"
            "DATABASE SCHEMA (only these exist):\n"
            f"{schema_summary}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer ONLY these 4 points using the schema above:\n"
            "1. TABLES NEEDED: Which tables are required?\n"
            "2. COLUMNS NEEDED: Which exact columns (use table.column format)?\n"
            "3. FILTERS: What WHERE conditions are needed? (use exact column names)\n"
            "4. JOINS: Which foreign keys to join on?\n\n"
            "Use ONLY tables and columns from the schema. Do NOT invent anything.\n"
            "Be concise, one line per point."
        )
        return self._generate(planning_prompt)


class SQLGenerator:
    def generate(self, prompt: str) -> str:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def regenerate_with_feedback(
        self,
        base_prompt: str,
        failed_sql: str,
        error_message: str,
        schema_summary: str,
    ) -> str:
        repair_prompt = (
            f"{base_prompt}\n\n"
            "---\n\n"
            "PREVIOUS SQL (FAILED):\n"
            f"{failed_sql}\n\n"
            "VALIDATION ERROR:\n"
            f"{error_message}\n\n"
            "AVAILABLE DATABASE SCHEMA:\n"
            f"{schema_summary}\n\n"
            "Fix the SQL query using ONLY available tables/columns.\n"
            "If no safe mapping exists, return exactly: NO_SQL\n"
            "Output ONLY SQL or NO_SQL."
        )
        return self.generate(repair_prompt)


class SchemaInspector:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.tables: Dict[str, Set[str]] = {}

    def load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [row[0] for row in cursor.fetchall()]

            for table in table_names:
                cursor.execute(f"PRAGMA table_info({table});")
                cols = {row[1].lower() for row in cursor.fetchall()}
                self.tables[table.lower()] = cols
        finally:
            conn.close()

    def summary(self) -> str:
        lines: List[str] = []
        for table in sorted(self.tables):
            cols = ", ".join(sorted(self.tables[table]))
            lines.append(f"- {table}: {cols}")
        return "\n".join(lines)


class SQLValidator:
    FORBIDDEN_KEYWORDS = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "replace",
    ]
    SQL_KEYWORDS = {
        "select", "from", "join", "left", "right", "inner", "outer", "on",
        "where", "group", "by", "order", "having", "limit", "as", "distinct",
        "count", "sum", "avg", "min", "max", "and", "or", "not", "in", "like",
        "is", "null", "asc", "desc", "case", "when", "then", "else", "end",
    }
    def __init__(self, schema: SchemaInspector) -> None:
        self.schema = schema

    def clean_sql(self, raw_output: str) -> str:
        """
        Nettoie la sortie du modèle pour extraire uniquement le SQL.
        """
        text = raw_output.strip()

        # Retirer les blocs markdown ```sql ... ```
        text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # Garder à partir du premier SELECT si du texte précède
        match = re.search(r"\bselect\b", text, flags=re.IGNORECASE)
        if match:
            text = text[match.start():]

        return text.strip()

    def _extract_table_names(self, sql: str) -> Set[str]:
        table_matches = re.findall(
            r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            sql,
            flags=re.IGNORECASE,
        )
        return {name.lower() for name in table_matches}

    def _extract_qualified_columns(self, sql: str) -> Set[Tuple[str, str]]:
        matches = re.findall(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
            sql,
        )
        return {(t.lower(), c.lower()) for t, c in matches}

    def _extract_table_aliases(self, sql: str) -> Set[str]:
        aliases = re.findall(
            r"\b(?:from|join)\s+[a-zA-Z_][a-zA-Z0-9_]*\s+(?:as\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b",
            sql,
            flags=re.IGNORECASE,
        )
        return {a.lower() for a in aliases}

    def _extract_select_aliases(self, sql: str) -> Set[str]:
        aliases = re.findall(
            r"\bas\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
            sql,
            flags=re.IGNORECASE,
        )
        return {a.lower() for a in aliases}

    def _check_unqualified_identifiers(self, sql: str) -> Tuple[bool, str]:
        # Remove quoted strings to avoid false positives on literal words.
        scrubbed = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql)
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", scrubbed)

        table_names = self._extract_table_names(sql)
        table_aliases = self._extract_table_aliases(sql)
        select_aliases = self._extract_select_aliases(sql)

        allowed_columns: Set[str] = set()
        for table in table_names:
            allowed_columns.update(self.schema.tables.get(table, set()))

        allowed = (
            self.SQL_KEYWORDS
            | table_names
            | table_aliases
            | select_aliases
            | allowed_columns
        )

        unknown: Set[str] = set()
        for tok in tokens:
            t = tok.lower()

            # Skip numbers and placeholders if any slipped through.
            if t.isdigit():
                continue
            if t in allowed:
                continue

            unknown.add(t)

        if unknown:
            return False, f"Identifiant(s) inconnu(s): {', '.join(sorted(unknown))}"

        return True, "Identifiants non qualifiés valides."

    def _check_schema_names(self, sql: str) -> Tuple[bool, str]:
        tables = self._extract_table_names(sql)
        unknown_tables = sorted(t for t in tables if t not in self.schema.tables)
        if unknown_tables:
            return False, f"Table(s) inexistante(s): {', '.join(unknown_tables)}"

        qualified = self._extract_qualified_columns(sql)
        bad_columns: List[str] = []
        for table, col in sorted(qualified):
            if table in self.schema.tables and col not in self.schema.tables[table]:
                bad_columns.append(f"{table}.{col}")
        if bad_columns:
            return False, f"Colonne(s) inexistante(s): {', '.join(bad_columns)}"

        unqualified_ok, unqualified_msg = self._check_unqualified_identifiers(sql)
        if not unqualified_ok:
            return False, unqualified_msg

        return True, "Noms de tables/colonnes valides."

    def _prepare_check(self, sql: str) -> Tuple[bool, str]:
        try:
            conn = sqlite3.connect(self.schema.db_path)
            cursor = conn.cursor()

            # Force SQLite to resolve tables/columns without returning data rows.
            # This catches errors like unknown unqualified columns (e.g., "diagnose").
            safe_sql = sql.strip().rstrip(";")
            cursor.execute(f"SELECT * FROM ({safe_sql}) AS _q LIMIT 0")
            cursor.fetchall()
            return True, "PREPARE OK"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def validate(self, sql: str) -> Tuple[bool, str]:
        if not sql:
            return False, "La requête SQL est vide."

        if sql.strip().upper() == "NO_SQL":
            return False, "Le modèle a retourné NO_SQL (pas de mapping fiable)."

        normalized = sql.strip().lower()

        if not normalized.startswith("select"):
            return False, "Seules les requêtes SELECT sont autorisées."

        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", normalized):
                return False, f"Mot-clé interdit détecté : {keyword}"

        schema_ok, schema_msg = self._check_schema_names(sql)
        if not schema_ok:
            return False, schema_msg

        prepare_ok, prepare_msg = self._prepare_check(sql)
        if not prepare_ok:
            return False, f"Validation SQL/Schéma échouée: {prepare_msg}"

        return True, "SQL valide."


class SQLExecutor:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def execute(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base SQLite introuvable : {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return columns, rows
        finally:
            conn.close()


def print_table(columns: List[str], rows: List[Tuple[Any, ...]]) -> None:
    if not columns:
        print("Aucune colonne retournée.")
        return

    print("\nRésultat SQL :\n")

    header = " | ".join(columns)
    separator = "-" * len(header)
    print(header)
    print(separator)

    if not rows:
        print("(aucune ligne)")
        return

    for row in rows[:20]:
        print(" | ".join(str(value) for value in row))

    if len(rows) > 20:
        print(f"... ({len(rows) - 20} lignes supplémentaires non affichées)")


def main() -> None:
    from query_construction import QueryConstructor
    from rerank_results import HeuristicReranker
    from hybrid_retrieval import HybridRetriever
    from router import QueryRouter

    hybrid = HybridRetriever()
    reranker = HeuristicReranker()
    router = QueryRouter()
    constructor = QueryConstructor()
    generator = SQLGenerator()
    planner = ChainOfThoughtPlanner(generator.generate)
    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    validator = SQLValidator(inspector)
    executor = SQLExecutor(DB_PATH)

    print("Système NLQ vers SQL prêt.")
    print("Tape une question en anglais sur la base de données.")
    print("Tape 'exit' pour quitter.\n")

    try:
        while True:
            query = input("Question > ").strip()

            if query.lower() in {"exit", "quit"}:
                print("Fin du programme.")
                break

            if not query:
                print("Veuillez entrer une question.")
                continue

            print("\n" + "=" * 120)
            print(f"QUESTION: {query}")
            print("=" * 120)

            # Démarrer une trace Langfuse pour cette question
            trace = _langfuse.trace(
                name="nlq-to-sql",
                input={"question": query},
            ) if _langfuse else None

            # 0. Routing
            route = router.route(query)
            if trace:
                trace.event(name="routing", output={"route": route})

            if route != "SQL_QUERY":
                if route == "SCHEMA_HELP":
                    msg = "Question orientée schéma détectée. Utilise une question métier pour générer du SQL."
                else:
                    msg = "Question hors périmètre SQL détectée. Reformule avec une demande liée à la base hospitalière."

                print(f"\n{msg}")

                if trace:
                    trace.event(name="error", output={"error": msg})
                    trace.score(name="sql_valid", value=0)
                    trace.score(name="execution_success", value=0)
                    trace.score(name="no_error", value=0)
                    trace.update(output={"status": "routed-non-sql", "route": route, "message": msg})
                    _langfuse.flush()
                continue

            # 1. Retrieval + reranking
            hybrid_results = hybrid.search(query)
            reranked = reranker.rerank(query, hybrid_results, top_k_final=8)
            if trace:
                trace.event(
                    name="retrieval",
                    output={"top_chunks": [c.get("content_id") for c in reranked]},
                )

            # 2. Chain-of-Thought: raisonnement avant génération SQL
            print("\n[CoT] Analyse de la question en cours...")
            cot_plan = planner.plan(query, inspector.summary())
            print(f"[CoT] Plan:\n{cot_plan}\n")
            if trace:
                trace.event(name="chain-of-thought", output={"plan": cot_plan})

            # 3. Query construction avec le plan injecté
            prompt = constructor.build_context(query, reranked, cot_plan=cot_plan)
            if trace:
                trace.event(name="prompt", output={"prompt": prompt})

            # 4. LLM generation + correction loop
            max_attempts = 3
            sql = ""
            is_valid = False
            message = ""

            for attempt in range(1, max_attempts + 1):
                raw_sql = generator.generate(prompt) if attempt == 1 else generator.regenerate_with_feedback(
                    base_prompt=prompt,
                    failed_sql=sql,
                    error_message=message,
                    schema_summary=inspector.summary(),
                )
                sql = validator.clean_sql(raw_sql)
                is_valid, message = validator.validate(sql)

                print(f"\nSQL généré (tentative {attempt}/{max_attempts}) :\n")
                print(sql)
                print(f"\nValidation : {message}")

                if trace:
                    trace.event(
                        name="sql_generation",
                        output={"attempt": attempt, "sql": sql, "valid": is_valid, "validation_message": message},
                    )

                if is_valid:
                    break

            if not is_valid:
                if trace:
                    trace.event(name="error", output={"error": message, "failed_sql": sql})
                    trace.score(name="sql_valid", value=0)
                    trace.score(name="execution_success", value=0)
                    trace.score(name="no_error", value=0)
                    trace.update(output={"status": "failed", "reason": message})
                    _langfuse.flush()
                print("\nEchec après correction automatique. Veuillez reformuler la question.")
                continue

            # 5. Execution
            try:
                columns, rows = executor.execute(sql)
                print_table(columns, rows)
                if trace:
                    trace.event(
                        name="execution",
                        output={"columns": columns, "rows": [list(r) for r in rows[:5]]},
                    )
                    trace.score(name="sql_valid", value=1)
                    trace.score(name="execution_success", value=1 if rows else 0)
                    trace.score(name="no_error", value=1)
                    trace.update(output={
                        "status": "success",
                        "sql": sql,
                        "rows_returned": len(rows),
                        "attempts": attempt,
                    })
                    _langfuse.flush()
            except Exception as e:
                print(f"\nErreur lors de l'exécution SQL : {e}")
                if trace:
                    trace.event(name="error", output={"error": str(e), "sql": sql})
                    trace.score(name="sql_valid", value=1)
                    trace.score(name="execution_success", value=0)
                    trace.score(name="no_error", value=0)
                    trace.update(output={"status": "sql-error", "error": str(e)})
                    _langfuse.flush()

            print("\n")
    finally:
        hybrid.close()

if __name__ == "__main__":
    main()