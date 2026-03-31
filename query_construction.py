from __future__ import annotations

from typing import Any, Dict, List

MAX_SCHEMA = 2
MAX_JOINS = 2
MAX_EXAMPLES = 3
MAX_RULES = 1
MAX_DESCRIPTIONS = 1
MAX_VOCAB = 1


class QueryConstructor:
    def build_context(
        self, query: str, reranked_chunks: List[Dict[str, Any]], cot_plan: str = ""
    ) -> str:
        selected = self._select_chunks(reranked_chunks)

        rules = self._collect(selected, "rule")
        schemas = self._collect(selected, "schema")
        joins = self._collect(selected, "join_pattern")
        examples = self._collect(selected, "example")
        descriptions = self._collect(selected, "description")
        vocabulary = self._collect(selected, "vocabulary")

        prompt_parts: List[str] = []

        # System instructions
        prompt_parts.append(
            "You are an expert SQLite query generator with strict attention to schema accuracy.\n"
            "TASK: Generate ONLY a valid SQLite SELECT query. Output ONLY SQL code.\n"
            "NO explanations, NO markdown, NO comments, NO extra text.\n\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY tables and columns from the SCHEMA section below.\n"
            "2. NEVER create or invent table names or column names.\n"
            "3. NEVER use columns that don't exist (e.g., doctor_name, patient_name, age, appointment_time).\n"
            "4. Use the EXACT column names shown in schema (e.g., first_name, last_name - NOT name).\n"
            "5. Only join tables using foreign key relationships shown in schema.\n"
            "6. If uncertain or mapping is missing, return exactly: NO_SQL\n"
            "7. VERIFY every column name exists before using it.\n"
        )

        # Schema first - most important
        if schemas:
            prompt_parts.append(
                "SCHEMA (Only these tables and columns exist - NOTHING ELSE):\n"
                + "\n\n".join(schemas)
            )

        # Explicit column existence rules
        prompt_parts.append(
            "COLUMN NAMES - MEMORIZE THESE:\n\n"
            "doctors: doctor_id, first_name, last_name, specialty, department_id, email\n"
            "patients: patient_id, first_name, last_name, gender, birth_date, city, phone\n"
            "appointments: appointment_id, patient_id, doctor_id, department_id, appointment_date, reason, diagnosis, status\n"
            "prescriptions: prescription_id, appointment_id, patient_id, medication_name, dosage, duration_days\n"
            "departments: department_id, department_name, floor_number, building_name\n\n"
            "THESE DO NOT EXIST (common hallucinations):\n"
            "❌ doctor_name, patient_name, user_name (use first_name + last_name)\n"
            "❌ age (use birth_date)\n"
            "❌ appointment_time (use appointment_date)\n"
            "❌ phone_number (use phone)\n"
            "❌ consultation, medical_history, treatment (tables don't exist)\n"
            "❌ doctor_specialty, patient_city (use specialty, city directly)\n"
        )

        if rules:
            prompt_parts.append("SQL GENERATION RULES:\n" + "\n\n".join(rules))

        if descriptions:
            prompt_parts.append("TABLE DESCRIPTIONS:\n" + "\n\n".join(descriptions))

        if vocabulary:
            prompt_parts.append("SEMANTIC MAPPINGS:\n" + "\n\n".join(vocabulary))

        if joins:
            prompt_parts.append("VALID JOIN PATTERNS ONLY:\n" + "\n\n".join(joins))

        if examples:
            prompt_parts.append(
                "EXAMPLES TO FOLLOW (exact column usage):\n" + "\n\n".join(examples)
            )

        prompt_parts.append(
            "ALIAS & QUALIFICATION RULES (MANDATORY):\n"
            "1. ALWAYS use table aliases: FROM doctors d, patients p, appointments a, etc.\n"
            "2. ALWAYS qualify columns with aliases: d.first_name, p.birth_date, a.appointment_date\n"
            "3. NEVER mix table name with alias: use d.specialty ONLY, never doctors.specialty\n"
            "4. Standard aliases: p=patients, d=doctors, dep=departments, a=appointments, pr=prescriptions\n"
            "5. In SELECT clauses inside aliases in WHERE, use: (SELECT a.reason FROM appointments a WHERE ...)\n"
        )

        # Chain-of-Thought plan injected here if available
        if cot_plan:
            prompt_parts.append(
                "PRE-ANALYSIS (use this to guide your SQL - tables/columns already identified):\n"
                f"{cot_plan}\n\n"
                "⚠️ Use ONLY the tables and columns listed above. Ignore any not in the schema."
            )

        prompt_parts.append(
            f"QUESTION: {query}\n\n"
            "Generate only SELECT query using schema above. Verify all column names exist.\n"
            "DOUBLE-CHECK: All column names are qualified with aliases, no table names mixed in."
        )

        return "\n\n" + "\n\n---\n\n".join(prompt_parts) + "\n"

    def _select_chunks(
        self, reranked_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []

        counters = {
            "schema": 0,
            "join_pattern": 0,
            "example": 0,
            "rule": 0,
            "description": 0,
            "vocabulary": 0,
        }

        limits = {
            "schema": MAX_SCHEMA,
            "join_pattern": MAX_JOINS,
            "example": MAX_EXAMPLES,
            "rule": MAX_RULES,
            "description": MAX_DESCRIPTIONS,
            "vocabulary": MAX_VOCAB,
        }

        seen_ids = set()

        for chunk in reranked_chunks:
            chunk_type = chunk.get("chunk_type")
            content_id = chunk.get("content_id")

            if not chunk_type or not content_id:
                continue

            if content_id in seen_ids:
                continue

            if chunk_type not in limits:
                continue

            if counters[chunk_type] >= limits[chunk_type]:
                continue

            selected.append(chunk)
            counters[chunk_type] += 1
            seen_ids.add(content_id)

        return selected

    @staticmethod
    def _collect(chunks: List[Dict[str, Any]], chunk_type: str) -> List[str]:
        return [
            chunk["text"] for chunk in chunks if chunk.get("chunk_type") == chunk_type
        ]
