from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = BASE_DIR / "context"
OUTPUT_FILE = BASE_DIR / "context_chunks.json"


@dataclass
class ContextChunk:
    content_id: str
    text: str
    source_file: str
    chunk_type: str
    entity: str
    subtype: str
    priority: str


class ContextLoader:
    def __init__(self, context_dir: Path) -> None:
        self.context_dir = context_dir

    def load_files(self) -> Dict[str, str]:
        files = [
            "schema.txt",
            "table_descriptions.txt",
            "vocabulary.txt",
            "examples.txt",
            "sql_rules.txt",
            "join_patterns.txt",
        ]

        loaded: Dict[str, str] = {}
        for filename in files:
            path = self.context_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Fichier introuvable : {path}")
            loaded[filename] = path.read_text(encoding="utf-8")
        return loaded


class ContextChunker:
    def chunk_all(self, docs: Dict[str, str]) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []

        chunks.extend(self._chunk_schema(docs["schema.txt"]))
        chunks.extend(self._chunk_descriptions(docs["table_descriptions.txt"]))
        chunks.extend(self._chunk_vocabulary(docs["vocabulary.txt"]))
        chunks.extend(self._chunk_examples(docs["examples.txt"]))
        chunks.extend(self._chunk_sql_rules(docs["sql_rules.txt"]))
        chunks.extend(self._chunk_join_patterns(docs["join_patterns.txt"]))

        return chunks

    def _chunk_schema(self, text: str) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []

        table_pattern = r"(TABLE\s+\w+.*?)(?=TABLE\s+\w+|RELATIONSHIPS|$)"
        table_blocks = re.findall(table_pattern, text, flags=re.DOTALL)

        for block in table_blocks:
            match = re.search(r"TABLE\s+(\w+)", block)
            if not match:
                continue

            entity = match.group(1).strip()

            chunks.append(
                ContextChunk(
                    content_id=f"schema_{entity}",
                    text=block.strip(),
                    source_file="schema.txt",
                    chunk_type="schema",
                    entity=entity,
                    subtype="table_definition",
                    priority="high",
                )
            )

        relationships_match = re.search(r"(RELATIONSHIPS.*)$", text, flags=re.DOTALL)
        if relationships_match:
            chunks.append(
                ContextChunk(
                    content_id="schema_relationships",
                    text=relationships_match.group(1).strip(),
                    source_file="schema.txt",
                    chunk_type="schema",
                    entity="relationships",
                    subtype="relationships",
                    priority="high",
                )
            )

        return chunks

    def _chunk_descriptions(self, text: str) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []
        blocks = re.findall(r"(TABLE\s+\w+.*?)(?=TABLE\s+\w+|$)", text, flags=re.DOTALL)

        for block in blocks:
            match = re.search(r"TABLE\s+(\w+)", block)
            if not match:
                continue

            entity = match.group(1).strip()

            chunks.append(
                ContextChunk(
                    content_id=f"desc_{entity}",
                    text=block.strip(),
                    source_file="table_descriptions.txt",
                    chunk_type="description",
                    entity=entity,
                    subtype="business_description",
                    priority="medium",
                )
            )

        return chunks

    def _chunk_vocabulary(self, text: str) -> List[ContextChunk]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        patient_doctor: List[str] = []
        departments_appointments: List[str] = []
        prescriptions: List[str] = []
        domain_terms: List[str] = []

        for line in lines:
            lower = line.lower()

            if any(
                word in lower
                for word in [
                    "patient",
                    "malade",
                    "personne",
                    "doctor",
                    "médecin",
                    "specialty",
                    "spécialité",
                ]
            ):
                patient_doctor.append(line)

            elif any(
                word in lower
                for word in [
                    "department",
                    "service",
                    "unité",
                    "floor",
                    "building",
                    "appointment",
                    "consultation",
                    "rendez-vous",
                    "reason",
                    "motif",
                    "diagnosis",
                    "diagnostic",
                    "status",
                    "date",
                ]
            ):
                departments_appointments.append(line)

            elif any(
                word in lower
                for word in [
                    "prescription",
                    "ordonnance",
                    "medication",
                    "médicament",
                    "medicine",
                    "dosage",
                    "duration",
                    "durée",
                ]
            ):
                prescriptions.append(line)

            else:
                domain_terms.append(line)

        chunks: List[ContextChunk] = []

        if patient_doctor:
            chunks.append(
                ContextChunk(
                    content_id="vocab_patients_doctors",
                    text="\n".join(patient_doctor),
                    source_file="vocabulary.txt",
                    chunk_type="vocabulary",
                    entity="patients_doctors",
                    subtype="semantic_mapping",
                    priority="medium",
                )
            )

        if departments_appointments:
            chunks.append(
                ContextChunk(
                    content_id="vocab_departments_appointments",
                    text="\n".join(departments_appointments),
                    source_file="vocabulary.txt",
                    chunk_type="vocabulary",
                    entity="departments_appointments",
                    subtype="semantic_mapping",
                    priority="medium",
                )
            )

        if prescriptions:
            chunks.append(
                ContextChunk(
                    content_id="vocab_prescriptions",
                    text="\n".join(prescriptions),
                    source_file="vocabulary.txt",
                    chunk_type="vocabulary",
                    entity="prescriptions",
                    subtype="semantic_mapping",
                    priority="medium",
                )
            )

        if domain_terms:
            chunks.append(
                ContextChunk(
                    content_id="vocab_domain_terms",
                    text="\n".join(domain_terms),
                    source_file="vocabulary.txt",
                    chunk_type="vocabulary",
                    entity="global",
                    subtype="domain_terms",
                    priority="medium",
                )
            )

        return chunks

    def _chunk_examples(self, text: str) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []
        blocks = re.findall(
            r"(QUESTION:.*?SQL:\s*.*?;)(?=\n\s*QUESTION:|$)", text, flags=re.DOTALL
        )

        for idx, block in enumerate(blocks, start=1):
            question_match = re.search(
                r"QUESTION:\s*(.+?)\nSQL:", block, flags=re.DOTALL
            )
            sql_match = re.search(r"SQL:\s*(.+)", block, flags=re.DOTALL)

            question = question_match.group(1).strip() if question_match else ""
            sql = sql_match.group(1).strip() if sql_match else ""

            entity = self._infer_entity_from_text(block)
            subtype = self._infer_example_subtype(sql)
            safe_id = self._make_safe_id(question) or f"example_{idx}"

            chunks.append(
                ContextChunk(
                    content_id=f"example_{safe_id}",
                    text=block.strip(),
                    source_file="examples.txt",
                    chunk_type="example",
                    entity=entity,
                    subtype=subtype,
                    priority="high",
                )
            )

        return chunks

    def _chunk_sql_rules(self, text: str) -> List[ContextChunk]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        title = lines[0]
        content_lines = lines[1:]

        general_lines: List[str] = [title]
        domain_lines: List[str] = []

        for line in content_lines:
            lower = line.lower()
            if any(
                word in lower
                for word in [
                    "department",
                    "medication",
                    "consultations",
                    "appointments table",
                    "outside the database scope",
                    "schema is insufficient",
                ]
            ):
                domain_lines.append(line)
            else:
                general_lines.append(line)

        chunks: List[ContextChunk] = [
            ContextChunk(
                content_id="rules_sql_general",
                text="\n".join(general_lines).strip(),
                source_file="sql_rules.txt",
                chunk_type="rule",
                entity="global",
                subtype="general",
                priority="high",
            )
        ]

        if domain_lines:
            chunks.append(
                ContextChunk(
                    content_id="rules_sql_domain",
                    text=(title + "\n" + "\n".join(domain_lines)).strip(),
                    source_file="sql_rules.txt",
                    chunk_type="rule",
                    entity="global",
                    subtype="domain",
                    priority="high",
                )
            )

        return chunks

    def _chunk_join_patterns(self, text: str) -> List[ContextChunk]:
        chunks: List[ContextChunk] = []
        blocks = re.findall(
            r"(PATTERN\s+\d+.*?)(?=PATTERN\s+\d+|$)", text, flags=re.DOTALL
        )

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue

            pattern_name = lines[1]
            safe_name = (
                pattern_name.lower()
                .replace(" -> ", "_")
                .replace("-", "_")
                .replace(" ", "_")
            )
            subtype = "multi_join" if pattern_name.count("->") >= 2 else "simple_join"

            chunks.append(
                ContextChunk(
                    content_id=f"join_{safe_name}",
                    text=block.strip(),
                    source_file="join_patterns.txt",
                    chunk_type="join_pattern",
                    entity=safe_name,
                    subtype=subtype,
                    priority="high",
                )
            )

        return chunks

    @staticmethod
    def _make_safe_id(text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")[:50]

    @staticmethod
    def _infer_entity_from_text(text: str) -> str:
        lower = text.lower()

        if "prescription" in lower or "medication" in lower or "paracetamol" in lower:
            return "prescriptions"
        if "appointment" in lower or "consultation" in lower or "rendez-vous" in lower:
            return "appointments"
        if "doctor" in lower or "médecin" in lower:
            return "doctors"
        if "department" in lower or "cardiology" in lower or "service" in lower:
            return "departments"
        if "patient" in lower:
            return "patients"

        return "global"

    @staticmethod
    def _infer_example_subtype(sql: str) -> str:
        sql_upper = sql.upper()

        if "GROUP BY" in sql_upper and "COUNT(" in sql_upper:
            return "group_by"
        if "JOIN" in sql_upper and "WHERE" in sql_upper:
            return "filter_join"
        if "JOIN" in sql_upper:
            return "join"
        if "COUNT(" in sql_upper:
            return "count"
        if "WHERE" in sql_upper:
            return "filter"

        return "simple"


def save_chunks_to_json(chunks: List[ContextChunk], output_path: Path) -> None:
    data = [asdict(chunk) for chunk in chunks]
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    loader = ContextLoader(CONTEXT_DIR)
    docs = loader.load_files()

    chunker = ContextChunker()
    chunks = chunker.chunk_all(docs)

    save_chunks_to_json(chunks, OUTPUT_FILE)

    print(f"{len(chunks)} chunks créés avec succès.\n")

    for chunk in chunks:
        print("=" * 80)
        print(f"content_id  : {chunk.content_id}")
        print(f"source_file : {chunk.source_file}")
        print(f"chunk_type  : {chunk.chunk_type}")
        print(f"entity      : {chunk.entity}")
        print(f"subtype     : {chunk.subtype}")
        print(f"priority    : {chunk.priority}")
        print("text preview:")
        print(chunk.text[:500])
        print()

    print(f"Fichier JSON généré : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
