from __future__ import annotations

import re
from typing import List


class QueryTranslator:
    def translate(self, query: str) -> List[str]:
        """
        Retourne la question originale + quelques reformulations utiles
        pour améliorer le retrieval.
        """
        query = query.strip()
        variants = [query]

        lower = query.lower()

        # Cas prescriptions / médicaments
        if any(word in lower for word in ["paracetamol", "medication", "medications", "prescription", "prescriptions", "drug", "drugs"]):
            variants.extend(self._prescription_variants(query))

        # Cas doctors / departments
        if any(word in lower for word in ["doctor", "doctors", "médecin", "médecins", "cardiology", "department", "departments", "service"]):
            variants.extend(self._doctor_department_variants(query))

        # Cas appointments
        if any(word in lower for word in ["appointment", "appointments", "consultation", "consultations", "rendez-vous"]):
            variants.extend(self._appointment_variants(query))

        # Cas schema help
        if any(word in lower for word in ["which table", "what table", "contains", "schema", "column", "columns"]):
            variants.extend(self._schema_variants(query))

        # Nettoyage des doublons en gardant l'ordre
        unique_variants = []
        seen = set()
        for variant in variants:
            normalized = variant.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_variants.append(variant.strip())

        return unique_variants

    def _prescription_variants(self, query: str) -> List[str]:
        variants = []

        med_match = re.search(r"\b(paracetamol|ibuprofen|aspirin|amoxicillin|insulin)\b", query, flags=re.IGNORECASE)
        medication = med_match.group(1) if med_match else "medication"

        variants.append(f"patients prescribed {medication}")
        variants.append(f"patients linked to prescriptions with medication_name {medication}")
        variants.append(f"patient names from prescriptions where medication is {medication}")

        return variants

    def _doctor_department_variants(self, query: str) -> List[str]:
        variants = []

        dept_match = re.search(r"\b(cardiology|neurology|pediatrics|emergency|dermatology)\b", query, flags=re.IGNORECASE)
        department = dept_match.group(1) if dept_match else "department"

        variants.append(f"doctors in {department} department")
        variants.append(f"doctors joined with departments filtered by department_name {department}")
        variants.append(f"doctors belonging to department {department}")

        return variants

    def _appointment_variants(self, query: str) -> List[str]:
        return [
            "appointments grouped by doctor",
            "count appointments per doctor",
            "appointments with patient and doctor joins",
        ]

    def _schema_variants(self, query: str) -> List[str]:
        return [
            "schema information for relevant table",
            "which table stores the requested data",
            "database table and column definition for this concept",
        ]


def print_variants(query: str, variants: List[str]) -> None:
    print("\n" + "=" * 100)
    print(f"QUESTION ORIGINALE : {query}")
    print("=" * 100)

    for idx, variant in enumerate(variants, start=1):
        print(f"{idx}. {variant}")


def main() -> None:
    translator = QueryTranslator()

    test_queries = [
        "Which patients received Paracetamol?",
        "Which doctors work in Cardiology?",
        "How many appointments did each doctor have?",
        "Show appointments with patient and doctor names.",
        "Which table contains prescriptions?",
    ]

    for query in test_queries:
        variants = translator.translate(query)
        print_variants(query, variants)


if __name__ == "__main__":
    main()