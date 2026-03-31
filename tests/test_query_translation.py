"""Tests for query translation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_translation import QueryTranslator


def test_translator_returns_original():
    """Test that original query is always included."""
    translator = QueryTranslator()
    variants = translator.translate("How many patients?")
    assert "How many patients?" in variants


def test_translator_prescription_variants():
    """Test prescription-related variants."""
    translator = QueryTranslator()
    variants = translator.translate("Which patients received Paracetamol?")
    assert len(variants) > 1
    assert any("paracetamol" in v.lower() for v in variants)


def test_translator_doctor_variants():
    """Test doctor-related variants."""
    translator = QueryTranslator()
    variants = translator.translate("Which doctors work in Cardiology?")
    assert len(variants) > 1
    assert any("cardiology" in v.lower() for v in variants)


def test_translator_no_duplicates():
    """Test that variants have no duplicates."""
    translator = QueryTranslator()
    variants = translator.translate("Show appointments with doctors")
    normalized = [v.strip().lower() for v in variants]
    assert len(normalized) == len(set(normalized))
