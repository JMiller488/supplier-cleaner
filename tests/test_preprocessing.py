"""
test_preprocessing.py
---------------------
Unit tests for the supplier name preprocessing pipeline.
"""

import pytest
from supplier_cleaner.preprocessing import preprocess_supplier_name


class TestPreprocessSupplierName:
    """Tests for the preprocess_supplier_name function."""

    def test_lowercases_input(self):
        assert preprocess_supplier_name("ACME CORP") == "acme corp"

    def test_removes_stop_words(self):
        assert preprocess_supplier_name("Acme Pty Ltd") == "acme"

    def test_replaces_ampersand(self):
        assert preprocess_supplier_name("Smith & Jones") == "smith jones"

    def test_abbreviates_state_names(self):
        assert preprocess_supplier_name("Acme New South Wales") == "acme nsw"

    def test_converts_ordinals(self):
        assert preprocess_supplier_name("First National Bank") == "1st national bank"

    def test_removes_punctuation(self):
        assert preprocess_supplier_name("Acme, Corp.") == "acme corp"

    def test_collapses_whitespace(self):
        assert preprocess_supplier_name("Acme  Corp") == "acme corp"

    def test_empty_string(self):
        assert preprocess_supplier_name("") == ""

    def test_only_stop_words(self):
        assert preprocess_supplier_name("Pty Ltd") == ""