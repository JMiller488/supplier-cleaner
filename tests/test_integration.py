"""
test_integration.py
-------------------
End-to-end integration test for the supplier cleaning pipeline.

Unlike the unit tests, this loads the real sentence-transformer model
and verifies that preprocessing + embedding + grouping work together
to correctly cluster supplier name variants.
"""

from supplier_cleaner.preprocessing import preprocess_supplier_name
from supplier_cleaner.grouping import group_suppliers


class TestEndToEndPipeline:
    """Integration tests for the full preprocessing → grouping pipeline."""

    def test_variants_grouped_correctly(self):
        """Obvious supplier name variants should be grouped together,
        while distinct suppliers should remain separate."""
        raw_names = [
            "Acme Pty Ltd",
            "ACME PTY LTD",
            "acme pty limited",
            "Globex Corporation",
            "GLOBEX CORP",
        ]

        preprocessed = [preprocess_supplier_name(name) for name in raw_names]
        groups = group_suppliers(preprocessed)

        # All Acme variants should map to the same canonical name
        acme_names = preprocessed[:3]
        acme_groups = {groups[name] for name in acme_names}
        assert len(acme_groups) == 1, f"Acme variants split into {acme_groups}"

        # All Globex variants should map to the same canonical name
        globex_names = preprocessed[3:]
        globex_groups = {groups[name] for name in globex_names}
        assert len(globex_groups) == 1, f"Globex variants split into {globex_groups}"

        # Acme and Globex should be in different groups
        assert acme_groups != globex_groups, "Acme and Globex incorrectly merged"
