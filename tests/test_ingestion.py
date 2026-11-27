from pathlib import Path
import unittest

from app.ingestion import (
    _context_prefix,
    _node_id,
    _normalize_line,
    _parse_pdf_pages,
    _parse_provision_heading,
)


class IngestionTest(unittest.TestCase):
    def test_normalize_line_removes_pdf_artifacts(self):
        line = "  Section\u00a0  22\u00adA.\t Theft  "
        self.assertEqual(
            _normalize_line(line),
            "Section 22A. Theft"
        )

    def test_parse_provision_heading_supports_letter_suffix(self):
        lines = ["22-A.  Punishment"]
        self.assertEqual(
            _parse_provision_heading(lines, 0),
            ("22-A", "Punishment")
        )

    def test_parse_split_letter_heading(self):
        lines = ["22", "A.", "Punishment"]
        self.assertEqual(
            _parse_provision_heading(lines, 0),
            ("22-A", "Punishment")
        )

    def test_constitution_uses_article_labels_and_metadata(self):
        pages = [
            "Preamble.\n1. The Republic and its territories.",
        ]
        documents = _parse_pdf_pages(
            Path("data/Constitution of Pakistan.pdf"),
            pages
        )

        preamble = next(
            document
            for document in documents
            if document.metadata["provision_number"] == "Preamble"
        )
        article = next(
            document
            for document in documents
            if document.metadata["provision_number"] == "1"
        )

        self.assertEqual(preamble.metadata["provision_type"], "preamble")
        self.assertEqual(article.metadata["provision_type"], "article")
        self.assertEqual(
            article.metadata["parent_provision_id"],
            article.metadata["provision_id"]
        )
        self.assertIn("Article 1", article.text)

    def test_context_prefix_includes_hierarchy(self):
        text = _context_prefix(
            "Pakistan Penal Code",
            "X",
            "Of Offences",
            "II",
            "General Exceptions",
            "section",
            "22-A",
            "Punishment",
        )
        self.assertEqual(
            text,
            "Pakistan Penal Code\n"
            "Part X: Of Offences\n"
            "Chapter II: General Exceptions\n"
            "Section 22-A: Punishment"
        )

    def test_chunk_id_is_stable(self):
        first = _node_id("same-provision", 3)
        second = _node_id("same-provision", 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
