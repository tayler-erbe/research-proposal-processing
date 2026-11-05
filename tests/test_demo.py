import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from demo import build_index, pdf_document, search


class OfflineDemoTests(unittest.TestCase):
    def test_evidence_is_literal_and_page_correct(self):
        docs = [{"id": "a", "pages": ["unrelated geology", "Annual reporting is required."]},
                {"id": "b", "pages": ["Imaging study."]}]
        matches = search(build_index(docs), "reporting")
        self.assertEqual(len(matches), 1)
        result = matches[0]
        self.assertEqual((result["document_id"], result["page"]), ("a", 2))
        self.assertIn(result["quote"], docs[0]["pages"][1])

    def test_no_match_does_not_invent_evidence(self):
        self.assertEqual(search(build_index([{"id": "a", "pages": ["Geology."]}]), "quantum"), [])

    def test_empty_scan_requires_review(self):
        result = build_index([{"id": "scan", "pages": [""]}])[0]
        self.assertTrue(result["needs_ocr_review"])
        self.assertEqual(result["pages"][0]["keywords"], [])

    def test_invalid_documents_are_rejected(self):
        for documents in [[], [{"id": "a", "pages": []}],
                          [{"id": "a", "pages": [None]}],
                          [{"id": "a", "pages": ["x"]}, {"id": "a", "pages": ["y"]}]]:
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                build_index(documents)

    def test_pdf_text_extraction_and_blank_pages(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=400, height=400)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 30 300 Td (Synthetic imaging research.) Tj ET")
            page[NameObject("/Contents")] = stream
            writer.add_blank_page(width=400, height=400)
            with path.open("wb") as output:
                writer.write(output)
            doc = pdf_document(path)
            self.assertIn("imaging", doc["pages"][0])
            self.assertEqual(len(doc["sha256"]), 64)
            index = build_index([doc])
            self.assertTrue(index[0]["needs_ocr_review"])
            self.assertEqual(search(index, "imaging")[0]["page"], 1)

    def test_cli_works_outside_repository_without_database(self):
        script = Path(__file__).resolve().parents[1] / "demo.py"
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(script), "--query", "reporting"],
                                    cwd=folder, capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["matches"])
            self.assertEqual(payload["matches"][0]["document_id"], "SYNTHETIC-AWARD-001")


if __name__ == "__main__":
    unittest.main()
