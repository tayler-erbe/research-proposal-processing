"""Offline, evidence-linked search demonstration; no production imports or network calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from processing.text_cleaner import clean_text

ROOT = Path(__file__).resolve().parent
STOPWORDS = set("the and for with from this that are was will our their into using a an of to in on is by at as be".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]


def validate(documents):
    if not isinstance(documents, list) or not documents:
        raise ValueError("Input must be a nonempty JSON list of documents.")
    seen = set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("Each document must be an object.")
        identifier = doc.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Each document needs a unique nonempty string id.")
        if not isinstance(doc.get("pages"), list) or not doc["pages"]:
            raise ValueError("Each document needs a nonempty pages list.")
        if not all(isinstance(page, str) for page in doc["pages"]):
            raise ValueError("Pages must contain text strings.")
        seen.add(identifier)
    return documents


def pdf_document(path):
    # Optional PDF support. No OCR: a scan without text remains empty.
    from pypdf import PdfReader
    path = Path(path)
    raw = path.read_bytes()
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported by this demonstration.")
    pages = [page.extract_text() or "" for page in reader.pages]
    return {"id": path.stem, "pages": pages, "sha256": hashlib.sha256(raw).hexdigest()}


def build_index(documents):
    """Retain complete source pages; derive deterministic frequency keywords."""
    validate(documents)
    indexed = []
    for doc in documents:
        page_records = []
        for number, text in enumerate(doc["pages"], 1):
            page_records.append({"page": number, "text": text,
                                 "cleaned_text": clean_text(text),
                                 "keywords": [w for w, _ in Counter(tokens(text)).most_common(10)]})
        indexed.append({"id": doc["id"], "pages": page_records,
                        "source_sha256": doc.get("sha256"),
                        "needs_ocr_review": any(not p.strip() for p in doc["pages"])})
    return indexed


def search(indexed, query):
    """Return lexical matches with literal source quotes, never generated answers."""
    terms = set(tokens(query))
    matches = []
    for doc in indexed:
        for page in doc["pages"]:
            score = len(terms.intersection(tokens(page["text"])))
            if not score:
                continue
            # Center a literal excerpt on the first matching token.
            match = next(m for m in re.finditer(r"[a-z]{3,}", page["text"], re.I)
                         if m.group().lower() in terms)
            start = max(0, match.start() - 60)
            quote = page["text"][start:match.end() + 180]
            matches.append({"document_id": doc["id"], "page": page["page"],
                            "score": score, "quote": quote})
    return sorted(matches, key=lambda r: (-r["score"], r["document_id"], r["page"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="Approved JSON documents; defaults to synthetic fixtures")
    inputs.add_argument("--pdf", type=Path, help="Approved local text PDF; requires requirements-demo.txt")
    parser.add_argument("--query", default="imaging reporting", help="Lexical page search")
    args = parser.parse_args()
    try:
        documents = ([pdf_document(args.pdf)] if args.pdf else
                     json.loads((args.input or ROOT / "examples/synthetic_documents.json").read_text(encoding="utf-8")))
        index = build_index(documents)
        print(json.dumps({"mode": "offline lexical demonstration; not production NLP or AI Q&A",
                          "documents": index, "query": args.query,
                          "matches": search(index, args.query)}, indent=2))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Demo could not run: {exc}\n")


if __name__ == "__main__":
    main()
