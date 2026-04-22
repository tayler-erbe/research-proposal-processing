# spaCy-based named entity extraction.
#
# Not called by the default pipeline — the entity columns in the output
# table are populated as NULLs. This module is kept wired up so
# extraction can be toggled back on via an NLP-pipeline flag when the
# downstream use case justifies the per-document cost (~1-2s per doc,
# which is the dominant wall-clock cost for larger runs).
#
# Uses the small English model (en_core_web_sm) because the medium/large
# variants don't produce noticeably better entity spans on the kinds of
# formal prose typical of research proposals.

import spacy

print("[INIT] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp(text)

    # Return the surface text of every entity found. Dedupe and sort
    # so the output is stable across runs on the same input, which
    # matters for reproducibility when diff-ing pipeline outputs.
    entities = [ent.text for ent in doc.ents]
    entities = list(sorted(set(entities)))

    return entities
