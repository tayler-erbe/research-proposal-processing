# HERDS field classifier.
#
# Each proposal gets scored against all 26 HERDS taxonomy nodes via a
# blend of semantic embedding similarity and keyword overlap. The top
# scoring node becomes the HERDS_SUB label; the top-3 set is also
# surfaced for downstream aggregation.
#
# The scoring blend is deliberate:
#
#   final_score = 0.75 * cosine(doc_embedding, taxonomy_embedding)
#               + 0.25 * keyword_overlap(doc_text, taxonomy_keywords)
#
# Pure semantic similarity tended to misclassify proposals with heavy
# boilerplate toward whichever broad category the boilerplate leaned
# (Education comes up a lot because grant narratives include training
# plans). The keyword component re-anchors the score on terms the
# taxonomy considers authoritative for each field.
#
# Performance notes — the speed-relevant design choices:
#
#   1. Single shared SentenceTransformer model. KeyBERT reuses it.
#   2. Taxonomy embeddings are pre-computed once and cached to disk.
#   3. The 26 cosine comparisons per document are done as a single
#      matrix multiply rather than 26 separate calls.
#   4. Batch classification encodes all documents in one model.encode
#      call rather than one per document.

import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from data.herds_taxonomy import taxonomy

# Scores below this are treated as "no confident match" — we return
# nulls for the label but keep the numeric score so downstream code
# can see how close it came. 0.12 is empirically where scores stop
# corresponding to meaningful topical matches.
MIN_SCORE_THRESHOLD = 0.12

TAXONOMY_CACHE_PATH = Path("storage/cache/herds_taxonomy_embeddings.pkl")

# ─── Shared model singleton ──────────────────────────────────────────
# The model path is resolved relative to the package root so the
# sentence transformer can be shipped locally rather than pulled from
# HuggingFace on every fresh environment.

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data/models/all-MiniLM-L6-v2"

print("[HERDS] Loading embedding model...")
_model = SentenceTransformer(str(MODEL_PATH))
print("[HERDS] Model loaded.")


def _get_shared_model():
    """Exposed to keyword_service so KeyBERT can reuse this instance
    instead of loading a second copy."""
    return _model


# ─── Taxonomy embedding cache ────────────────────────────────────────

def _build_taxonomy_embeddings():
    """Encode each taxonomy node's keyword list and pickle the results.
    Runs once per fresh environment; subsequent imports load the cache."""
    nodes        = []
    descriptions = []
    for broad, broad_data in taxonomy.items():
        for sub, sub_data in broad_data["subcategories"].items():
            keywords    = sub_data["keywords"]
            description = " ".join(keywords)
            nodes.append({"broad": broad, "sub": sub, "keywords": keywords})
            descriptions.append(description)

    print(f"[HERDS] Building {len(nodes)} taxonomy embeddings (batch)...")
    embeddings = _model.encode(descriptions, batch_size=32, show_progress_bar=False)
    for i, emb in enumerate(embeddings):
        nodes[i]["embedding"] = emb

    TAXONOMY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAXONOMY_CACHE_PATH, "wb") as f:
        pickle.dump(nodes, f)
    print(f"[HERDS] Taxonomy cache saved → {TAXONOMY_CACHE_PATH}")
    return nodes


def _load_taxonomy_embeddings():
    if TAXONOMY_CACHE_PATH.exists():
        print(f"[HERDS] Loading taxonomy embeddings from cache...")
        with open(TAXONOMY_CACHE_PATH, "rb") as f:
            nodes = pickle.load(f)
        print(f"[HERDS] {len(nodes)} taxonomy nodes loaded from cache.")
        return nodes
    return _build_taxonomy_embeddings()


taxonomy_embeddings = _load_taxonomy_embeddings()
print(f"[HERDS] {len(taxonomy_embeddings)} taxonomy nodes ready.")

# Pre-normalize the taxonomy matrix once so per-document scoring is
# just one matrix multiply + one vector normalization.
_taxonomy_matrix = np.vstack([n["embedding"] for n in taxonomy_embeddings])
_taxonomy_norm   = _taxonomy_matrix / (
    np.linalg.norm(_taxonomy_matrix, axis=1, keepdims=True) + 1e-9
)


# ─── Scoring ─────────────────────────────────────────────────────────

def keyword_score(text, keywords):
    """Fraction of a node's keywords that appear verbatim in the text.
    Cheap but effective — catches cases where the embedding is ambiguous
    but the document explicitly uses the category's own vocabulary."""
    if not isinstance(text, str):
        return 0.0
    text = text.lower()
    return sum(1 for k in keywords if k in text) / max(len(keywords), 1)


def classify_text_all(text, top_n=3):
    """Encode once, score against every taxonomy node, return both the
    top-1 and top-N results. The single-encode + matrix multiply pattern
    is much faster than scoring each node independently."""
    if not isinstance(text, str) or not text.strip():
        return (None, None, None), []

    text_emb  = _model.encode(text)
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-9)

    # (26, dim) @ (dim,) = (26,) — all cosine scores in one op.
    cosine_scores = _taxonomy_norm @ text_norm

    results = []
    for i, node in enumerate(taxonomy_embeddings):
        e_score = float(cosine_scores[i])
        k_score = keyword_score(text, node["keywords"])
        score   = (0.75 * e_score) + (0.25 * k_score)
        results.append({"broad": node["broad"], "sub": node["sub"], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    if best["score"] < MIN_SCORE_THRESHOLD:
        print(f"[HERDS DROP] Score too low: {best['score']:.3f} | {best['sub']}")
        best_result = (None, None, best["score"])
    else:
        print(f"[HERDS MATCH] {best['sub']} | Score: {best['score']:.3f}")
        best_result = (best["broad"], best["sub"], best["score"])

    top_n_results = [
        (r["broad"], r["sub"], r["score"])
        for r in results[:top_n]
        if r["score"] >= MIN_SCORE_THRESHOLD
    ]

    return best_result, top_n_results


def assign_herds_batch(texts, top_n=3):
    """Batch classifier — the main speed win for bulk runs.

    Encodes the entire batch in one model.encode call, then loops
    through the resulting embedding matrix applying the same scoring
    logic as the single-document version. 5-10x faster than looping
    over classify_text_all() for real-sized batches."""
    if not texts:
        return []

    total = len(texts)
    print(f"[HERDS] Batch classifying {total} documents...")

    # Filter out empty inputs so the encoder doesn't waste time on them,
    # but track original indices so the output preserves caller's order.
    valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    valid_texts   = [texts[i] for i in valid_indices]

    if valid_texts:
        doc_embeddings = _model.encode(
            valid_texts, batch_size=32, show_progress_bar=False
        )
    else:
        doc_embeddings = []

    valid_results = {}
    for idx, (text, text_emb) in enumerate(zip(valid_texts, doc_embeddings)):
        text_norm     = text_emb / (np.linalg.norm(text_emb) + 1e-9)
        cosine_scores = _taxonomy_norm @ text_norm

        results = []
        for i, node in enumerate(taxonomy_embeddings):
            e_score = float(cosine_scores[i])
            k_score = keyword_score(text, node["keywords"])
            score   = (0.75 * e_score) + (0.25 * k_score)
            results.append({"broad": node["broad"], "sub": node["sub"], "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]

        if best["score"] < MIN_SCORE_THRESHOLD:
            best_result = (None, None, best["score"])
        else:
            best_result = (best["broad"], best["sub"], best["score"])

        top_n_results = [
            (r["broad"], r["sub"], r["score"])
            for r in results[:top_n]
            if r["score"] >= MIN_SCORE_THRESHOLD
        ]

        valid_results[valid_indices[idx]] = (best_result, top_n_results)

    # Rebuild output in original order, filling empties for skipped inputs.
    output = [
        valid_results.get(i, ((None, None, None), []))
        for i in range(total)
    ]

    print(f"[HERDS] Batch classification complete.")
    return output


# ─── Backward-compatible single-doc APIs ─────────────────────────────

def assign_herds_category(text):
    """Single-doc convenience wrapper. Use assign_herds_batch for bulk."""
    best, _ = classify_text_all(text, top_n=1)
    return best


def assign_herds_top3(text):
    """Single-doc top-3 as an underscore-joined CSV string, matching
    the format the analytics DB's HERDS_FIELD column stores."""
    _, top_n = classify_text_all(text, top_n=3)
    if not top_n:
        return None
    labels = [str(m[1]).strip().replace(" ", "_").lstrip("_") for m in top_n if m[1]]
    return ",".join(labels) if labels else None
