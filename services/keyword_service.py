# Three-way keyword extraction: TF-IDF, KeyBERT, LDA.
#
# The three methods deliberately pick up different signal:
#
#   TF-IDF   — frequency. "What terms are disproportionately common in
#              this document vs. the rest of the corpus?" Cheap, robust,
#              but prone to picking up proposal boilerplate if the
#              remove_terms list isn't tight.
#
#   KeyBERT  — semantic. "What phrases are embedding-close to the
#              document's centroid?" Catches multi-word concepts that
#              TF-IDF can't express even with ngram_range=(1,3). Slow
#              per-doc unless batched; the batch helper below matters.
#
#   LDA      — corpus-level topics. Produces the same keywords for
#              every row in a single run, which is a feature not a
#              bug — they're the "what is this batch of proposals
#              about, collectively" backdrop.
#
# KeyBERT and the HERDS classifier share a single SentenceTransformer
# instance. Loading it twice was costing ~500MB of RAM and ~30 seconds
# of startup — the _get_shared_model() hook in herds_classification_service
# is the shared-instance source of truth; we reuse it here.

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition           import LatentDirichletAllocation

from services.text_filter_service import is_valid_token


def clean_tokens(text):
    """Apply the same real-word filter the main pipeline uses.
    Critical that this matches — otherwise TF-IDF and the rest of the
    pipeline disagree on what counts as a valid keyword."""
    return " ".join([
        w.lower().strip()
        for w in text.split()
        if is_valid_token(w)
    ])


# ─── TF-IDF ──────────────────────────────────────────────────────────

def extract_tfidf_keywords(text, top_n=30):
    """Single-doc TF-IDF. Rarely what you want — prefer the batched
    fit_transform in nlp_pipeline.py for real corpus-wide scoring."""
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        text = clean_tokens(text)
        if not text:
            return []

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            max_features=1000
        )

        X        = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        scores   = X.toarray()[0]

        keywords = [
            features[i]
            for i in scores.argsort()[::-1][:top_n]
        ]

        # Drop numeric or symbol-heavy "keywords"; keep phrases
        # (contain a space) and pure-alphabetic single words.
        return list(set([k for k in keywords if k.isalpha() or " " in k]))

    except Exception as e:
        print(f"TF-IDF failed: {e}")
        return []


def get_tfidf_keywords(text):
    return extract_tfidf_keywords(text)


# ─── KeyBERT ─────────────────────────────────────────────────────────
# Module-level singleton pattern. The first call triggers model load;
# subsequent calls reuse the cached instance. We also try to reuse the
# model already loaded by the HERDS service to avoid a second copy of
# all-MiniLM-L6-v2 in memory.

_keybert_model = None


def _get_keybert_model():
    global _keybert_model
    if _keybert_model is None:
        try:
            from keybert import KeyBERT
            try:
                from services.herds_classification_service import _get_shared_model
                st_model = _get_shared_model()
                print("[KeyBERT] Reusing shared model from HERDS service.")
            except Exception:
                from sentence_transformers import SentenceTransformer
                print("[KeyBERT] Loading model (all-MiniLM-L6-v2)...")
                st_model = SentenceTransformer("all-MiniLM-L6-v2")
            _keybert_model = KeyBERT(model=st_model)
            print("[KeyBERT] Model ready.")
        except Exception as e:
            print(f"[KeyBERT] Failed to load model: {e}")
            _keybert_model = None
    return _keybert_model


def extract_keybert_keywords(text, top_n=25):
    """Single-doc KeyBERT. Use the batch version for anything larger
    than a handful of documents — the overhead of the model call is
    paid per-document and adds up fast."""
    if not isinstance(text, str) or not text.strip():
        return []

    kw_model = _get_keybert_model()
    if kw_model is None:
        return []

    try:
        # MMR with diversity=0.5 is a meaningful lever — lower values
        # produce more "similar phrases clustered together" output,
        # higher values broaden topical coverage at the cost of each
        # individual keyphrase being slightly less central. 0.5 is
        # empirically a reasonable middle ground for proposal text.
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
        )
        return [kw for kw, score in keywords if kw and len(kw) >= 3]

    except Exception as e:
        print(f"[KeyBERT] Extraction failed: {e}")
        return []


def extract_keybert_keywords_batch(texts, top_n=25, batch_size=32):
    """Batch extraction — the version the pipeline actually uses.

    The KeyBERT library itself doesn't expose a true batched API; the
    "batching" here is the loop + model reuse. Still gets a 5-10x
    speedup over the naive .apply() approach because we amortize the
    per-call Python overhead and keep the sentence transformer warm.

    Reduce batch_size to 16 on machines with tight RAM."""
    if not isinstance(texts, list) or len(texts) == 0:
        return []

    kw_model = _get_keybert_model()
    if kw_model is None:
        print("[KeyBERT] Model unavailable — returning empty keyword lists")
        return [[] for _ in texts]

    total   = len(texts)
    results = []

    print(f"[KeyBERT] Batch extracting from {total} documents (batch_size={batch_size})...")

    for start in range(0, total, batch_size):
        batch         = texts[start : start + batch_size]
        batch_results = []

        for i, text in enumerate(batch):
            if not isinstance(text, str) or not text.strip():
                batch_results.append([])
                continue
            try:
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    use_mmr=True,
                    diversity=0.5,
                    top_n=top_n,
                )
                batch_results.append(
                    [kw for kw, score in keywords if kw and len(kw) >= 3]
                )
            except Exception as e:
                # One bad doc shouldn't sink the batch.
                print(f"[KeyBERT] Doc {start + i} failed: {e}")
                batch_results.append([])

        results.extend(batch_results)
        print(f"[KeyBERT] Processed {min(start + batch_size, total)}/{total} docs")

    return results


def get_keybert_keywords(text):
    return extract_keybert_keywords(text)


# ─── LDA ─────────────────────────────────────────────────────────────

def extract_lda_keywords(texts, num_topics=5, words_per_topic=10):
    """Corpus-level LDA. Pass in the entire list of document strings.
    Returns one flat deduplicated list of keywords covering the whole
    corpus — this is the "backdrop" signal referenced in nlp_pipeline.py.

    Bailouts at the top handle the degenerate cases where LDA has
    nothing to work with: an empty corpus, a 1-document corpus, or a
    vocabulary that nothing in the corpus uses more than once."""
    if not isinstance(texts, list):
        return []

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

    if len(valid_texts) < 2:
        print("[LDA] Not enough documents — skipping")
        return []

    try:
        cleaned_texts = [clean_tokens(t) for t in valid_texts]
        cleaned_texts = [t for t in cleaned_texts if t.strip()]

        if len(cleaned_texts) < 2:
            print("[LDA] Not enough non-empty documents after cleaning — skipping")
            return []

        # min_df of 2 requires every vocab term to appear in at least
        # two documents. Relaxed to 1 for small corpora where that'd
        # leave us with an empty vocabulary.
        min_df = 2 if len(cleaned_texts) >= 10 else 1

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=1000,
            min_df=min_df,
        )

        X = vectorizer.fit_transform(cleaned_texts)

        if X.shape[1] == 0:
            print("[LDA] Empty vocabulary — skipping")
            return []

        # Don't ask for more topics than we have documents — LDA will
        # run but the output is meaningless.
        n_topics = min(num_topics, len(cleaned_texts))

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
        )

        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()

        keywords = []
        for topic in lda.components_:
            top_indices = topic.argsort()[-words_per_topic:]
            keywords.extend([feature_names[i] for i in top_indices])

        return list(set(keywords))

    except Exception as e:
        print(f"LDA failed: {e}")
        return []


def get_lda_keywords(text):
    """Single-doc LDA wrapper — reuses the batch API with a list of one.
    Exists for legacy import compatibility; nothing in the current
    pipeline calls it."""
    if not isinstance(text, str) or not text.strip():
        return []
    return extract_lda_keywords([text])
