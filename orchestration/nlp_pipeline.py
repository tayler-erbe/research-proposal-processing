# NLP orchestration — the expensive stage.
#
# Reads the merged proposal parquet produced by pipeline.py and runs a
# three-way keyword extraction ensemble (TF-IDF, KeyBERT, LDA) plus a
# semantic-embedding HERDS field classifier on top. Outputs a second
# parquet shaped for the analytics DB plus three staging tables.
#
# Sequence matters: each step assumes the columns added by the prior
# step exist and are clean. The 11-step numbering in the console output
# is meant to make it easy to tell exactly where a run died if something
# throws.

import re
import json
import pickle
import pandas as pd
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from services.text_cleaning_service import clean_text
from services.text_filter_service   import is_valid_token
from services.keyword_service       import get_lda_keywords

lemmatizer   = WordNetLemmatizer()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = PROJECT_ROOT / "storage" / "intermediate_tables"

# 6000 chars is roughly a scientific-abstract's worth of prose; beyond
# that, keyword extraction returns diminishing signal and starts
# picking up boilerplate like references and acknowledgments.
MAX_CHARS          = 6000

# 40 real words is the practical minimum below which keyword extraction
# produces noise (or empty results) rather than signal. Proposals with
# fewer than 40 clean tokens are flagged SUCCESSFULLY_PARSED=N and skip
# the NLP steps, but still flow through to the output table with nulls.
MIN_WORD_THRESHOLD = 40


# ─── Loaders ─────────────────────────────────────────────────────────

def load_remove_terms():
    """Corpus-specific stopwords: budget line items, PI names, city
    names. Built out iteratively from early pipeline runs; see
    data/remove_budget_terms_names_cities.parquet."""
    df = pd.read_parquet(DATA_DIR / "remove_budget_terms_names_cities.parquet")
    return df["Remove_Terms"].dropna().str.lower().tolist()


def load_real_word_vocab():
    """Return the combined NLTK-words + WordNet vocabulary set.

    Cached as a pickle to avoid the ~30s rebuild on every pipeline run.
    If the cache doesn't exist yet, build it once at runtime and save.
    The one-time build is fine; it's the per-run rebuild that hurts."""
    cache_path = DATA_DIR / "real_word_vocab.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[VOCAB] Cache not found — building at runtime (run build_vocab_cache.py once to fix)")
    from nltk.corpus import words, wordnet
    ENGLISH_WORDS = set(words.words())
    WORDNET_WORDS = set()
    for syn in wordnet.all_synsets():
        for lemma in syn.lemmas():
            WORDNET_WORDS.add(lemma.name().lower())
    vocab = ENGLISH_WORDS | WORDNET_WORDS
    with open(cache_path, "wb") as f:
        pickle.dump(vocab, f)
    return vocab


def reduce_keywords(keywords, max_len=1000):
    """Deduplicate, lowercase, and truncate a comma-separated keyword
    string to fit a VARCHAR2 column. Truncation is done at a comma
    boundary rather than mid-word so the stored string stays parseable
    downstream."""
    if not keywords:
        return None
    tokens = [k.strip() for k in keywords.split(",")] if isinstance(keywords, str) else keywords
    tokens = list(set([t.lower().strip() for t in tokens if t]))
    result = ",".join(tokens)
    if len(result) > max_len:
        cut = result[:max_len]
        last_comma = cut.rfind(",")
        result = cut[:last_comma] if last_comma > 0 else cut
    return result


def remove_name_like_tokens(text):
    """Drop tokens that look like proper nouns. Cheap heuristic — catches
    most Title-Case names and ALL-CAPS acronyms. Not perfect; proper-noun
    removal is a known weak point of any English NLP pipeline, and this
    version trades recall for speed."""
    if not isinstance(text, str):
        return text
    return " ".join(
        w for w in text.split()
        if not w.istitle() and not (w.isupper() and len(w) > 2)
    )


# ─── Main pipeline ───────────────────────────────────────────────────

def run_nlp_pipeline():
    print("\n===================================")
    print(" NLP PIPELINE STARTING")
    print("===================================\n")

    # STEP 1 — Load the merged table produced by pipeline.py.
    parquet_path = INTERMEDIATE_DIR / "proposal_full_table.parquet"
    if not parquet_path.exists():
        # This is the "no new proposals this run" case, not an error.
        print("[NLP] No proposal_full_table.parquet found — no new proposals to process, skipping.")
        print("\n===================================")
        print(" NLP PIPELINE SKIPPED (nothing to do)")
        print("===================================\n")
        return None

    df = pd.read_parquet(parquet_path)
    print(f"[1/11] Loaded {len(df)} documents")
    if "FILE_DATA_ID" not in df.columns:
        raise ValueError("FILE_DATA_ID missing — cannot proceed")
    df["LENGTH"] = df["CONTENT"].fillna("").apply(len)

    # STEP 2 — Load external resources once, up front.
    from nltk.corpus import stopwords
    stop_words    = set(stopwords.words("english"))
    remove_terms  = set(load_remove_terms()) | stop_words
    REAL_WORD_SET = load_real_word_vocab()
    print(f"[2/11] Resources loaded ({len(remove_terms):,} remove terms, {len(REAL_WORD_SET):,} vocab words)")

    # STEP 3 — Coarse text cleaning: lowercase, strip URLs, strip non-alpha,
    # drop short tokens, drop remove-terms. Output is lowercased prose.
    df["CLEAN_CONTENT"] = df["CONTENT"].fillna("").apply(lambda x: clean_text(x, remove_terms))
    print("[3/11] Text cleaning complete")

    # STEP 4 — Lemmatize + strip name-like tokens. Single-pass lemmatizer
    # is used deliberately; POS-tagged lemmatization more than doubled
    # runtime and produced negligibly different downstream keywords.
    def lemmatize_text(text):
        if not isinstance(text, str):
            return ""
        return " ".join([lemmatizer.lemmatize(t) for t in text.split()])

    df["LEMMATIZED_CONTENT"] = df["CLEAN_CONTENT"].apply(lemmatize_text)
    df["LEMMATIZED_CONTENT"] = df["LEMMATIZED_CONTENT"].apply(remove_name_like_tokens)
    print("[4/11] Lemmatization + name filtering complete")

    # STEP 4D — Hard-filter tokens to "real words" via the vocab set.
    # This is the expensive filter — it's what guarantees downstream
    # keyword extraction doesn't pick up OCR garble like "oraion"
    # or "commitee" as valid keywords.
    def is_real_word(w):
        if not isinstance(w, str):
            return False
        w = w.lower().strip()
        if w.isdigit() or not re.match(r'^[a-zA-Z\-]+$', w):
            return False
        if len(w) < 3 or len(w) > 30 or len(set(w)) == 1:
            return False
        return w in REAL_WORD_SET

    def clean_to_real_words(text):
        if not isinstance(text, str):
            return "", 0
        valid = [w.lower().strip() for w in text.split() if w not in remove_terms and is_real_word(w)]
        return " ".join(valid), len(valid)

    df["CLEAN_REAL_TEXT"], df["REAL_WORD_COUNT"] = zip(*df["LEMMATIZED_CONTENT"].apply(clean_to_real_words))

    # STEP 4E — Partition into parseable vs. unparseable. Unparseable
    # rows are kept (set aside in df_skipped) so they still land in the
    # output table, just with nulls in the keyword columns.
    df["SUCCESSFULLY_PARSED"] = df["REAL_WORD_COUNT"].apply(lambda x: "Y" if x >= MIN_WORD_THRESHOLD else "N")
    df_valid   = df[df["SUCCESSFULLY_PARSED"] == "Y"].copy()
    df_skipped = df[df["SUCCESSFULLY_PARSED"] == "N"].copy()
    print(f"[4E/11] Word filter: {len(df_valid)} valid, {len(df_skipped)} skipped")
    df = df_valid

    df["TEXT_FOR_KEYWORDS"] = df["CLEAN_REAL_TEXT"].str[:MAX_CHARS]

    # STEP 5A — TF-IDF over the whole corpus. Frequency-based signal:
    # what terms are disproportionately common in this doc vs. others?
    texts_for_tfidf = df["CLEAN_REAL_TEXT"].fillna("").tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=1000)
    try:
        X = vectorizer.fit_transform(texts_for_tfidf)
        feature_names = vectorizer.get_feature_names_out()

        def get_top_tfidf(row_idx, top_n=30):
            row_arr = X[row_idx].toarray()[0]
            top_idx = row_arr.argsort()[::-1][:top_n]
            return [feature_names[i] for i in top_idx if feature_names[i].isalpha() or " " in feature_names[i]]

        df["TFIDF_KEYWORDS"] = [get_top_tfidf(i) for i in range(X.shape[0])]
    except Exception as e:
        # TF-IDF fails on tiny corpora (<2 docs) or pathological vocabs.
        # Leave the column empty rather than crashing the whole run.
        print(f"[TF-IDF ERROR] {e}")
        df["TFIDF_KEYWORDS"] = [[] for _ in range(len(df))]
    print("[5A/11] TF-IDF complete")

    # STEP 5B — KeyBERT. Semantic signal: what phrases are embedding-
    # similar to the document centroid? Completely different information
    # from TF-IDF, which is why we run both.
    from services.keyword_service import extract_keybert_keywords_batch
    _keybert_texts   = df["TEXT_FOR_KEYWORDS"].fillna("").tolist()
    _keybert_results = extract_keybert_keywords_batch(_keybert_texts, top_n=25, batch_size=32)
    df["KEYBERT_KEYWORDS"] = _keybert_results
    print("[5B/11] KeyBERT complete")

    # STEP 5C — LDA. Corpus-level topic signal, intentionally identical
    # for every row in a given run. Provides a stable "what is this
    # corpus about" backdrop that the per-doc TF-IDF/KeyBERT keywords
    # are interpreted against.
    try:
        from services.keyword_service import extract_lda_keywords
        lda_keywords = extract_lda_keywords(df["CLEAN_REAL_TEXT"].fillna("").tolist())
        df["LDA_KEYWORDS"] = [lda_keywords] * len(df)
    except Exception as e:
        print(f"[LDA ERROR] {e}")
        df["LDA_KEYWORDS"] = [[] for _ in range(len(df))]
    print("[5C/11] LDA complete")

    # STEP 5D — Clean keyword lists. Drops duplicates, drops trivially
    # weak single-word keywords ("data", "study", "result"), drops
    # same-word-repeated phrases ("data data data"), drops all-weak
    # multi-word phrases.
    WEAK_WORDS = {"high","low","used","use","work","year","number","group","important","general","study","result","data","analysis"}

    def clean_keywords(keyword_list):
        if not isinstance(keyword_list, list):
            return []
        cleaned, seen = [], set()
        for t in keyword_list:
            if not isinstance(t, str):
                continue
            t = t.lower().strip()
            if not t or t in seen:
                continue
            words_in_t = t.split()
            if len(words_in_t) > 1 and len(set(words_in_t)) == 1:
                continue
            if len(words_in_t) > 1 and sum(1 for w in words_in_t if w in WEAK_WORDS) == len(words_in_t):
                continue
            if len(words_in_t) == 1 and (t in WEAK_WORDS or len(t) <= 3):
                continue
            if len(t) < 3:
                continue
            cleaned.append(t)
            seen.add(t)
        return cleaned

    df["TFIDF_KEYWORDS"]   = df["TFIDF_KEYWORDS"].apply(clean_keywords)
    df["KEYBERT_KEYWORDS"] = df["KEYBERT_KEYWORDS"].apply(clean_keywords)
    df["LDA_KEYWORDS"]     = df["LDA_KEYWORDS"].apply(clean_keywords)

    # STEP 5E — Merge all three keyword sources, preserving order of
    # first appearance (TF-IDF → KeyBERT → LDA) as a soft ranking.
    def merge_keywords(row):
        combined, seen = [], set()
        for col in ["TFIDF_KEYWORDS", "KEYBERT_KEYWORDS", "LDA_KEYWORDS"]:
            for k in (row.get(col) or []):
                if k not in seen:
                    seen.add(k)
                    combined.append(k)
        return combined
    df["ALL_KEYWORDS"] = df.apply(merge_keywords, axis=1)

    # STEP 5F — Flatten list columns to CSV strings for DB storage.
    def safe_join_keywords(x):
        return ",".join(x) if isinstance(x, list) and x else None
    df["SKLEARN_TOP_KEYWORDS_STR"] = df["TFIDF_KEYWORDS"].apply(safe_join_keywords)
    df["KEYBERT_KEYWORDS_STR"]     = df["KEYBERT_KEYWORDS"].apply(safe_join_keywords)
    df["LDA_KEYWORDS_STR"]         = df["LDA_KEYWORDS"].apply(safe_join_keywords)
    df["ALL_KEYWORDS_STR"]         = df["ALL_KEYWORDS"].apply(safe_join_keywords)

    # STEP 5G — Build the HERDS classifier input. HERDS sees:
    #   keywords   (signal: what this doc is specifically about)
    #   first 2500 chars of cleaned text   (signal: general topic)
    #   first 5 sentences                  (signal: the abstract opener)
    # Concatenation is intentional — the classifier treats it as a
    # single bag of words, and the repetition of the keyword terms
    # in both sections amplifies their weight.
    def build_herds_input(row):
        text, keywords = row["TEXT_FOR_KEYWORDS"], row["TFIDF_KEYWORDS"]
        parts = []
        if isinstance(keywords, list) and keywords:
            parts.append(" ".join(keywords))
        if isinstance(text, str) and text.strip():
            parts.append(text[:2500])
            parts.append(" ".join(text.split(".")[:5]))
        return " ".join(parts) if parts else None
    df["HERDS_INPUT"] = df.apply(build_herds_input, axis=1)
    print("[5G/11] Keywords built")

    # STEP 6 — HERDS classification. Matrix-multiplied cosine similarity
    # against 26 taxonomy nodes, blended 0.75/0.25 with keyword overlap.
    # See services/herds_classification_service.py for the math.
    from services.herds_classification_service import assign_herds_batch
    batch_results = assign_herds_batch(df["HERDS_INPUT"].tolist(), top_n=3)
    df["HERDS_BROAD"]          = [r[0][0] for r in batch_results]
    df["HERDS_SUB"]            = [r[0][1] for r in batch_results]
    df["HERDS_SCORE"]          = [r[0][2] for r in batch_results]
    df["HERDS_FIELDS_SKLEARN"] = df["HERDS_SUB"]

    def _format_top3(top_n_list):
        if not top_n_list:
            return None
        labels = [str(m[1]).strip().replace(" ", "_").lstrip("_") for m in top_n_list if m[1]]
        return ", ".join(labels) if labels else None

    df["HERDS_TOP3"] = [_format_top3(r[1]) for r in batch_results]
    print(f"[6/11] HERDS classification complete")
    print(df["HERDS_SUB"].value_counts(dropna=False).head(10).to_string())

    # STEP 7 — Final DB-safe string columns + placeholder entity columns.
    # Entity extraction is stubbed here — the spaCy entity extractor
    # exists (services/entity_extraction_service.py) but is gated behind
    # a perf flag and not run by default in the current pipeline.
    def safe_join(x):
        return ",".join(sorted(set([str(i).strip() for i in x if i]))) or None if isinstance(x, list) else x
    for col in ["TFIDF_KEYWORDS", "KEYBERT_KEYWORDS", "LDA_KEYWORDS", "ALL_KEYWORDS"]:
        if col in df.columns:
            df[f"{col}_STR"] = df[col].apply(safe_join)
    for col in ["ENTITIES","ORGANIZATIONS","PERSONS","LOCATIONS","TIME","FAC","EVENT","MONEY","PRODUCT"]:
        df[col] = None
    print("[7/11] Keyword strings created")

    # STEP 8 — VARCHAR2-safe reduced versions of each keyword column.
    # The DB has columns at varying widths (1000, 500) for storing
    # truncated versions — we produce both full and reduced at this
    # stage so the writers downstream don't have to re-truncate.
    df["SKLEARN_TOP_KEYWORDS_REDUCED"] = df["TFIDF_KEYWORDS_STR"].apply(lambda x: reduce_keywords(x, 1000))
    df["TOPIC_KEYWORDS"]               = df["LDA_KEYWORDS_STR"]
    df["TOPIC_KEYWORDS_REDUCED"]       = df["LDA_KEYWORDS_STR"].apply(lambda x: reduce_keywords(x, 500))
    df["KEYBERT_OUT"]                  = df["KEYBERT_KEYWORDS_STR"]
    df["KEYBERT_OUT_REDUCED"]          = df["KEYBERT_KEYWORDS_STR"].apply(lambda x: reduce_keywords(x, 1000))

    def extract_unique_keybert(kb_str):
        """Break KeyBERT phrases back into individual tokens, dedupe.
        Useful for downstream search indexing that wants unigrams rather
        than the original multi-word keyphrases."""
        if not kb_str:
            return None
        words = set()
        for phrase in kb_str.split(","):
            for w in phrase.strip().split():
                words.add(w.lower())
        return ",".join(words)

    df["UNIQUE_KEYBERT"]         = df["KEYBERT_KEYWORDS_STR"].apply(extract_unique_keybert)
    df["UNIQUE_KEYBERT_REDUCED"] = df["UNIQUE_KEYBERT"].apply(lambda x: reduce_keywords(x, 1000))
    print("[8/11] Reduced columns created")

    # STEP 9 — Rename columns to match the analytics DB schema.
    # Most notably CLEAN_CONTENT → CLEANED_TEXT because the existing
    # DB column has always been spelled that way, and TFIDF_KEYWORDS_STR
    # → SKLEARN_TOP_KEYWORDS because "sklearn" was the original
    # classifier name carried through the schema.
    df = df.rename(columns={
        "TFIDF_KEYWORDS_STR": "SKLEARN_TOP_KEYWORDS",
        "CLEAN_CONTENT":      "CLEANED_TEXT",
        "CLEAN_REAL_TEXT":    "CLEANED_TEXT_REDUCED",
    })
    df = df.drop(columns=["KEYBERT_KEYWORDS","LDA_KEYWORDS","TFIDF_KEYWORDS",
                          "ALL_KEYWORDS","LEMMATIZED_CONTENT","ENTITIES_TAGGED",
                          "TEXT_FOR_KEYWORDS"], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]
    cols = ["FILE_DATA_ID"] + [c for c in df.columns if c != "FILE_DATA_ID"]
    df = df[cols]
    print(f"[9/11] Columns cleaned ({len(df.columns)} total)")

    # STEP 10 — Assemble the NLP output DataFrame.
    for col in ["ORGANIZATIONS","PERSONS","LOCATIONS","TIME","FAC","EVENT","MONEY","PRODUCT"]:
        if col not in df.columns:
            df[col] = None

    OUTPUT_COLS = [
        "ID","PROPOSAL_NUMBER","MODULE_NUMBER",
        "CLEANED_TEXT","CLEANED_TEXT_REDUCED",
        "TOPIC_KEYWORDS","TOPIC_KEYWORDS_REDUCED",
        "KEYBERT_OUT","KEYBERT_OUT_REDUCED",
        "UNIQUE_KEYBERT","UNIQUE_KEYBERT_REDUCED",
        "SKLEARN_TOP_KEYWORDS","SKLEARN_TOP_KEYWORDS_REDUCED",
        "HERDS_FIELDS_SKLEARN","HERDS_SUB","HERDS_SCORE","HERDS_TOP3",
        "ENTITIES","ORGANIZATIONS","PERSONS","LOCATIONS",
        "TIME","FAC","EVENT","MONEY","PRODUCT",
        "UPDATE_TIMESTAMP","SUCCESSFULLY_PARSED",
        "LENGTH","DOCUMENT_TYPE","ALPHA_RATIO"
    ]

    nlp_output_df = df[OUTPUT_COLS].copy()
    for col in nlp_output_df.select_dtypes(include=["object"]).columns:
        nlp_output_df[col] = nlp_output_df[col].astype(object)

    # Append the skipped (SUCCESSFULLY_PARSED=N) rows so every
    # FILE_DATA_ID from this run has a row in the output table,
    # even if most columns are null.
    df_skipped = df_skipped.copy()
    for col in OUTPUT_COLS:
        if col not in df_skipped.columns:
            df_skipped[col] = None
    df_skipped = df_skipped.reindex(columns=OUTPUT_COLS)
    for col in df_skipped.select_dtypes(include=["object"]).columns:
        df_skipped[col] = df_skipped[col].astype(object)

    nlp_output_df = pd.concat([nlp_output_df, df_skipped], ignore_index=True)
    print(f"[10/11] Output prepared: {len(nlp_output_df)} rows")

    # STEP 10B — Force everything into parquet-writable types.
    # pd.to_parquet chokes on mixed types within a column (list here,
    # string there), so normalize up front.
    def normalize_for_parquet(x):
        try:
            if isinstance(x, (list, tuple, dict)):
                return json.dumps(x)
            elif pd.isna(x):
                return None
            elif isinstance(x, (int, float, bool)):
                return x
            else:
                return str(x)
        except Exception:
            return None

    cols_to_normalize = nlp_output_df.select_dtypes(include=["object","string"]).columns
    for col in cols_to_normalize:
        nlp_output_df[col] = nlp_output_df[col].apply(normalize_for_parquet)

    output_path = INTERMEDIATE_DIR / "proposal_full_table_nlp.parquet"
    nlp_output_df.to_parquet(output_path, index=False)
    print(f"[10/11] NLP output saved → {output_path.name}")

    # STEP 11 — Build the three staging tables in the exact shapes
    # the analytics DB expects (see building_staging_tables_service).
    from services.building_staging_tables_service import build_staging_tables
    build_staging_tables(INTERMEDIATE_DIR)
    print("[11/11] Staging tables built")

    print("\n===================================")
    print(" NLP PIPELINE COMPLETE")
    print("===================================\n")

    return nlp_output_df
