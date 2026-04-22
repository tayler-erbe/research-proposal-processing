# Token-level validator used by the NLP pipeline's real-word filter.
#
# This module maintains a combined vocabulary set (NLTK words + WordNet
# lemmas) loaded once at import time from a pickle cache. Without the
# cache, building this set takes ~30 seconds per process — intolerable
# for pipeline runs. The cache is built by scripts/build_vocab_cache.py
# and lives in data/real_word_vocab.pkl (gitignored — regenerate locally).
#
# The validator blends the static vocab set with wordfreq's corpus-based
# frequency check. The frequency check catches modern terminology like
# "machine learning" or "metagenomics" that predate WordNet's lemma list.
#
# NOTE: An earlier version of this file used SciSpaCy's en_core_sci_sm
# model for biomedical-domain vocabulary. It was removed because the
# model's install path wasn't reliably available across environments,
# and the zipf_frequency fallback handles most of the same terms.

import re
import pickle
from pathlib import Path
from wordfreq import zipf_frequency


# ─── Load vocabulary once at import time ─────────────────────────────

print("[FILTER] Loading vocabularies...")

_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "real_word_vocab.pkl"

if _CACHE_PATH.exists():
    with open(_CACHE_PATH, "rb") as f:
        _REAL_WORD_SET = pickle.load(f)
    print(f"[FILTER] Loaded {len(_REAL_WORD_SET):,} words from cache.")
else:
    # Runtime build if the cache isn't present. Slow enough that the
    # print message serves as a reminder to run the cache-build script.
    print("[FILTER] Cache not found — building vocab at runtime (slow).")
    print("[FILTER] Run scripts/build_vocab_cache.py once to fix this.")
    from nltk.corpus import words, wordnet
    _ENGLISH_WORDS = set(words.words())
    _WORDNET_WORDS = set()
    for syn in wordnet.all_synsets():
        for lemma in syn.lemmas():
            _WORDNET_WORDS.add(lemma.name().lower())
    _REAL_WORD_SET = _ENGLISH_WORDS | _WORDNET_WORDS
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_REAL_WORD_SET, f)
    print(f"[FILTER] Built and cached {len(_REAL_WORD_SET):,} words.")


# ─── Token validation ────────────────────────────────────────────────

def is_valid_token(w):
    """Return True iff w looks like a legitimate English content token.

    The four fast-rejects at the top (digit, non-alpha, length, single
    unique char) short-circuit about 90% of incoming tokens — the
    wordfreq check is relatively expensive, so putting it last matters
    for throughput on long texts."""
    if not isinstance(w, str):
        return False
    w = w.lower().strip()

    if w.isdigit():
        return False
    if not re.match(r'^[a-zA-Z\-]+$', w):
        return False
    if len(w) < 3 or len(w) > 25:
        return False
    if len(set(w)) == 1:              # "aaaa", "bbb", etc.
        return False

    if w in _REAL_WORD_SET:
        return True

    # Zipf frequency > 3.0 is roughly "common enough to appear in
    # everyday text more than ~1 in 1000 words". Low enough bar to
    # admit domain terms, high enough to exclude OCR artifacts.
    if zipf_frequency(w, "en") > 3.0:
        return True

    return False


# Alias kept for backward compatibility with nlp_pipeline.py.
def is_real_word(w):
    return is_valid_token(w)


def clean_to_real_words(text):
    """Return (cleaned_text, token_count) for a string.
    Token count is the decider for the SUCCESSFULLY_PARSED flag upstream."""
    if not isinstance(text, str):
        return "", 0
    tokens = text.split()
    valid_tokens = [w.lower().strip() for w in tokens if is_valid_token(w)]
    return " ".join(valid_tokens), len(valid_tokens)
