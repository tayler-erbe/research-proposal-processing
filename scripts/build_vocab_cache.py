#!/usr/bin/env python3
# One-time builder for data/real_word_vocab.pkl.
#
# Run this once per fresh environment. The pickle it produces is the
# vocabulary set used by the text filter to decide what counts as a
# real word. Without this cache, the pipeline builds the same set at
# runtime — works fine, but costs ~30 seconds on every run.
#
# Usage:
#   python scripts/build_vocab_cache.py
#
# NLTK data requirement: this script requires the 'words' and 'wordnet'
# corpora to be present. If they aren't, run:
#
#   python -c "import nltk; nltk.download('words'); nltk.download('wordnet')"

import pickle
from pathlib import Path

print("Building real word vocab cache...")

from nltk.corpus import words, wordnet

ENGLISH_WORDS = set(words.words())
print(f"  NLTK words:    {len(ENGLISH_WORDS):,}")

WORDNET_WORDS = set()
for syn in wordnet.all_synsets():
    for lemma in syn.lemmas():
        WORDNET_WORDS.add(lemma.name().lower())
print(f"  WordNet words: {len(WORDNET_WORDS):,}")

REAL_WORD_SET = ENGLISH_WORDS | WORDNET_WORDS
print(f"  Combined:      {len(REAL_WORD_SET):,}")

# Resolve output path relative to repo root rather than CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
output    = REPO_ROOT / "data" / "real_word_vocab.pkl"
output.parent.mkdir(exist_ok=True)
with open(output, "wb") as f:
    pickle.dump(REAL_WORD_SET, f)

print(f"\nSaved to {output}")
print("Restart your kernel, then run the pipeline.")
