# Aggressive text cleaner used as the first pass before lemmatization.
#
# The order of operations here matters:
#   1. lowercase + newline strip — gets us to a uniform base
#   2. URL strip — PDFs from government and academic sites are URL-heavy
#   3. bracketed/punctuation strip — quotes and parens break tokenization
#   4. non-alpha strip — numbers and leftover symbols go
#   5. whitespace collapse
#   6. token-length + removal-term filter — drops < 4 and > 15 character
#      tokens, plus the domain-specific remove terms
#
# The 3-to-15 character window is deliberate: under 3 rules out most
# abbreviations and initials, over 15 catches concatenated garble from
# broken PDF extraction like "theproposalbudget".

import re


def clean_text(text, remove_terms):
    if not isinstance(text, str):
        return ""

    text = text.lower().replace('\n', ' ')

    # URLs first, before we strip non-alpha characters they depend on.
    text = re.sub(r'http\S+|www\S+', '', text)

    # Common punctuation that turns into token noise.
    text = re.sub(r'[-,:$#^&*(){};"\'"]', ' ', text)

    # Everything that isn't a letter or whitespace.
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Collapse runs of whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    # Length-window + remove-term filter.
    words = [
        w for w in text.split()
        if 3 < len(w) <= 15
        and w not in remove_terms
    ]

    return " ".join(words)
