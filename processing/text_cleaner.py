import re


def clean_text(text):

    if text is None:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()