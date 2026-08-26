from __future__ import annotations
_PUNCTUATION_NORMALIZATION_MAP = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
}

def normalize_punctuation(text: str) -> str:

    for unicode_char, ascii_char in _PUNCTUATION_NORMALIZATION_MAP.items():
        text = text.replace(unicode_char, ascii_char)
    return text
