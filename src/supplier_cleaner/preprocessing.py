"""
preprocessing.py
----------------
Functions for normalising raw supplier name strings before
similarity-based grouping. Handles abbreviation expansion,
stop-word removal, and whitespace removal.
"""

import re

STATE_MAPPING = {
    "new south wales": "nsw",
    "queensland": "qld",
    "victoria": "vic",
    "south australia": "sa",
    "western australia": "wa",
    "tasmania": "tas",
    "northern territory": "nt",
    "australian capital territory": "act",
}

ORDINAL_MAPPING = {
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "sixth": "6th",
    "seventh": "7th",
    "eighth": "8th",
    "ninth": "9th",
    "tenth": "10th",
}

STOP_WORDS = {
    "and",
    "corporation",
    "enterprise",
    "incorporated",
    "us",
    "international",
    "llc",
    "pty",
    "ltd",
    "limited",
    "australia",
    "australasia",
}


def preprocess_supplier_name(name: str) -> str:
    """Normalise a raw supplier name string.

    Steps applied in order:
        1. Lowercase
        2. Replace '&' with 'and'
        3. Expand Australian state names to abbreviations
        4. Expand ordinal words to numerals
        5. Strip non-alphanumeric characters
        6. Remove stop words
        7. Collapse whitespace

    Args:
        name: Raw supplier name string.

    Returns:
        Normalised supplier name string.
    """
    name = name.lower()
    name = name.replace("&", "and")
    name = " ".join(STATE_MAPPING.get(w, w) for w in name.split())
    name = " ".join(ORDINAL_MAPPING.get(w, w) for w in name.split())
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = " ".join(w for w in name.split() if w not in STOP_WORDS)
    name = re.sub(r" +", " ", name).strip()
    return name
