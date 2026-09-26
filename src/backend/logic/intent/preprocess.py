import re

_SPACES = re.compile(r"\s+")
_KEEP = re.compile(r"[^a-z0-9\s']")


def clean_text(text: str) -> str:
    """Lowercase, drop punctuation (keeps apostrophes/digits) and collapse whitespace."""
    text = text.lower().replace("’", "'")
    text = _KEEP.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
