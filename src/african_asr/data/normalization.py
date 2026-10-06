"""Text normalization utilities for ASR evaluation."""

import re
import unicodedata


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    unicode_normalize: str | None = "NFC",
    custom_replacements: dict[str, str] | None = None,
) -> str:
    """Normalize an ASR transcription.

    Parameters
    ----------
    text:
        Raw transcription.
    lowercase:
        Convert text to lowercase.
    remove_punctuation:
        Remove punctuation while preserving letters, digits,
        whitespace, and apostrophes.
    unicode_normalize:
        Unicode normalization form such as NFC or NFKC.
        Set to None to disable Unicode normalization.
    custom_replacements:
        Optional mapping of strings to replace before normalization.

    Returns
    -------
    str
        Normalized transcription.
    """
    if custom_replacements:
        for old, new in custom_replacements.items():
            text = text.replace(old, new)

    if unicode_normalize:
        text = unicodedata.normalize(unicode_normalize, text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
