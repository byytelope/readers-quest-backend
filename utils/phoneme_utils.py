# pyright: basic

from phonemizer import phonemize


def map_to_phonemes(text: str) -> str:
    """
    Converts text into phonemes using the phonemizer library.
    """
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        with_stress=True,
    )

    return str(phonemes)
