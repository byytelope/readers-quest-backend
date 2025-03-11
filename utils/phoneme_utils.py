import string
from difflib import SequenceMatcher
from phonemizer import phonemize  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from utils.types import Feedback


def remove_punctuation(sentence: str) -> str:
    """
    Removes punctuation from the given sentence.
    """
    return sentence.translate(str.maketrans("", "", string.punctuation))


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


def calculate_grade(
    expected_sentence: str,
    recognized_sentence: str,
    expected_phonemes: str,
    recognized_phonemes: str,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> tuple[float, list[Feedback]]:
    """
    Compute a pronunciation score using word-level and phoneme-level matching.
    Generate feedback for individual words.
    """
    expected_words = remove_punctuation(expected_sentence.lower()).split()
    recognized_words = remove_punctuation(recognized_sentence.lower()).split()

    expected_phoneme_groups = expected_phonemes.split()
    recognized_phoneme_groups = recognized_phonemes.split()

    sm = SequenceMatcher(None, expected_words, recognized_words)
    phoneme_scores: list[float] = []
    feedback: list[Feedback] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for x in range(i2 - i1):
                e_idx = i1 + x
                r_idx = j1 + x
                feedback.append({
                    "type": "correct",
                    "word": recognized_words[r_idx],
                    "index": e_idx,
                })
                if e_idx < len(expected_phoneme_groups) and r_idx < len(
                    recognized_phoneme_groups
                ):
                    phoneme_score = SequenceMatcher(
                        None,
                        expected_phoneme_groups[e_idx],
                        recognized_phoneme_groups[r_idx],
                    ).ratio()
                    phoneme_scores.append(phoneme_score)
        elif tag == "replace":
            length = max(i2 - i1, j2 - j1)
            for x in range(length):
                e_idx = i1 + x
                r_idx = j1 + x
                if e_idx < i2 and r_idx < j2:
                    feedback.append({
                        "type": "mispronounced",
                        "word": recognized_words[r_idx],
                        "expected": expected_words[e_idx],
                        "index": e_idx,
                    })
                    if e_idx < len(expected_phoneme_groups) and r_idx < len(
                        recognized_phoneme_groups
                    ):
                        phoneme_score = SequenceMatcher(
                            None,
                            expected_phoneme_groups[e_idx],
                            recognized_phoneme_groups[r_idx],
                        ).ratio()
                        phoneme_scores.append(phoneme_score)
                elif e_idx < i2:
                    feedback.append({
                        "type": "missing",
                        "word": expected_words[e_idx],
                        "index": e_idx,
                    })
                elif r_idx < j2:
                    feedback.append({"type": "extra", "word": recognized_words[r_idx]})
        elif tag == "delete":
            for idx in range(i1, i2):
                feedback.append({
                    "type": "missing",
                    "word": expected_words[idx],
                    "index": idx,
                })
        elif tag == "insert":
            for idx in range(j1, j2):
                feedback.append({"type": "extra", "word": recognized_words[idx]})

    phoneme_score = sum(phoneme_scores) / len(phoneme_scores) if phoneme_scores else 0
    word_matches = sum(
        1
        for op, i1, i2, _, _ in sm.get_opcodes()
        if op == "equal"
        for _ in range(i2 - i1)
    )
    total_words = max(len(expected_words), len(recognized_words))
    wer_score = word_matches / total_words
    final_score = alpha * wer_score + beta * phoneme_score

    return round(final_score, 2), feedback
