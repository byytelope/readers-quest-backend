# pyright: basic

import string
from difflib import SequenceMatcher

from phonemizer import phonemize


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
    alpha=0.5,
    beta=0.5,
) -> tuple[float, list[str]]:
    """
    Compute a pronunciation score using word-level and phoneme-level matching.
    Generate feedback for individual words.
    """
    expected_words = remove_punctuation(expected_sentence.lower()).split()
    recognized_words = remove_punctuation(recognized_sentence.lower()).split()

    expected_phoneme_groups = expected_phonemes.split()
    recognized_phoneme_groups = recognized_phonemes.split()

    total_words = max(len(expected_words), len(recognized_words))
    word_matches = sum(1 for e, r in zip(expected_words, recognized_words) if e == r)
    wer_score = word_matches / total_words

    phoneme_scores = []
    feedback = []

    for i, expected_word in enumerate(expected_words):
        if i < len(recognized_words):
            recognized_word = recognized_words[i]
            if expected_word != recognized_word:
                feedback.append(
                    f"Mispronounced: '{recognized_word}' (expected: '{expected_word}')"
                )
            else:
                feedback.append(f"Correct: '{recognized_word}'")

            if i < len(expected_phoneme_groups) and i < len(recognized_phoneme_groups):
                phoneme_score = SequenceMatcher(
                    None, expected_phoneme_groups[i], recognized_phoneme_groups[i]
                ).ratio()

                phoneme_scores.append(phoneme_score)
        else:
            feedback.append(f"Missing word: '{expected_word}'")
            phoneme_scores.append(0)

    for j in range(len(recognized_words) - len(expected_words)):
        feedback.append(f"Extra word: '{recognized_words[len(expected_words) + j]}'")

    phoneme_score = sum(phoneme_scores) / len(phoneme_scores) if phoneme_scores else 0
    final_score = alpha * wer_score + beta * phoneme_score

    return final_score, feedback
