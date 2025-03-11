from typing import TypedDict, Literal


class FeedbackExtra(TypedDict):
    type: Literal["extra"]
    word: str


class FeedbackMispronounced(TypedDict):
    type: Literal["mispronounced"]
    index: int
    word: str
    expected: str


class FeedbackIndexed(TypedDict):
    type: Literal["missing", "correct"]
    index: int
    word: str


Feedback = FeedbackIndexed | FeedbackExtra | FeedbackMispronounced
