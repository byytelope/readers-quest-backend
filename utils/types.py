from typing import TypedDict, Literal


class FeedbackSingle(TypedDict):
    type: Literal["missing", "extra", "correct"]
    word: str


class FeedbackMispronounced(TypedDict):
    type: Literal["mispronounced"]
    expected: str
    word: str


Feedback = FeedbackMispronounced | FeedbackSingle
