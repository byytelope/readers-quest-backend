# pyright: basic

import logging
import os
from dataclasses import dataclass

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MAIN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MAIN_MODEL_DIR = "models/wav2vec2"

logger = logging.getLogger("uvicorn")


@dataclass
class MLModels:
    processor: Wav2Vec2Processor
    model: Wav2Vec2ForCTC


def load_models() -> MLModels:
    if not os.path.exists(MAIN_MODEL_DIR):
        print("Downloading model...")
        processor = Wav2Vec2Processor.from_pretrained(MAIN_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MAIN_MODEL_NAME)
        assert isinstance(processor, Wav2Vec2Processor)

        os.makedirs(MAIN_MODEL_DIR, exist_ok=True)
        processor.save_pretrained(MAIN_MODEL_DIR)
        model.save_pretrained(MAIN_MODEL_DIR)
        print(f"Model saved to {MAIN_MODEL_DIR}")
    else:
        logger.info(f"Loading model from {MAIN_MODEL_DIR}...")
        processor = Wav2Vec2Processor.from_pretrained(MAIN_MODEL_DIR)
        model = Wav2Vec2ForCTC.from_pretrained(MAIN_MODEL_DIR)
        assert isinstance(processor, Wav2Vec2Processor)

    return MLModels(processor=processor, model=model)
