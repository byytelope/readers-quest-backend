# pyright: basic

import logging
import os
from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MAIN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MAIN_MODEL_DIR = "models/wav2vec2"
EMOTION_MODEL_DIR = "./svc_model.joblib"
EMOTION_SCALER_DIR = "./scaler.joblib"
EMOTION_ENCODER_DIR = "./encoder.joblib"

logger = logging.getLogger("uvicorn")


@dataclass
class MLModels:
    processor: Wav2Vec2Processor
    main_model: Wav2Vec2ForCTC
    emotion_model: Any
    emotion_scaler: StandardScaler
    emotion_encoder: Any


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
        main_model = Wav2Vec2ForCTC.from_pretrained(MAIN_MODEL_DIR)
        assert isinstance(processor, Wav2Vec2Processor)

    emotion_model = joblib.load(EMOTION_MODEL_DIR)
    emotion_scaler = joblib.load(EMOTION_SCALER_DIR)
    emotion_encoder = joblib.load(EMOTION_ENCODER_DIR)

    return MLModels(
        processor=processor,
        main_model=main_model,
        emotion_model=emotion_model,
        emotion_scaler=emotion_scaler,
        emotion_encoder=emotion_encoder,
    )
