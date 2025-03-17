# pyright: basic

import logging
import os
from dataclasses import dataclass

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

MAIN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MAIN_MODEL_DIR = "models/speech"
# EMOTION_MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
EMOTION_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
EMOTION_MODEL_DIR = "models/emotion"

logger = logging.getLogger("uvicorn")


@dataclass
class MLModels:
    main_processor: Wav2Vec2Processor
    main_model: Wav2Vec2ForCTC
    emotion_feature_extractor: Wav2Vec2FeatureExtractor
    emotion_model: Wav2Vec2ForCTC


def load_models() -> MLModels:
    if not os.path.exists(MAIN_MODEL_DIR):
        print("Downloading main model...")
        main_processor = Wav2Vec2Processor.from_pretrained(MAIN_MODEL_NAME)
        main_model = Wav2Vec2ForCTC.from_pretrained(MAIN_MODEL_NAME)
        assert isinstance(main_processor, Wav2Vec2Processor)

        os.makedirs(MAIN_MODEL_DIR, exist_ok=True)
        main_processor.save_pretrained(MAIN_MODEL_DIR)
        main_model.save_pretrained(MAIN_MODEL_DIR)
        print(f"Main model saved to {MAIN_MODEL_DIR}")
    else:
        logger.info(f"Loading main model from {MAIN_MODEL_DIR}...")
        main_processor = Wav2Vec2Processor.from_pretrained(MAIN_MODEL_DIR)
        main_model = Wav2Vec2ForCTC.from_pretrained(MAIN_MODEL_DIR)
        assert isinstance(main_processor, Wav2Vec2Processor)

    if not os.path.exists(EMOTION_MODEL_DIR):
        print("Downloading emotion model...")
        emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            EMOTION_MODEL_NAME
        )
        emotion_model = Wav2Vec2ForCTC.from_pretrained(EMOTION_MODEL_NAME)
        assert isinstance(emotion_feature_extractor, Wav2Vec2FeatureExtractor)

        os.makedirs(EMOTION_MODEL_DIR, exist_ok=True)
        emotion_feature_extractor.save_pretrained(EMOTION_MODEL_DIR)
        emotion_model.save_pretrained(EMOTION_MODEL_DIR)
        print(f"Emotion model saved to {EMOTION_MODEL_DIR}")
    else:
        logger.info(f"Loading emotion model from {EMOTION_MODEL_DIR}...")
        emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            EMOTION_MODEL_DIR
        )
        emotion_model = Wav2Vec2ForCTC.from_pretrained(EMOTION_MODEL_DIR)
        assert isinstance(emotion_feature_extractor, Wav2Vec2FeatureExtractor)

    return MLModels(
        main_processor=main_processor,
        main_model=main_model,
        emotion_feature_extractor=emotion_feature_extractor,
        emotion_model=emotion_model,
    )
