import unittest
from io import BytesIO

from app.models import load_models
from app.services.audio_processing import process_audio


class TestPhonemesRecognition(unittest.TestCase):
    def test_process_audio(self):
        print("RUNNING TEST: Audio")

        ml_models = load_models()
        audio_path = "./audio/hello.wav"

        with open(audio_path, "rb") as f:
            audio_bytes = BytesIO(f.read())

        transcription, phonemes = process_audio(
            audio_bytes, ml_models.processor, ml_models.model
        )

        self.assertIsInstance(transcription, str)
        self.assertIsInstance(phonemes, str)
