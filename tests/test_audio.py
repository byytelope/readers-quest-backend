import unittest
from io import BytesIO

from app.models import load_models
from app.services.audio_processing import convert_to_wav, process_audio


class TestPhonemesRecognition(unittest.TestCase):
    def test_process_audio(self):
        print("RUNNING TEST: Audio")

        ml_models = load_models()
        audio_path = "./audio/321.m4a"

        with open(audio_path, "rb") as f:
            audio_bytes = BytesIO(f.read())
            extension = audio_path.split(".")[-1].lower()
            if extension != "wav":
                audio_bytes = convert_to_wav(audio_bytes, format=extension)

        transcription, phonemes, frustration = process_audio(audio_bytes, ml_models)

        print(transcription, phonemes, frustration)

        self.assertIsInstance(transcription, str)
        self.assertIsInstance(phonemes, str)
        self.assertIsInstance(frustration, bool)
