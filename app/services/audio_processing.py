# pyright: basic

from io import BytesIO

import librosa
import torch
from pydub import AudioSegment

from app.models import MLModels
from utils.phoneme_utils import map_to_phonemes


def convert_to_wav(audio_bytes: BytesIO, format: str) -> BytesIO:
    """
    Converts audio data to wav format.
    """
    audio = AudioSegment.from_file(audio_bytes, format=format)
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    return wav_buffer


def process_audio(audio_bytes: BytesIO, models: MLModels) -> tuple[str, str, bool]:
    """
    Processes an audio file for phoneme recognition.
    """
    waveform, _ = librosa.load(audio_bytes, sr=16000)

    input_values = models.main_processor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    logits = models.main_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = models.main_processor.batch_decode(predicted_ids)[0]
    phonemes = map_to_phonemes(transcription)

    emotion_inputs = models.emotion_processor(
        waveform, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = models.emotion_model(**emotion_inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()

    emotions = ["neutral", "happy", "angry", "sad"]
    print(emotions[int(predicted_id)])

    frustrated = emotions[int(predicted_id)] in ("angry", "sad")

    return transcription, phonemes, frustrated
