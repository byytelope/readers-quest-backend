# pyright: basic

from io import BytesIO

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler

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

    mfccs = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
    mfccs_normalized = models.emotion_scaler.transform(mfccs_mean)
    emotion_num = models.emotion_model.predict(mfccs_normalized)
    emotion = models.emotion_encoder.inverse_transform(emotion_num)
    frustration = emotion in ["frustrated", "sad"]

    input_values = models.processor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    logits = models.main_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = models.processor.batch_decode(predicted_ids)[0]
    phonemes = map_to_phonemes(transcription)

    return transcription, phonemes, frustration
