# pyright: basic

from io import BytesIO

import librosa
import numpy as np
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


def emotion_extract_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    features = np.array([])
    features = np.hstack((
        features,
        np.mean(librosa.feature.zero_crossing_rate(y=waveform).T, axis=0),
        np.mean(
            librosa.feature.chroma_stft(
                S=np.abs(librosa.stft(waveform)), sr=sample_rate
            ).T,
            axis=0,
        ),
        np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate).T, axis=0),
        np.mean(librosa.feature.rms(y=waveform).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate).T, axis=0),
    ))

    return features


def process_audio(audio_bytes: BytesIO, models: MLModels) -> tuple[str, str, bool]:
    """
    Processes an audio file for phoneme recognition.
    """
    waveform, _ = librosa.load(audio_bytes, sr=16000)
    emotion_features = emotion_extract_features(waveform, 16000).reshape(1, -1)
    predicted_emotion = models.emotion_model.predict(emotion_features)
    emotion = (
        models.emotion_label_encoder.inverse_transform(predicted_emotion)[0]
        .lower()
        .split("_")[-1]
    )
    frustrated = emotion in ["sad", "angry"]

    input_values = models.processor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    logits = models.main_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = models.processor.batch_decode(predicted_ids)[0]
    phonemes = map_to_phonemes(transcription)

    return transcription, phonemes, frustrated
