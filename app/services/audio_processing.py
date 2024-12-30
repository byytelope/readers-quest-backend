# pyright: basic

from io import BytesIO

import librosa
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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


def process_audio(
    audio_bytes: BytesIO, processor: Wav2Vec2Processor, model: Wav2Vec2ForCTC
) -> tuple[str, str]:
    """
    Processes an audio file for phoneme recognition.
    """
    waveform, _ = librosa.load(audio_bytes, sr=16000)
    input_values = processor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    phonemes = map_to_phonemes(transcription)

    return transcription, phonemes
