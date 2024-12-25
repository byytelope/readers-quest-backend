# pyright: basic

from io import BytesIO

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.phoneme_utils import map_to_phonemes


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

    # Get predicted IDs
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode predicted IDs to text
    transcription = processor.batch_decode(predicted_ids)[0]

    # Map text to phonemes
    phoneme_output = map_to_phonemes(transcription)

    return transcription, phoneme_output
