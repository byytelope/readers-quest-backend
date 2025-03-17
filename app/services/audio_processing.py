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

    # emotion_features = models.emotion_feature_extractor(
    #     waveform, sampling_rate=16000, padding=True, return_tensors="pt"
    # )

    # print("HERE")

    # with torch.no_grad():
    #     input_values = emotion_features.input_values
    #     if not isinstance(input_values, torch.Tensor):
    #         input_values = torch.tensor(input_values)

    #     outputs = models.emotion_model(input_values)
    #     predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)
    #     predicted_label = torch.argmax(predictions, dim=-1)
    #     emotion = models.emotion_model.config.id2label[int(predicted_label.item())]

    #     print(emotion)

    # frustrated = emotion in ["sad", "angry"]
    frustrated = False

    input_values = models.main_processor(
        waveform, return_tensors="pt", sampling_rate=16000
    ).input_values

    logits = models.main_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = models.main_processor.batch_decode(predicted_ids)[0]
    phonemes = map_to_phonemes(transcription)

    return transcription, phonemes, frustrated
