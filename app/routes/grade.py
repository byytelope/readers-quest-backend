from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.services.audio_processing import convert_to_wav, process_audio
from utils.phoneme_utils import calculate_grade, map_to_phonemes

router = APIRouter(prefix="/grade", tags=["Grade pronunciation"])


@router.post("/")
async def grade_pronunciation(
    request: Request, audio: UploadFile = File(...), expected_text: str = Form(...)
):
    """
    Recognize phonemes from an uploaded audio file.
    """
    try:
        assert isinstance(audio.content_type, str)
        audio_format = audio.content_type.split("/")[-1].removeprefix("x-")
        audio_bytes = BytesIO(await audio.read())

        if audio_format != "wav":
            audio_bytes = convert_to_wav(audio_bytes, format=audio_format)

        transcription, phonemes, frustrated = process_audio(
            audio_bytes, request.state.ml_models
        )

        grade, feedback = calculate_grade(
            expected_text,
            transcription,
            map_to_phonemes(expected_text),
            phonemes,
        )

        return {"grade": grade, "frustrated": frustrated, "feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
