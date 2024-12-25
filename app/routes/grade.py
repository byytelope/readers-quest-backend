from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.audio_processing import process_audio

router = APIRouter(prefix="/grade", tags=["Grade pronunciation"])


@router.post("/")
async def recognize_phonemes(request: Request, file: UploadFile = File(...)):
    """
    Recognize phonemes from an uploaded audio file.
    """
    try:
        audio = await file.read()
        audio_bytes = BytesIO(audio)
        transcription, phonemes = process_audio(
            audio_bytes,
            request.state.ml_models.processor,
            request.state.ml_models.model,
        )

        return {"transcription": transcription, "phonemes": phonemes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
