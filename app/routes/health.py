from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health Check"])


@router.get("")
def health_check():
    """
    Check API health status.
    """
    return {"status": "OK"}
