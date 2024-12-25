from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models import load_models
from app.routes import grade, health


@asynccontextmanager
async def lifespan(_: FastAPI):
    ml_models = load_models()
    yield {"ml_models": ml_models}

    del ml_models


client = FastAPI(lifespan=lifespan)
client.include_router(grade.router)
client.include_router(health.router)
