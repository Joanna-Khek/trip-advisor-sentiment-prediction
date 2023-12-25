from loguru import logger
from fastapi import APIRouter, HTTPException
from src.predict import make_prediction
from app import schemas, __version__
from app.config import settings

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name = settings.PROJECT_NAME,
        api_version = __version__
    )
    return health.dict()

@api_router.get("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_text: str):

    logger.info(f"Making predictions on inputs: {input_text}")
    results = make_prediction(input_text)

    logger.info(f"Prediction results: {results.get('prediction')}")
    logger.info(f"Probability: {results.get('probability')}")

    return results
