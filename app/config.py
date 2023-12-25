from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_STR: str = "/api/v1"
    PROJECT_NAME: str = "Trip Advisor Sentiment Prediction API"

settings = Settings()