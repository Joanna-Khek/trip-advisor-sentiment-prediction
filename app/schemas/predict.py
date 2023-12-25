from pydantic import BaseModel

class PredictionResults(BaseModel):
    prediction: str
    probability: float