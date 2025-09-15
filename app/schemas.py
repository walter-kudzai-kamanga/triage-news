from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    probabilities: Dict[str, float]
    processed_text: str

class TrainingRequest(BaseModel):
    model_type: str = "naive_bayes"

class TrainingResponse(BaseModel):
    message: str
    model_type: str
    accuracy: float
    training_samples: int

class BatchPredictionRequest(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, str]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
