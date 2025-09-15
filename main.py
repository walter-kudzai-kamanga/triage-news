from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import joblib
from pathlib import Path

# Import your classifier
from app.utils.classifier import classifier

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Create main app
app = FastAPI(
    title="News Classifier Web",
    description="Web interface for news classification AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - serve from root static directory
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Pydantic models
class PredictionRequest(BaseModel):
    text: str


class TrainingRequest(BaseModel):
    model_type: str = "naive_bayes"


class BatchPredictionRequest(BaseModel):
    texts: List[str]


# Load model on startup
@app.on_event("startup")
async def startup_event():
    model_path = BASE_DIR / "models" / "news_classifier.pkl"
    try:
        if model_path.exists():
            classifier.load_model(model_path)
            print(f"Model loaded successfully: {classifier.model_type}")
        else:
            print("No pre-trained model found. Train a model using /api/train endpoint.")
    except Exception as e:
        print(f"Error loading model: {e}")


# Web routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# API routes
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "model_type": classifier.model_type
    }


@app.post("/api/train")
async def train_model(request: TrainingRequest):
    try:
        result = classifier.train(request.model_type)

        # Save the trained model
        model_dir = BASE_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "news_classifier.pkl"
        classifier.save_model(model_path)

        return {
            "message": "Model trained successfully",
            "model_type": result["model_type"],
            "accuracy": result["accuracy"],
            "training_samples": result["training_samples"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/api/predict")
async def predict_news(request: PredictionRequest):
    if classifier.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not trained. Please train the model first using /api/train endpoint."
        )

    try:
        result = classifier.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/predict/batch")
async def predict_batch_news(request: BatchPredictionRequest):
    if classifier.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not trained. Please train the model first using /api/train endpoint."
        )

    try:
        predictions = []
        for text in request.texts:
            result = classifier.predict(text)
            predictions.append({
                "text": text,
                "category": result["category"],
                "confidence": result["probabilities"].get(result["category"], 0.0)
            })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/api/model/info")
async def get_model_info():
    if classifier.model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded"
        )

    return {
        "model_type": classifier.model_type,
        "classes": classifier.label_encoder.classes_.tolist(),
        "is_trained": classifier.model is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )