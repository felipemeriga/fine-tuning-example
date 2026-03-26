"""
FastAPI server for the fine-tuned DistilBERT sentiment analysis model.

Run with:
    uv run uvicorn api:app --reload

Then send requests:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"text": "This movie was absolutely fantastic!"}'
"""

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_DIR = "./sentiment-model"
LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Positive"}

# Store the classifier in app state so it's loaded once at startup
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model from {MODEL_DIR} (device={'cuda' if device == 0 else 'cpu'})...")
    classifier = pipeline("sentiment-analysis", model=MODEL_DIR, device=device)
    print("Model loaded and ready.")
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="Binary sentiment classification using a fine-tuned DistilBERT model.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float


class BatchPredictRequest(BaseModel):
    texts: list[str]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify the sentiment of a single text."""
    result = classifier(request.text)[0]
    return PredictResponse(
        text=request.text,
        label=LABEL_MAP[result["label"]],
        score=round(result["score"], 4),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Classify the sentiment of multiple texts in one request."""
    results = classifier(request.texts)
    predictions = [
        PredictResponse(
            text=text,
            label=LABEL_MAP[result["label"]],
            score=round(result["score"], 4),
        )
        for text, result in zip(request.texts, results)
    ]
    return BatchPredictResponse(predictions=predictions)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_DIR}
