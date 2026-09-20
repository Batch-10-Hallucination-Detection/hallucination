from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.app.detection.member4_pipeline import analyze_responses


app = FastAPI(
    title="Hallucination Detection API",
    description="Member 4 analysis API",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    responses: List[str]
    token_confidence: float = 0.8


@app.get("/")
def root():
    return {
        "message": "Hallucination Detection API is running"
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):

    result = analyze_responses(
        request.responses,
        token_confidence=request.token_confidence
    )

    return result