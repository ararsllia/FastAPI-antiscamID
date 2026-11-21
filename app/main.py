
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.preprocess import preprocess
from app.rules import find_indicators, build_advice


# ==========================================
# CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent.parent / "model"

MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
AI_API_KEY = os.getenv("AI_API_KEY")

state = {}


# ==========================================
# MODEL LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(BASE / "antiscam_model.joblib")
    state["meta"] = json.loads(
        (BASE / "model_meta.json").read_text()
    )

    classes = list(state["model"].classes_)

    state["idx"] = (
        classes.index(1)
        if 1 in classes
        else classes.index("penipuan")
    )

    yield
    state.clear()


app = FastAPI(
    title="AntiScamID AI",
    description="AI-powered Scam Message Detection Service",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# REQUEST SCHEMAS
# ==========================================

class PredictIn(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CHARS
    )


class BatchIn(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50
    )


# ==========================================
# RESPONSE SCHEMAS
# ==========================================

class IndicatorOut(BaseModel):
    kode: str
    keterangan: str


class PredictOut(BaseModel):
    label: str
    prob_penipuan: float
    model_version: str
    indicators: list[IndicatorOut] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    processing_ms: float


# ==========================================
# API KEY AUTHENTICATION
# ==========================================

def verify_api_key(
    api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    )
):
    # Local development: authentication can be disabled
    # by leaving AI_API_KEY unset.
    # Deployment: AI_API_KEY must be configured.
    if AI_API_KEY and (
        not api_key
        or not secrets.compare_digest(api_key, AI_API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# ==========================================
# CLASSIFICATION
# ==========================================

def decide(probability: float) -> str:
    threshold = state["meta"]["threshold"]

    if probability >= threshold["T_BEST"]:
        return "PENIPUAN"

    if probability >= threshold["T_LOW"]:
        return "PERLU_REVIEW"

    return "AMAN"


def predict_probs(texts: list[str]) -> list[float]:
    processed_texts = [
        preprocess(text)
        for text in texts
    ]

    probabilities = state["model"].predict_proba(
        processed_texts
    )

    return probabilities[:, state["idx"]].tolist()


# ==========================================
# MESSAGE ANALYSIS
# ==========================================

def analyze_message(text: str) -> PredictOut:
    start_time = time.perf_counter()

    # 1. Machine Learning prediction
    probability = predict_probs([text])[0]

    label = decide(probability)

    # 2. Rule-based indicators
    indicators = find_indicators(text)

    # 3. Map model labels to rules.py labels
    if label == "PENIPUAN":
        advice_label = "indikasi penipuan"

    elif label == "PERLU_REVIEW":
        advice_label = "belum dapat dipastikan"

    else:
        advice_label = "pesan normal"

    # 4. Generate safety recommendations
    advice = build_advice(
        indicators,
        advice_label
    )

    # 5. Measure processing time
    processing_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return PredictOut(
        label=label,
        prob_penipuan=round(probability, 4),
        model_version=state["meta"]["model_version"],
        indicators=indicators,
        advice=advice,
        processing_ms=round(processing_ms, 2)
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    if not state:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    meta = state["meta"]

    return {
        "status": "ok",
        "model_version": meta["model_version"],
        "algoritma": "TFIDF + LogisticRegression"
    }


# ==========================================
# SINGLE MESSAGE PREDICTION
# ==========================================

@app.post(
    "/predict",
    response_model=PredictOut
)
def predict(
    body: PredictIn,
    _auth=Header(default=None, include_in_schema=False)
):
    verify_api_key(_auth)
    return analyze_message(body.text)


# ==========================================
# BATCH MESSAGE PREDICTION
# ==========================================

@app.post(
    "/predict/batch",
    response_model=list[PredictOut]
)
def predict_batch(
    body: BatchIn,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    )
):
    verify_api_key(x_api_key)

    results = []

    for text in body.texts:
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Message cannot be empty"
            )

        if len(text) > MAX_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"Message exceeds {MAX_CHARS} characters"
            )

        results.append(analyze_message(text))

    return results
#testing