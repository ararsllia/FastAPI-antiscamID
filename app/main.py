import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from app.preprocess import preprocess
from app.rules import build_advice, find_indicators

log = logging.getLogger("antiscam")
logging.basicConfig(level=logging.INFO)

# ==========================================
# CONFIGURATION
# ==========================================

BASE = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
AI_API_KEY = os.getenv("AI_API_KEY")

state: dict = {}


# ==========================================
# MODEL LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(BASE / "antiscam_model.joblib")
    state["meta"] = json.loads((BASE / "model_meta.json").read_text())

    # classes_ bisa [0, 1] (numpy int) atau ["normal", "penipuan"]
    classes = [c.item() if hasattr(c, "item") else c for c in state["model"].classes_]
    state["idx"] = classes.index(1) if 1 in classes else classes.index("penipuan")

    # tes sekali saat startup supaya error model langsung kelihatan di log, bukan saat request
    state["model"].predict_proba([preprocess("tes")])
    log.info("Model loaded: %s | classes=%s", state["meta"].get("model_version"), classes)
    yield
    state.clear()


app = FastAPI(
    title="AntiScamID AI",
    description="AI-powered Scam Message Detection Service",
    version="1.0.1",
    lifespan=lifespan,
)


# ==========================================
# REQUEST / RESPONSE SCHEMAS
# ==========================================

def _clean(v: str) -> str:
    v = (v or "").strip()
    if not v:
        raise ValueError("Message cannot be empty")
    if len(v) > MAX_CHARS:
        raise ValueError(f"Message exceeds {MAX_CHARS} characters")
    return v


class PredictIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return _clean(v)


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v: list[str]) -> list[str]:
        return [_clean(t) for t in v]


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
# API KEY AUTHENTICATION (dipakai lewat Depends, JANGAN dipanggil manual)
# ==========================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)   # muncul sebagai tombol "Authorize" di /docs


def verify_api_key(x_api_key: str | None = Security(api_key_header)):
    # AI_API_KEY kosong = auth mati (development). Deployment WAJIB diisi.
    if AI_API_KEY and (not x_api_key or not secrets.compare_digest(x_api_key.encode(), AI_API_KEY.encode())):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ==========================================
# CLASSIFICATION
# ==========================================

def decide(probability: float) -> str:
    t = state["meta"]["threshold"]
    if probability >= t["T_BEST"]:
        return "PENIPUAN"
    if probability >= t["T_LOW"]:
        return "PERLU_REVIEW"
    return "AMAN"


ADVICE_LABEL = {"PENIPUAN": "indikasi penipuan", "PERLU_REVIEW": "belum dapat dipastikan", "AMAN": "pesan normal"}


def predict_probs(texts: list[str]) -> list[float]:
    proba = state["model"].predict_proba([preprocess(t) for t in texts])
    return [float(p) for p in proba[:, state["idx"]]]


def analyze_message(text: str, probability: float | None = None) -> PredictOut:
    start = time.perf_counter()
    if probability is None:
        probability = predict_probs([text])[0]
    label = decide(probability)
    indicators = find_indicators(text)
    advice = build_advice(indicators, ADVICE_LABEL[label])
    return PredictOut(
        label=label,
        prob_penipuan=round(probability, 4),
        model_version=str(state["meta"].get("model_version", "unknown")),
        indicators=indicators,
        advice=advice,
        processing_ms=round((time.perf_counter() - start) * 1000, 2),
    )


def _safe(fn):
    """Kalau model error, log traceback lengkap & balas 503 (bukan 500 tanpa info)."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=503, detail="Analysis is temporarily unavailable")


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health")
def health():
    if not state:
        raise HTTPException(status_code=503, detail="Model is not available")
    meta = state["meta"]
    return {"status": "ok", "model_version": meta.get("model_version"), "algoritma": meta.get("algoritma", "TFIDF + LogisticRegression")}


@app.post("/predict", response_model=PredictOut, dependencies=[Depends(verify_api_key)])
def predict(body: PredictIn):
    return _safe(lambda: analyze_message(body.text))


@app.post("/predict/batch", response_model=list[PredictOut], dependencies=[Depends(verify_api_key)])
def predict_batch(body: BatchIn):
    def run():
        probs = predict_probs(body.texts)   # satu kali panggil model untuk semua pesan
        return [analyze_message(t, p) for t, p in zip(body.texts, probs)]
    return _safe(run)
