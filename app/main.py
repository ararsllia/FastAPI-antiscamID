import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent / "model"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # model dimuat sekali saat server start
    state["model"] = joblib.load(BASE / "antiscam_model.joblib")
    state["meta"] = json.loads((BASE / "model_meta.json").read_text())
    classes = list(state["model"].classes_)
    # sesuaikan dengan label kelas penipuan di datamu (1 / "penipuan")
    state["idx"] = classes.index(1) if 1 in classes else classes.index("penipuan")
    yield
    state.clear()


app = FastAPI(title="API Deteksi Penipuan", version="1.0.0", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class PredictOut(BaseModel):
    label: str            # PENIPUAN | PERLU_REVIEW | AMAN
    prob_penipuan: float
    model_version: str


def decide(p: float) -> str:
    t = state["meta"]["threshold"]
    if p >= t["T_BEST"]:
        return "PENIPUAN"
    if p >= t["T_LOW"]:
        return "PERLU_REVIEW"
    return "AMAN"


def predict_probs(texts: list[str]) -> list[float]:
    return state["model"].predict_proba(texts)[:, state["idx"]].tolist()


@app.get("/health")
def health():
    m = state["meta"]
    return {"status": "ok", "model_version": m["model_version"], "metrik_test": m["metrik_test"]}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    p = predict_probs([body.text])[0]
    return PredictOut(label=decide(p), prob_penipuan=round(p, 4),
                      model_version=state["meta"]["model_version"])


@app.post("/predict/batch", response_model=list[PredictOut])
def predict_batch(body: BatchIn):
    v = state["meta"]["model_version"]
    return [PredictOut(label=decide(p), prob_penipuan=round(p, 4), model_version=v)
            for p in predict_probs(body.texts)]