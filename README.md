# AntiScamID AI

### Intelligent Scam Message Detection Service for Indonesian Text

AntiScamID AI is the machine learning service behind **AntiScamID**, a capstone project that helps people do a first check on suspicious Indonesian-language messages (SMS, WhatsApp, Telegram). It classifies a message, shows the suspicious indicators it found, and gives practical safety advice before the user replies, clicks a link, or transfers money.

The service is a lightweight **FastAPI** REST API. The web app backend and the Telegram bot call it over the internal network.

> ⚠️ Results are an **early indication** based on a model and rules. They are **not** proof of fraud and **not** a legal decision.

---

## Table of Contents
1. [Key Features](#key-features)
2. [How It Works](#how-it-works)
3. [Model & Performance](#model--performance)
4. [Project Structure](#project-structure)
5. [Quick Start: Try It in 10 Minutes](#quick-start-try-it-in-10-minutes)
6. [API Reference](#api-reference)
7. [Integration Guide (for RPL & Network teams)](#integration-guide)
8. [Docker Deployment](#docker-deployment)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Limitations](#limitations)
12. [Credits & Disclaimer](#credits--disclaimer)

---

## Key Features

| Feature | Description |
|---|---|
| **Message classification** | TF-IDF + Logistic Regression model trained on Indonesian messages |
| **3-level result** | `indikasi penipuan` · `pesan normal` · `belum dapat dipastikan` |
| **Rule-based indicators** | Links, OTP/PIN requests, `.apk` files, prize claims, block threats, urgency, upfront fees, impersonated institutions. These are shown **separately** from the model prediction. |
| **Safety advice** | Practical recommendations based on the detected indicators |
| **Batch analysis** | Up to 50 messages per request |
| **API key auth** | `X-API-Key` header, enabled through an environment variable |
| **Health check** | `/health` returns the service status, model version, and thresholds |
| **Test page** | A simple web UI at `/` for quick manual testing and demos |

## How It Works

```
Message ──► Preprocessing ──► TF-IDF + Logistic Regression ──► Score (0–1)
   │        (lowercase, slang       (word + character n-grams)       │
   │         normalization,                                           ▼
   │         <URL> <PHONE>                                  Thresholds T_LOW / T_HIGH
   │         <MONEY> <APK> tokens)                                    │
   │                                                                  ▼
   └──────► Rule-based indicators ──► Safety advice ──►  JSON response
```

1. **Preprocessing.** Text is lowercased, and slang is normalized (`dgn` → `dengan`). Links, phone numbers, money amounts, and `.apk` files are replaced with tokens instead of being deleted, because they are strong scam signals.
2. **Classification.** The model outputs a score between 0 and 1.
3. **Thresholds.** The score is mapped to 3 categories:
   - `score ≥ T_HIGH` → **indikasi penipuan**
   - `score ≤ T_LOW` → **pesan normal**
   - in between → **belum dapat dipastikan**

   The thresholds are chosen on training data (out-of-fold), never on the test set.
4. **Indicators & advice.** Regex rules explain *what* looks suspicious. One indicator alone does not prove fraud.

## Model & Performance

| Item | Value |
|---|---|
| Text representation | TF-IDF (word 1–2 gram + character 2–5 gram) |
| Classifier | Logistic Regression (`class_weight="balanced"`) |
| Tuning | Grid search with stratified 5-fold cross-validation |
| Test set | 25% of **real** messages only (no synthetic data), locked before training |

Results on the held-out test set:

| Model | F1 (scam) | Precision | Recall | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** (deployed) | **0.901** | **0.936** | 0.869 | 0.944 | 0.988 |
| IndoBERT (`indobert-base-p1`, comparison) | 0.902 | 0.876 | 0.929 | 0.940 | 0.990 |

**Why TF-IDF?** It has nearly the same F1 as IndoBERT, higher precision (it flags normal messages as scams less often), it is a few MB instead of about 500 MB, and it runs fast on a CPU.

**Dataset:** Indonesian SMS dataset by Rahmi & Wibisono (CC BY-SA 4.0), plus a few real messages from news articles, plus synthetic messages covering recent scam patterns (used for training only). See `notebooks/` for EDA and training details.

## Project Structure

```
antiscam-ai-api/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── preprocess.py    # must be identical to preprocess() in the training notebook
│   ├── rules.py         # rule-based indicators + advice
│   └── static/
│       └── index.html   # test page served at "/"
├── model/
│   ├── antiscam_model.joblib   # trained model (from notebook 02)
│   ├── model_meta.json         # version, thresholds, metrics, sklearn version
│   └── key_norm.csv            # slang normalization dictionary
├── requirements.txt
├── Dockerfile
└── README.md
```

> The model files are produced by `02_Training_AntiScamID.ipynb`. Ask the AI team for `antiscam_model.joblib` and `model_meta.json` if they are not in the repo.

---

## Quick Start: Try It in 10 Minutes

### Prerequisites
- **Python 3.12 or 3.13.** ⚠️ Do **not** use Python 3.14/3.15 yet, because numpy/scikit-learn may not have ready-made packages for those versions and the install will fail.
- Git (optional), plus VS Code or any terminal

Check your version:
```bash
python3 --version
```

### 1. Get the code
```bash
git clone <repo-url>
cd antiscam-ai-api
```
Or download and extract the zip from the team drive.

### 2. Put the model files in `model/`
```
model/antiscam_model.joblib
model/model_meta.json
model/key_norm.csv
```

### 3. Create a virtual environment and install
**macOS / Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
> The `scikit-learn` version in `requirements.txt` **must match** `sklearn_version` in `model/model_meta.json` (currently `1.8.0`).

### 4. Run the service
```bash
uvicorn app.main:app --reload --port 8000
```
You should see `Uvicorn running on http://127.0.0.1:8000`.

### 5. Try it
| What | URL |
|---|---|
| 🖥️ Test page (paste a message, click **Periksa Pesan**) | http://localhost:8000 |
| 📘 Interactive API docs (Swagger) | http://localhost:8000/docs |
| ❤️ Health check | http://localhost:8000/health |

From the terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Selamat! Anda menang hadiah Rp100 juta, klik bit.ly/xx segera"}'
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | – | Test page (HTML) |
| GET | `/health` | – | Service status, model version, thresholds |
| POST | `/predict` | `X-API-Key`* | Classify one message |
| POST | `/predict/batch` | `X-API-Key`* | Classify up to 50 messages |

\* Only required when the `AI_API_KEY` environment variable is set.

### `POST /predict`
**Request**
```json
{ "text": "Selamat! Anda menang hadiah Rp100 juta, klik bit.ly/xx segera" }
```
**Response `200`**
```json
{
  "label": "indikasi penipuan",
  "score": 0.9213,
  "model_version": "tfidf-lr-v1.0",
  "indicators": [
    { "kode": "tautan", "keterangan": "Mengandung tautan" },
    { "kode": "klaim_hadiah", "keterangan": "Klaim hadiah/undian" },
    { "kode": "desakan_waktu", "keterangan": "Desakan untuk segera bertindak" }
  ],
  "advice": [
    "Jangan langsung membuka tautan. Cek alamat situs resmi secara manual.",
    "Hadiah resmi tidak pernah meminta biaya di muka. Cek ke kanal resmi penyelenggara."
  ],
  "processing_ms": 1.05
}
```
*Illustrative example. Real values come from the model.*

| Field | Type | Notes |
|---|---|---|
| `label` | string | `indikasi penipuan` · `pesan normal` · `belum dapat dipastikan` |
| `score` | float 0–1 | Model output. **Do not** display it as "92% scam". |
| `model_version` | string | Store this with each check for traceability |
| `indicators` | list | Rule-based, **independent** of the model |
| `advice` | list of strings | Ready to show to users |
| `processing_ms` | float | Server-side processing time |

**Errors**

| Code | When |
|---|---|
| `401` | Wrong or missing `X-API-Key` |
| `422` | Empty text, text longer than `MAX_CHARS` (default 2000), or invalid JSON |

### `POST /predict/batch`
```json
{ "texts": ["pesan 1", "pesan 2"] }
```
Returns a list of `/predict` responses in the same order.

### `GET /health`
```json
{ "status": "ok", "model_version": "tfidf-lr-v1.0", "algoritma": "TFIDF + LogisticRegression",
  "threshold": { "T_LOW": 0.2, "T_BEST": 0.5, "T_HIGH": 0.8 } }
```

---

## Integration Guide

### For the RPL team (Next.js backend)
Call the AI service **from the server side** (a tRPC procedure or API route), never directly from the browser, so the API key stays secret.

```ts
// server-side only
export async function checkMessage(text: string) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 5000); // 5 s timeout
  try {
    const res = await fetch(`${process.env.AI_SERVICE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": process.env.AI_API_KEY! },
      body: JSON.stringify({ text }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`AI service error ${res.status}`);
    return await res.json(); // { label, score, model_version, indicators, advice }
  } catch {
    return null; // show "Pemeriksaan sedang tidak tersedia". Never fake a result.
  } finally {
    clearTimeout(timer);
  }
}
```
Suggested columns for the `checks` table: `label`, `score`, `model_version`, `created_at`. Avoid storing the full message text unless the user agrees to it.

### For the Telegram bot
```python
import os, requests

def check(text: str) -> str:
    try:
        r = requests.post(f"{os.environ['AI_SERVICE_URL']}/predict",
                          json={"text": text},
                          headers={"X-API-Key": os.environ["AI_API_KEY"]}, timeout=5)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return "Maaf, pemeriksaan sedang tidak tersedia. Coba lagi nanti."
    ind = "\n".join(f"• {i['keterangan']}" for i in d["indicators"]) or "• Tidak ada indikator khusus"
    adv = "\n".join(f"– {a}" for a in d["advice"])
    return f"Hasil: {d['label'].upper()}\n\nIndikator:\n{ind}\n\nSaran:\n{adv}\n\n(Hasil ini indikasi, bukan keputusan hukum.)"
```

### For the Network team
- Run the AI container on an **internal Docker network**. The backend and bot reach it at, for example, `http://ai-service:8000`.
- **Do not** expose port 8000 to the internet.
- Set `AI_API_KEY` from a secret or `.env` file. Never commit it to the repo.
- Use `/health` for container health checks and monitoring.
- Suggested tests: latency, behavior when the AI service is down, rate limiting at the reverse proxy, and checking that the port cannot be reached from outside.

---

## Docker Deployment

```bash
docker build -t antiscam-ai .
docker run -d --name ai-service -p 8000:8000 \
  -e AI_API_KEY=change-me -e CORS_ORIGINS=https://your-web-domain \
  antiscam-ai
```

Example `docker-compose.yml` snippet:
```yaml
services:
  ai-service:
    build: ./antiscam-ai-api
    environment:
      AI_API_KEY: ${AI_API_KEY}
      CORS_ORIGINS: https://your-web-domain
    networks: [internal]
    restart: unless-stopped
    # no "ports:" means it is not exposed publicly; the backend reaches it via http://ai-service:8000

networks:
  internal:
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AI_API_KEY` | *(empty = auth off)* | Required in deployment. Sent by clients as the `X-API-Key` header. |
| `MAX_CHARS` | `2000` | Maximum message length |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set this to your web domain in production. |
| `MODEL_DIR` | `./model` | Folder containing the model files |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `metadata-generation-failed` / `Unknown compiler(s)` while installing numpy or scikit-learn | Your Python is too new (3.14/3.15). Install **Python 3.13**, recreate `.venv`, and reinstall. |
| `zsh: command not found: uvicorn` | The install failed or the venv is not active. Run `source .venv/bin/activate` and reinstall. |
| `InconsistentVersionWarning` or the model fails to load | The `scikit-learn` version differs from `sklearn_version` in `model_meta.json`. Install that exact version. |
| `FileNotFoundError: antiscam_model.joblib` | Model files are missing from `model/`. |
| Test page says *"Pemeriksaan sedang tidak tersedia"* | The API is not running, or the API URL field on the page is wrong. |
| `401 API key tidak valid` | Fill in the API key field on the test page or send the `X-API-Key` header. |
| Browser CORS error from another frontend | Add that frontend's origin to `CORS_ORIGINS`. |

## Limitations
- The real training data is small and mostly from around 2015, and some labels are noisy. Newer scam patterns are partly covered by synthetic data.
- The model can miss new scam patterns or flag unusual normal messages. The `belum dapat dipastikan` category exists for this reason.
- The model only classifies text. It does not verify links, phone numbers, or images, and it cannot detect edited transfer receipts.
- Retrain periodically using reports that admins have verified.

## Credits & Disclaimer
- **Team:** AntiScamID Capstone, Informatika, UIN Sunan Kalijaga, 2026 (AI · RPL · Network).
- **Dataset:** Rahmi, F. & Wibisono, Y. (2016). *Aplikasi SMS Spam Filtering pada Android menggunakan Naive Bayes.* CC BY-SA 4.0.

AntiScamID AI is an informational and preventive aid. Its output is **not** definitive proof of fraud or a legal determination. Always verify suspicious messages through official channels.

---

**AntiScamID AI: making digital communication safer with AI.**
