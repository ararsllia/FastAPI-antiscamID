"""Preprocessing teks. HARUS identik dengan fungsi preprocess() di notebook 02_Training_AntiScamID.ipynb."""
import re
from pathlib import Path

import pandas as pd

NORM_PATH = Path(__file__).resolve().parent.parent / "model" / "key_norm.csv"

norm_dict: dict[str, str] = {}
if NORM_PATH.exists():
    _kn = pd.read_csv(NORM_PATH)
    norm_dict = dict(zip(_kn["singkat"].astype(str).str.lower().str.strip(),
                         _kn["hasil"].astype(str).str.lower().str.strip()))

RE_URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|co\.id|id|xyz|site|online|click|top|vip|icu|net|org|info|me|ly|blogspot\.com)(?:/\S*)?)", re.I)
RE_APK = re.compile(r"\b\S+\.apk\b", re.I)
RE_PHONE = re.compile(r"(\+?62|\b0)8[\d\-\s]{7,13}\d|\b0?8\d{2}x+\d*|\+?62\d{2}x+", re.I)
RE_MONEY = re.compile(r"(rp\.?\s?[\d.,]+(\s?(jt|juta|rb|ribu|k))?|\$\s?[\d.,]+|£\s?[\d.,]+|\b\d+[.,]?\d*\s?(jt|juta|rb|ribu)\b)", re.I)
RE_NUM = re.compile(r"\d+")


def preprocess(text: str) -> str:
    t = str(text).lower()
    t = RE_APK.sub(" <APK> ", t)
    t = RE_URL.sub(" <URL> ", t)
    t = RE_PHONE.sub(" <PHONE> ", t)
    t = RE_MONEY.sub(" <MONEY> ", t)
    t = RE_NUM.sub(" <NUM> ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9<>_\s]", " ", t)
    return " ".join(norm_dict.get(w, w) for w in t.split())
