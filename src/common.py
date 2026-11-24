"""Shared paths, I/O helpers and experiment tracking."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = ROOT / "models"
CACHE = ROOT / "cache"
for _p in (RESULTS, PREDS, FIGS, TABLES, MODELS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
EXPERIMENT = "mtsamples-specialty-classification"

# Display names used in every table/figure
MODEL_NAMES = {
    "logreg": "TF-IDF + Logistic Regression",
    "svm": "TF-IDF + Linear SVM",
    "rf": "TF-IDF + Random Forest",
    "logreg_trunc256": "TF-IDF + LR (first 256 tokens)",
    "biobert": "BioBERT v1.2",
    "clinicalbert": "Bio_ClinicalBERT",
    "pubmedbert": "PubMedBERT (BiomedBERT)",
    "roberta": "RoBERTa-base",
}


def save_probs(model: str, split: str, probs: np.ndarray) -> None:
    d = PREDS / model
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{split}.npy", probs.astype(np.float32))


def load_probs(model: str, split: str) -> np.ndarray:
    return np.load(PREDS / model / f"{split}.npy")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def mlflow_setup():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow
