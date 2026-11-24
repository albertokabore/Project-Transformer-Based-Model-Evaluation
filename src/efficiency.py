"""Computational efficiency, measured end-to-end (raw text in -> probabilities out) on CPU.

Each model is measured in a *fresh subprocess* so that peak memory (Windows peak working set /
Linux max RSS) is attributable to that model. Latency = single-document predictions (batch=1) on
the first 30 test documents after one warm-up call; throughput = batched prediction on 64 docs.
For transformers the deployable FullClassifier is used, and its predictions on the full test set
are compared to those of the cached-state pipeline (consistency check). The best transformer is
additionally evaluated after int8 dynamic quantisation of its Linear layers.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np

from .common import MODELS, RESULTS, load_probs, save_json

N_LAT, N_THR = 30, 64
TRANSFORMERS = ["clinicalbert", "pubmedbert", "biobert", "roberta"]


def peak_mem_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    return (getattr(mi, "peak_wset", None) or mi.rss) / 2**20


def measure(model_key: str, quantize: bool = False) -> dict:
    import psutil

    from .data import load_splits

    s = load_splits()
    texts = s.test.text.tolist()
    lab2id = {l: i for i, l in enumerate(s.labels)}
    y = s.test.label.map(lab2id).values
    rss0 = psutil.Process().memory_info().rss / 2**20
    t0 = time.perf_counter()
    if model_key in TRANSFORMERS:
        import torch

        from .transformer import CHECKPOINTS, FullClassifier

        torch.set_num_threads(8)
        model = FullClassifier(model_key, len(s.labels))
        if quantize:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        predict = model.predict_proba
        n_params = sum(p.numel() for p in model.parameters())
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            torch.save(model.state_dict(), os.path.join(d, "m.pt"))
            size_mb = os.path.getsize(os.path.join(d, "m.pt")) / 2**20
    else:
        import joblib

        model = joblib.load(MODELS / f"{model_key}.joblib")
        predict = model.predict_proba
        n_params = None
        size_mb = (MODELS / f"{model_key}.joblib").stat().st_size / 2**20
    load_s = time.perf_counter() - t0
    predict(texts[:1])  # warm-up
    lat = []
    for t in texts[:N_LAT]:
        t1 = time.perf_counter()
        predict([t])
        lat.append(time.perf_counter() - t1)
    t1 = time.perf_counter()
    predict(texts[:N_THR])
    thr = N_THR / (time.perf_counter() - t1)
    res = {"model": model_key + ("_int8" if quantize else ""), "load_time_s": load_s, "latency_ms_median": 1000 * float(np.median(lat)),
           "latency_ms_p95": 1000 * float(np.percentile(lat, 95)), "throughput_docs_per_s": thr,
           "serialized_size_mb": size_mb, "n_parameters": n_params, "rss_before_load_mb": rss0}
    if model_key in TRANSFORMERS:
        full = predict(texts)
        res["test_accuracy_end_to_end"] = float((full.argmax(1) == y).mean())
        from sklearn.metrics import f1_score

        res["test_macro_f1_end_to_end"] = float(f1_score(y, full.argmax(1), average="macro"))
        if not quantize:
            cached = load_probs(model_key, "test")
            res["agreement_with_cached_pipeline"] = float((full.argmax(1) == cached.argmax(1)).mean())
            res["max_abs_prob_diff_vs_cached"] = float(np.abs(full - cached).max())
        else:
            np.save(RESULTS / "preds" / model_key / "test_int8.npy", full.astype(np.float32))
    res["peak_memory_mb"] = peak_mem_mb()
    return res


def main(models: list[str], quantize_best: str | None) -> None:
    out_path = RESULTS / "efficiency.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = [(m, False) for m in models] + ([(quantize_best, True)] if quantize_best else [])
    for m, q in jobs:
        args = [sys.executable, "-m", "src.efficiency", "--one", m] + (["--int8"] if q else [])
        proc = subprocess.run(args, capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print("FAILED", m, proc.stderr[-2000:])
            continue
        r = json.loads(line[-1][7:])
        results[r["model"]] = r
        print(r, flush=True)
        save_json(results, out_path)


if __name__ == "__main__":
    if "--one" in sys.argv:
        key = sys.argv[sys.argv.index("--one") + 1]
        print("RESULT " + json.dumps(measure(key, quantize="--int8" in sys.argv)), flush=True)
    else:
        q = sys.argv[sys.argv.index("--quantize") + 1] if "--quantize" in sys.argv else None
        ms = [a for a in sys.argv[1:] if not a.startswith("--") and a != q]
        main(ms or ["logreg", "svm", "rf"] + TRANSFORMERS, q)
