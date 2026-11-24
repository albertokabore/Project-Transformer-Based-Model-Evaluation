"""Audit the assignment's saved evidence without repeating model training.

Run with ``python -m src.review``. Checks are independent of the table-generation
functions where practical. This verifies artifact consistency, not clinical validity.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .data import build_dataset, load_splits

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    def close(name, actual, expected):
        check(name, np.allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True))

    s, rebuilt = load_splits(), build_dataset(save=False)
    # JSON serializes integer mapping keys as strings; normalize that representation.
    check("raw-data audit reproduces saved dataset statistics", s.stats == json.loads(json.dumps(rebuilt.stats)))
    for split in ("train", "val", "test", "ood"):
        check(f"raw data reproduces {split} rows and order", getattr(s, split).equals(getattr(rebuilt, split)))
    for a, b in itertools.combinations(("train", "val", "test", "ood"), 2):
        check(f"no exact normalized text overlap: {a}/{b}", not (set(getattr(s, a).text) & set(getattr(s, b).text)))

    lab = {v: i for i, v in enumerate(s.labels)}
    y, yv = s.test.label.map(lab).to_numpy(), s.val.label.map(lab).to_numpy()
    yb = label_binarize(y, classes=range(len(lab)))
    tables = ROOT / "results" / "tables"
    perf, rob = pd.read_csv(tables / "performance.csv"), pd.read_csv(tables / "robustness.csv")
    probs = {}
    val_scores = {}
    conditions = ("test", "val", "ood", "typo_10", "typo_25", "word_drop_20", "keyword_mask", "gender_swap")
    for row in perf.to_dict("records"):
        model = row["model"]
        probs[model] = {}
        for split in conditions if model != "logreg_trunc256" else ("test", "val"):
            p = np.load(ROOT / "results" / "preds" / model / f"{split}.npy")
            n = len(s.val) if split == "val" else len(s.ood) if split == "ood" else len(y)
            check(f"{model}/{split}: valid probability shape and values", p.shape == (n, len(lab)) and np.isfinite(p).all()
                  and (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=1e-5))
            probs[model][split] = p
        p = probs[model]["test"]
        pred = p.argmax(1)
        pr, re, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        metrics = {"Accuracy": accuracy_score(y, pred), "Macro-P": pr, "Macro-R": re, "Macro-F1": f,
                   "Weighted-F1": f1_score(y, pred, average="weighted"),
                   "ROC-AUC (macro OvR)": roc_auc_score(yb, p, average="macro"),
                   "ROC-AUC (micro)": roc_auc_score(yb, p, average="micro")}
        conf = p.max(1)
        edges = np.linspace(0, 1, 16)
        metrics["ECE"] = sum(mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
                             for lo, hi in zip(edges[:-1], edges[1:])
                             if (mask := ((conf > lo) & (conf <= hi))).any())
        for name, value in metrics.items():
            close(f"{model}: {name}", value, row[name])
        val_scores[model] = f1_score(yv, probs[model]["val"].argmax(1), average="macro")
        if model == "logreg_trunc256":
            continue
        r = rob.set_index("model").loc[model]
        for cond in ("typo_10", "typo_25", "word_drop_20", "keyword_mask"):
            close(f"{model}: robustness/{cond}", f1_score(y, probs[model][cond].argmax(1), average="macro"), r[cond])
        ood_conf = probs[model]["ood"].max(1)
        close(f"{model}: OOD AUROC", roc_auc_score(np.r_[np.ones(len(y)), np.zeros(len(s.ood))],
                                                               np.r_[conf, ood_conf]), r["OOD AUROC (MSP)"])
        close(f"{model}: OOD FPR", (ood_conf >= np.percentile(conf, 5)).mean(), r["OOD FPR@95TPR"])

    main_models = [m for m in perf.model if m != "logreg_trunc256"]
    summary = json.loads((ROOT / "results" / "analysis_summary.json").read_text())
    best = max(main_models, key=val_scores.get)
    check("reported winner matches validation-only selection", best == summary["best_overall"])
    vtable = pd.read_csv(tables / "validation_selection.csv", index_col=0).iloc[:, 0]
    for r in perf[perf.model.isin(main_models)].to_dict("records"):
        close(f"{r['model']}: validation table", val_scores[r["model"]], vtable[r["Model"]])
    for model in ("biobert", "clinicalbert", "pubmedbert", "roberta"):
        info = json.loads((ROOT / "results" / f"transformer_{model}.json").read_text())
        close(f"{model}: logged selected validation F1", info["best_val_macro_f1"], val_scores[model])
        grid_best = max(info["grid"], key=lambda g: g["best_val_macro_f1"])
        check(f"{model}: selected learning rate and epoch match training histories",
              grid_best["lr"] == info["best_lr"] and grid_best["best_epoch"] == info["best_epoch"]
              and all(np.isclose(g["best_val_macro_f1"], max(h["val_macro_f1"] for h in g["history"])) for g in info["grid"]))

    mc = pd.read_csv(tables / "mcnemar.csv")
    pvals = []
    for r in mc.to_dict("records"):
        a = probs[r["A"]]["test"].argmax(1) == y
        b = probs[r["B"]]["test"].argmax(1) == y
        ab, ba = int((a & ~b).sum()), int((~a & b).sum())
        exact = ab + ba < 25
        result = mcnemar([[0, ab], [ba, 0]], exact=exact, correction=not exact)
        close(f"McNemar/{r['A']}/{r['B']}: discordant counts", [ab, ba], [r["A right/B wrong"], r["A wrong/B right"]])
        close(f"McNemar/{r['A']}/{r['B']}: p", result.pvalue, r["p"])
        pvals.append(result.pvalue)
    close("McNemar Holm adjustment", multipletests(pvals, method="holm")[1], mc["p (Holm)"])

    cv = json.loads((ROOT / "results" / "cv_scores.json").read_text())
    cvt, cvs = pd.read_csv(tables / "cv_tests.csv"), pd.read_csv(tables / "cv_summary.csv")
    for r in cvs.to_dict("records"):
        score = cv[r["model"]]
        for key, arr in (("5x2cv", score["5x2"]), ("10-fold", score["10fold"])):
            close(f"CV/{r['model']}/{key}/mean", np.mean(arr), r[f"{key} mean"])
            close(f"CV/{r['model']}/{key}/sd", np.std(arr, ddof=1), r[f"{key} sd"])
    for r in cvt.to_dict("records"):
        a, b = cv[r["A"]], cv[r["B"]]
        d = np.array(a["5x2"]) - np.array(b["5x2"])
        var = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
        t = d[0, 0] / np.sqrt(var.mean())
        close(f"5x2cv/{r['A']}/{r['B']}", [t, 2 * stats.t.sf(abs(t), 5)], [r["5x2cv t"], r["5x2cv p"]])
        dd = np.array(a["10fold"]) - np.array(b["10fold"])
        tc = dd.mean() / np.sqrt((1 / 10 + 1 / 9) * dd.var(ddof=1))
        close(f"corrected CV/{r['A']}/{r['B']}", [tc, 2 * stats.t.sf(abs(tc), 9)],
              [r["corrected t (10-fold)"], r["corrected p (10-fold)"]])
    for col in ("5x2cv p", "corrected p (10-fold)"):
        dest = "5x2cv p (Holm)" if col.startswith("5") else "corrected p (Holm)"
        close(f"CV/{dest}", multipletests(cvt[col], method="holm")[1], cvt[dest])

    fairness = pd.read_csv(tables / "fairness.csv")
    for r in fairness.to_dict("records"):
        g = s.test[r["attribute"]].to_numpy()
        ga, gb = ("female", "male") if r["attribute"] == "sex" else ("18-64", "65+")
        pred = probs[r["model"]]["test"].argmax(1)
        a, b = g == ga, g == gb
        close(f"fairness/{r['model']}/{r['attribute']}: group accuracy", [accuracy_score(y[a], pred[a]), accuracy_score(y[b], pred[b])],
              [r["acc A"], r["acc B"]])
        dp = max(abs((pred[a] == c).mean() - (pred[b] == c).mean()) for c in range(len(lab)))
        close(f"fairness/{r['model']}/{r['attribute']}: DP", dp, r["DP diff (max over classes)"])
        gaps = [abs((pred[ia] == c).mean() - (pred[ib] == c).mean()) for c in range(len(lab))
                if (ia := (a & (y == c))).sum() >= 5 and (ib := (b & (y == c))).sum() >= 5]
        close(f"fairness/{r['model']}/{r['attribute']}: EO", [np.mean(gaps), max(gaps), len(gaps)],
              [r["EO gap mean |dTPR|"], r["EO gap max |dTPR|"], r["EO classes evaluated"]])
    for r in pd.read_csv(tables / "counterfactual_gender_swap.csv").to_dict("records"):
        p = probs[r["model"]]
        flip = p["test"].argmax(1) != p["gender_swap"].argmax(1)
        close(f"gender swap/{r['model']}", [flip.mean(), flip.sum()], [r["flip rate (all)"], r["flips (n)"]])
    efficiency = json.loads((ROOT / "results" / "efficiency.json").read_text())
    pi = np.load(ROOT / "results" / "preds" / "pubmedbert" / "test_int8.npy")
    close("quantized PubMedBERT F1", f1_score(y, pi.argmax(1), average="macro"), efficiency["pubmedbert_int8"]["test_macro_f1_end_to_end"])
    # Verify the serialized classical models still reproduce the saved probabilities.
    import joblib
    for m in ("logreg", "svm", "rf"):
        model = joblib.load(ROOT / "models" / f"{m}.joblib")
        close(f"{m}: model reproduces saved test probabilities", model.predict_proba(s.test.text.tolist()), probs[m]["test"])

    db = sqlite3.connect(f"file:{(ROOT / 'mlflow.db').as_posix()}?mode=ro", uri=True)
    tracking = {"run_status_counts": dict(db.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall()),
                "metric_records": db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]}
    db.close()
    hashes = {}
    for path in [ROOT / "data/raw/mtsamples.csv", ROOT / "data/processed/test.csv"] + sorted((ROOT / "results/preds").glob("*/*.npy")):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"scope": "Rebuilt splits; recomputed saved-prediction metrics, tests and subgroup results; verified classical model predictions. No transformer retraining or timing rerun.",
              "checks": checks, "passed": sum(x["passed"] for x in checks), "failed": sum(not x["passed"] for x in checks),
              "validation_selected_model": best, "mlflow": tracking, "sha256": hashes}
    target = ROOT / "results/review_audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "failed", "validation_selected_model", "mlflow")}, indent=2))
    for item in checks:
        if not item["passed"]:
            print("FAILED:", item["check"])
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
