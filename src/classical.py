"""Traditional ML baselines: TF-IDF + {Logistic Regression, Linear SVM, Random Forest}.

Hyper-parameters are selected by 5-fold stratified cross-validated grid search on the training
split (macro-F1). Class imbalance is handled with class weighting. All runs are logged to MLflow.
Outputs: class-probability arrays for val/test/OOD/perturbed test sets, fitted pipelines, the
keyword list used for the keyword-masking attack, and the 256-token truncation ablation.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .common import MODELS, ROOT, mlflow_setup, save_json, save_probs, RESULTS
from .data import SEED, load_splits
from .perturb import build_perturbations

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
TRUNC_TOKENIZER = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 256


def tfidf(max_features: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, max_features=max_features,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")


def specs():
    return {
        "logreg": (
            Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [1, 10, 100, 1000]},
        ),
        "svm": (
            Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(class_weight="balanced", max_iter=20000))]),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 0.5, 1, 5]},
        ),
        "rf": (
            Pipeline([("tfidf", tfidf(max_features=20000)),
                      ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
            {"clf__n_estimators": [300, 600], "clf__max_features": ["sqrt", 0.05], "clf__min_samples_leaf": [1, 2]},
        ),
    }


def with_proba(name: str, pipe: Pipeline, X, y) -> Pipeline:
    """LinearSVC has no predict_proba: wrap the tuned classifier in Platt (sigmoid) calibration."""
    if name != "svm":
        return pipe
    cal = Pipeline([("tfidf", pipe.named_steps["tfidf"]),
                    ("clf", CalibratedClassifierCV(pipe.named_steps["clf"], method="sigmoid", cv=CV))])
    return cal.fit(X, y)


def truncate_to_tokens(texts: list[str], max_len: int = MAX_LEN) -> list[str]:
    """Cut each text where the transformer tokenizer would truncate it (max_len incl. [CLS]/[SEP])."""
    from .transformer import get_tokenizer

    tok = get_tokenizer("clinicalbert")  # same (cased) tokenisation the transformer receives
    out = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_offsets_mapping=True, add_special_tokens=True)
        ends = [e for (s, e) in enc["offset_mapping"] if e > 0]
        out.append(t[: ends[-1]] if ends else t)
    return out


def latency(pipe, texts: list[str], n: int = 50) -> dict:
    single = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        pipe.predict_proba([t])
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    pipe.predict_proba(texts)
    batch = time.perf_counter() - t0
    return {"latency_ms_median": 1000 * float(np.median(single)), "latency_ms_p95": 1000 * float(np.percentile(single, 95)),
            "throughput_docs_per_s": len(texts) / batch}


def main() -> None:
    s = load_splits()
    lab2id = {l: i for i, l in enumerate(s.labels)}
    Xtr, ytr = s.train.text.tolist(), s.train.label.map(lab2id).values
    Xva, yva = s.val.text.tolist(), s.val.label.map(lab2id).values
    Xte = s.test.text.tolist()
    mlflow = mlflow_setup()
    report = {}

    perturbed = None
    for name, (pipe, grid) in specs().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=CV, n_jobs=1 if name == "rf" else -1, refit=True)
        gs.fit(Xtr, ytr)
        tune_s = time.time() - t0
        model = with_proba(name, gs.best_estimator_, Xtr, ytr)
        val_f1 = f1_score(yva, model.predict_proba(Xva).argmax(1), average="macro")
        joblib.dump(model, MODELS / f"{name}.joblib", compress=3)
        size_mb = (MODELS / f"{name}.joblib").stat().st_size / 2**20

        if name == "logreg":
            # keyword list for the masking attack: top-15 positive unigram coefficients per class
            vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
            feats = np.array(vec.get_feature_names_out())
            uni = np.array([" " not in f for f in feats])
            kw = set()
            for c in range(clf.coef_.shape[0]):
                order = np.argsort(-clf.coef_[c] * uni)
                kw.update(feats[order[:15]].tolist())
            save_json(sorted(kw), RESULTS / "keyword_mask_terms.json")
            perturbed = build_perturbations(Xte, kw)
            pd.DataFrame(perturbed).to_csv(ROOT / "data" / "processed" / "test_perturbed.csv", index=False)

        for split, texts in [("val", Xva), ("test", Xte), ("ood", s.ood.text.tolist())] + list(perturbed.items()):
            save_probs(name, split, model.predict_proba(texts))
        eff = latency(model, Xte)
        report[name] = {"best_params": {k: str(v) for k, v in gs.best_params_.items()}, "cv_macro_f1": gs.best_score_,
                        "cv_macro_f1_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
                        "val_macro_f1": val_f1, "tuning_time_s": tune_s, "refit_time_s": gs.refit_time_,
                        "model_size_mb": size_mb, "n_features": len(model.named_steps["tfidf"].vocabulary_), **eff}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, **report[name]["best_params"]})
            mlflow.log_metrics({k: v for k, v in report[name].items() if isinstance(v, (int, float))})
            mlflow.log_dict({"cv_results": pd.DataFrame(gs.cv_results_).astype(str).to_dict()}, "cv_results.json")
        print(name, report[name], flush=True)

    save_json(report, RESULTS / "classical_training.json")
    ablation_truncated(Xtr, ytr, Xva, Xte, mlflow)


def ablation_truncated(Xtr, ytr, Xva, Xte, mlflow) -> None:
    """Ablation: give LR exactly the input window the transformers see (first 256 word-pieces)."""
    from .common import load_json

    pipe, grid = specs()["logreg"]
    tr_tr, va_tr, te_tr = truncate_to_tokens(Xtr), truncate_to_tokens(Xva), truncate_to_tokens(Xte)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=CV, n_jobs=-1).fit(tr_tr, ytr)
    m = gs.best_estimator_
    save_probs("logreg_trunc256", "val", m.predict_proba(va_tr))
    save_probs("logreg_trunc256", "test", m.predict_proba(te_tr))
    report = load_json(RESULTS / "classical_training.json")
    report["logreg_trunc256"] = {"best_params": {k: str(v) for k, v in gs.best_params_.items()}, "cv_macro_f1": gs.best_score_,
                                 "cv_macro_f1_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
                                 "mean_words_kept": float(np.mean([len(t.split()) for t in te_tr]))}
    with mlflow.start_run(run_name="logreg_trunc256"):
        mlflow.log_params(report["logreg_trunc256"]["best_params"])
        mlflow.log_metric("cv_macro_f1", gs.best_score_)
    save_json(report, RESULTS / "classical_training.json")
    print(report["logreg_trunc256"])


if __name__ == "__main__":
    import sys

    if "--ablation-only" in sys.argv:
        _s = load_splits()
        _l = {l: i for i, l in enumerate(_s.labels)}
        ablation_truncated(_s.train.text.tolist(), _s.train.label.map(_l).values, _s.val.text.tolist(),
                           _s.test.text.tolist(), mlflow_setup())
    else:
        main()
