"""Evaluation & statistical analysis over saved prediction files.

Every number in the report is produced here from ``results/preds/<model>/<split>.npy``.
"""
from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .common import FIGS, MODEL_NAMES, PREDS, RESULTS, TABLES, load_json, load_probs, save_json
from .data import load_splits

MAIN = ["logreg", "svm", "rf", "clinicalbert", "pubmedbert", "biobert", "roberta"]
COLORS = {"logreg": "#2a78d6", "svm": "#eb6834", "rf": "#1baf7a", "clinicalbert": "#eda100", "pubmedbert": "#e87ba4",
          "biobert": "#008300", "roberta": "#6250d6", "logreg_trunc256": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
PERTURB = ["typo_10", "typo_25", "word_drop_20", "keyword_mask"]
B = 1000
RNG = np.random.default_rng(2024)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False, "axes.axisbelow": True})


def available(models):
    return [m for m in models if (PREDS / m / "test.npy").exists()]


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(e)


def boot_ci(y, pred, fn, idx):
    vals = np.array([fn(y[i], pred[i]) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def _literal(v: str):
    import ast

    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v  # plain strings such as "sqrt"


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


# ------------------------------------------------------------------------------------------
def performance(models, y, labels):
    C = len(labels)
    yb = label_binarize(y, classes=range(C))
    idx = [RNG.integers(0, len(y), len(y)) for _ in range(B)]
    rows, per_class = [], {}
    for m in models:
        p = load_probs(m, "test")
        pred = p.argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        acc_ci = boot_ci(y, pred, accuracy_score, idx)
        f1_ci = boot_ci(y, pred, macro_f1, idx)
        rows.append({"model": m, "Model": MODEL_NAMES[m], "Accuracy": accuracy_score(y, pred), "Acc 95% CI": acc_ci,
                     "Macro-P": pr, "Macro-R": rc, "Macro-F1": f1, "F1 95% CI": f1_ci,
                     "Weighted-F1": f1_score(y, pred, average="weighted"),
                     "ROC-AUC (macro OvR)": roc_auc_score(yb, p, average="macro"),
                     "ROC-AUC (micro)": roc_auc_score(yb, p, average="micro"), "ECE": ece(p, y)})
        per_class[MODEL_NAMES[m]] = f1_score(y, pred, average=None, labels=range(C), zero_division=0)
    df = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class, index=labels)
    pc["support"] = np.bincount(y, minlength=C)
    return df, pc, idx


def fig_confusion(models, y, labels):
    n = len(models)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.0 * rows))
    short = [l.split(" /")[0].split(" -")[0][:14] for l in labels]
    for ax, m in zip(axes.flat, models):
        cm = confusion_matrix(y, load_probs(m, "test").argmax(1), labels=range(len(labels)), normalize="true")
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
        for i, j in itertools.product(range(len(labels)), repeat=2):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}"[1:] if cm[i, j] < 1 else "1", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)), short, fontsize=6.5)
        ax.grid(False)
        ax.set_title(MODEL_NAMES[m], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Row-normalised confusion matrices on the held-out test set (n=%d)" % len(y), color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrices.png")
    plt.close(fig)


def fig_roc(models, y, labels, best):
    C = len(labels)
    yb = label_binarize(y, classes=range(C))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for m in models:
        p = load_probs(m, "test")
        fpr, tpr, _ = roc_curve(yb.ravel(), p.ravel())
        ax.plot(fpr, tpr, lw=2, color=COLORS[m], label=f"{MODEL_NAMES[m]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Micro-averaged ROC, all models")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    p = load_probs(best, "test")
    aucs = []
    for c in range(C):
        fpr, tpr, _ = roc_curve(yb[:, c], p[:, c])
        aucs.append((auc(fpr, tpr), c, fpr, tpr))
    aucs.sort()
    # highlight the 3 hardest classes; others recessive
    for k, (a, c, fpr, tpr) in enumerate(aucs):
        hard = k < 3
        ax.plot(fpr, tpr, lw=2 if hard else 1, color=["#e34948", "#eb6834", "#eda100"][k] if hard else "#b7b6b0",
                label=f"{labels[c]} (AUC={a:.3f})" if hard else None, zorder=3 if hard else 1)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"Per-class ROC, {MODEL_NAMES[best]} (3 lowest-AUC classes highlighted)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "roc_curves.png")
    plt.close(fig)


def fig_reliability(models, y):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    edges = np.linspace(0, 1, 11)
    for m in models:
        p = load_probs(m, "test")
        conf, ok = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (conf > lo) & (conf <= hi)
            if s.sum() >= 5:
                xs.append(conf[s].mean())
                ys.append(ok[s].mean())
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=COLORS[m], label=MODEL_NAMES[m])
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy", title="Reliability diagram (bins with n>=5)")
    ax.legend(fontsize=7)
    fig.savefig(FIGS / "reliability.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def mcnemar_tests(models, y):
    correct = {m: load_probs(m, "test").argmax(1) == y for m in models}
    rows = []
    for a, b in itertools.combinations(models, 2):
        n01 = int((correct[a] & ~correct[b]).sum())
        n10 = int((~correct[a] & correct[b]).sum())
        exact = (n01 + n10) < 25
        r = mcnemar([[0, n01], [n10, 0]], exact=exact, correction=not exact)
        rows.append({"A": a, "B": b, "A right/B wrong": n01, "A wrong/B right": n10, "test": "exact" if exact else "chi2 (cc)",
                     "statistic": float(r.statistic), "p": float(r.pvalue)})
    df = pd.DataFrame(rows)
    df["p (Holm)"] = multipletests(df.p, method="holm")[1]
    return df


def paired_bootstrap(models, y, best, idx):
    pb = load_probs(best, "test").argmax(1)
    rows = []
    for m in models:
        if m == best:
            continue
        pm = load_probs(m, "test").argmax(1)
        d = np.array([macro_f1(y[i], pb[i]) - macro_f1(y[i], pm[i]) for i in idx])
        rows.append({"reference": best, "vs": m, "delta macro-F1": macro_f1(y, pb) - macro_f1(y, pm),
                     "95% CI low": np.percentile(d, 2.5), "95% CI high": np.percentile(d, 97.5),
                     "P(delta<=0)": float((d <= 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
def cv_analysis(splits, labels):
    """Cross-validation on train+val (n=1172): classical pipelines (tuned params) and frozen
    transformer embeddings + LR probes. 5x2cv paired t-test (Dietterich, 1998) and 10-fold
    corrected resampled t-test (Nadeau & Bengio, 2003)."""
    import joblib
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .classical import specs
    from .common import CACHE, MODELS

    lab2id = {l: i for i, l in enumerate(labels)}
    pool = pd.concat([splits.train, splits.val], ignore_index=True)
    X = np.array(pool.text.tolist(), dtype=object)
    y = pool.label.map(lab2id).values
    tuned = load_json(RESULTS / "classical_training.json")
    estimators = {}
    for name, (pipe, _) in specs().items():
        p = clone(pipe)
        params = {k: _literal(v) for k, v in tuned[name]["best_params"].items()}
        p.set_params(**params)
        estimators[name] = ("text", p)
    for k in ["clinicalbert", "pubmedbert", "biobert", "roberta"]:
        if (CACHE / k / "train_pooled.npy").exists():
            emb = np.concatenate([np.load(CACHE / k / "train_pooled.npy"), np.load(CACHE / k / "val_pooled.npy")])
            estimators[f"{k}_frozen"] = (emb, make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000,
                                                                                                    class_weight="balanced")))

    def score(est, Xin, tr, te):
        e = clone(est[1])
        Xd = X if isinstance(est[0], str) else est[0]
        e.fit(Xd[tr], y[tr])
        return macro_f1(y[te], e.predict(Xd[te]))

    # 5x2 CV
    s52 = {n: np.zeros((5, 2)) for n in estimators}
    for r in range(5):
        skf = StratifiedKFold(2, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(skf.split(X, y)):
            for n, est in estimators.items():
                s52[n][r, f] = score(est, X, tr, te)
        print("5x2cv repetition", r + 1, "done", flush=True)
    # 10-fold
    s10 = {n: [] for n in estimators}
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y):
        for n, est in estimators.items():
            s10[n].append(score(est, X, tr, te))
    s10 = {n: np.array(v) for n, v in s10.items()}

    summary = pd.DataFrame({"model": list(estimators), "5x2cv mean": [s52[n].mean() for n in estimators],
                            "5x2cv sd": [s52[n].std(ddof=1) for n in estimators],
                            "10-fold mean": [s10[n].mean() for n in estimators], "10-fold sd": [s10[n].std(ddof=1) for n in estimators]})
    rows = []
    n_te, n_tr = len(y) / 10, len(y) * 9 / 10
    for a, b in itertools.combinations(estimators, 2):
        d = s52[a] - s52[b]
        var = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
        t52 = d[0, 0] / np.sqrt(var.mean())
        p52 = 2 * stats.t.sf(abs(t52), df=5)
        dd = s10[a] - s10[b]
        k = len(dd)
        tc = dd.mean() / np.sqrt((1 / k + n_te / n_tr) * dd.var(ddof=1)) if dd.var(ddof=1) > 0 else 0.0
        pc = 2 * stats.t.sf(abs(tc), df=k - 1)
        rows.append({"A": a, "B": b, "mean diff (10-fold)": dd.mean(), "5x2cv t": t52, "5x2cv p": p52,
                     "corrected t (10-fold)": tc, "corrected p (10-fold)": pc})
    tests = pd.DataFrame(rows)
    tests["5x2cv p (Holm)"] = multipletests(tests["5x2cv p"], method="holm")[1]
    tests["corrected p (Holm)"] = multipletests(tests["corrected p (10-fold)"], method="holm")[1]
    save_json({n: {"5x2": s52[n].tolist(), "10fold": s10[n].tolist()} for n in estimators}, RESULTS / "cv_scores.json")
    return summary, tests, s10


# ------------------------------------------------------------------------------------------
def robustness(models, y, y_ood_len):
    rows = []
    for m in models:
        clean = macro_f1(y, load_probs(m, "test").argmax(1))
        r = {"model": m, "Model": MODEL_NAMES[m], "clean": clean}
        for p in PERTURB:
            r[p] = macro_f1(y, load_probs(m, p).argmax(1))
        # OOD detection with maximum softmax probability (Hendrycks & Gimpel, 2017)
        msp_in, msp_out = load_probs(m, "test").max(1), load_probs(m, "ood").max(1)
        lab = np.r_[np.ones(len(msp_in)), np.zeros(len(msp_out))]
        sc = np.r_[msp_in, msp_out]
        r["OOD AUROC (MSP)"] = roc_auc_score(lab, sc)
        thr = np.percentile(msp_in, 5)  # keep 95% of in-distribution
        r["OOD FPR@95TPR"] = float((msp_out >= thr).mean())
        r["mean MSP in-dist"], r["mean MSP OOD"] = float(msp_in.mean()), float(msp_out.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def fig_robustness(rob):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    conds = ["clean"] + PERTURB
    names = {"clean": "Clean", "typo_10": "Typos 10%", "typo_25": "Typos 25%", "word_drop_20": "Word drop 20%",
             "keyword_mask": "Keyword mask"}
    w = 0.8 / len(rob)
    for i, (_, r) in enumerate(rob.iterrows()):
        ax.bar(np.arange(len(conds)) + (i - len(rob) / 2 + 0.5) * w, [r[c] for c in conds], width=w * 0.9,
               color=COLORS[r.model], label=r.Model)
    ax.set_xticks(range(len(conds)), [names[c] for c in conds])
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title("Robustness: macro-F1 under input perturbations")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIGS / "robustness.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def fairness(models, test, y, labels):
    C = len(labels)
    rows, eo_rows = [], []
    for attr, groups in [("sex", ["female", "male"]), ("age_group", ["18-64", "65+"])]:
        g = test[attr].values
        true_dp = max(abs((y[g == groups[0]] == c).mean() - (y[g == groups[1]] == c).mean()) for c in range(C))
        for m in models:
            pred = load_probs(m, "test").argmax(1)
            a, b, u = g == groups[0], g == groups[1], g == "unknown"
            r = {"attribute": attr, "model": m, "Model": MODEL_NAMES[m], "group A": f"{groups[0]} (n={a.sum()})",
                 "acc A": accuracy_score(y[a], pred[a]), "group B": f"{groups[1]} (n={b.sum()})",
                 "acc B": accuracy_score(y[b], pred[b]), "acc unknown": accuracy_score(y[u], pred[u]) if u.any() else np.nan,
                 "acc gap (A-B)": accuracy_score(y[a], pred[a]) - accuracy_score(y[b], pred[b])}
            ca, cb = int((pred[a] == y[a]).sum()), int((pred[b] == y[b]).sum())
            r["acc gap p (Fisher)"] = stats.fisher_exact([[ca, a.sum() - ca], [cb, b.sum() - cb]])[1]
            dp = [abs((pred[a] == c).mean() - (pred[b] == c).mean()) for c in range(C)]
            r["DP diff (max over classes)"] = max(dp)
            r["DP diff, ground truth"] = true_dp
            gaps = []
            for c in range(C):
                ia, ib = a & (y == c), b & (y == c)
                if ia.sum() >= 5 and ib.sum() >= 5:
                    gaps.append((labels[c], (pred[ia] == c).mean() - (pred[ib] == c).mean(), int(ia.sum()), int(ib.sum())))
            r["EO gap mean |dTPR|"] = float(np.mean([abs(x[1]) for x in gaps])) if gaps else np.nan
            r["EO gap max |dTPR|"] = float(np.max([abs(x[1]) for x in gaps])) if gaps else np.nan
            r["EO classes evaluated"] = len(gaps)
            rows.append(r)
            for lab, gap, na, nb in gaps:
                eo_rows.append({"attribute": attr, "model": m, "class": lab, f"TPR[{groups[0]}]-TPR[{groups[1]}]": gap,
                                "n_a": na, "n_b": nb})
    # counterfactual gender swap
    cf = []
    has_cue = (test.sex != "unknown").values
    for m in models:
        a, b = load_probs(m, "test"), load_probs(m, "gender_swap")
        flip = a.argmax(1) != b.argmax(1)
        cf.append({"model": m, "Model": MODEL_NAMES[m], "flip rate (all)": flip.mean(), "flips (n)": int(flip.sum()),
                   "flip rate (docs with sex cues)": flip[has_cue].mean(),
                   "mean |dP(true class)|": float(np.abs(a[np.arange(len(y)), y] - b[np.arange(len(y)), y]).mean()),
                   "macro-F1 swapped": macro_f1(y, b.argmax(1))})
    return pd.DataFrame(rows), pd.DataFrame(eo_rows), pd.DataFrame(cf)


# ------------------------------------------------------------------------------------------
def error_analysis(best, test, y, labels):
    p = load_probs(best, "test")
    pred = p.argmax(1)
    wrong = pred != y
    pairs = pd.Series([f"{labels[t]} -> {labels[q]}" for t, q in zip(y[wrong], pred[wrong])]).value_counts()
    conf_wrong = p.max(1)[wrong]
    by_len = pd.DataFrame({"n_words": test.n_words, "correct": ~wrong})
    by_len["length bin"] = pd.qcut(by_len.n_words, 4)
    lb = by_len.groupby("length bin", observed=True).correct.agg(["mean", "count"]).reset_index()
    lb["length bin"] = lb["length bin"].astype(str)
    return pairs, {"n_errors": int(wrong.sum()), "mean_conf_wrong": float(conf_wrong.mean()),
                   "mean_conf_right": float(p.max(1)[~wrong].mean()),
                   "errors_conf_gt_0.9": int((conf_wrong > 0.9).sum())}, lb


def fig_data_overview(s):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    allin = pd.concat([s.train.assign(split="train"), s.val.assign(split="val"), s.test.assign(split="test")])
    order = allin.label.value_counts().index
    left = np.zeros(len(order))
    for sp, col in [("train", "#2a78d6"), ("val", "#eb6834"), ("test", "#1baf7a")]:
        v = allin[allin.split == sp].label.value_counts().reindex(order).fillna(0).values
        axes[0].barh(order, v, left=left, color=col, label=sp, height=0.7, edgecolor="white", linewidth=1)
        left += v
    for i, t in enumerate(left):
        axes[0].text(t + 3, i, str(int(t)), va="center", fontsize=7, color=INK2)
    axes[0].invert_yaxis()
    axes[0].grid(axis="y", visible=False)
    axes[0].set(title="Unique documents per specialty (stratified 70/10/20 split)", xlabel="documents")
    axes[0].legend(fontsize=7, loc="lower right")
    bins = np.linspace(0, 2000, 41)
    axes[1].hist(allin.n_words.clip(upper=2000), bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.8, label="in-distribution")
    axes[1].hist(s.ood.n_words.clip(upper=2000), bins=bins, color="#eb6834", alpha=0.85, edgecolor="white", linewidth=0.8, label="OOD (unseen specialties)")
    axes[1].axvline(np.median(allin.n_words), color=INK, lw=1, ls="--")
    axes[1].text(np.median(allin.n_words) + 20, axes[1].get_ylim()[1] * 0.9, f"median {np.median(allin.n_words):.0f} words",
                 fontsize=7, color=INK)
    axes[1].set(title="Note length (words, clipped at 2,000)", xlabel="words per note", ylabel="documents")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "data_overview.png")
    plt.close(fig)


def fig_training_curves():
    keys = [k for k in ["clinicalbert", "pubmedbert", "biobert", "roberta"] if (RESULTS / f"transformer_{k}.json").exists()]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in keys:
        info = load_json(RESULTS / f"transformer_{k}.json")
        for g in info["grid"]:
            h = pd.DataFrame(g["history"])
            ls = "-" if g["lr"] == info["best_lr"] else ":"
            axes[0].plot(h.epoch, h.train_loss, ls=ls, lw=2, color=COLORS[k], marker="o", ms=3,
                         label=f"{MODEL_NAMES[k]}, lr={g['lr']:g}")
            axes[1].plot(h.epoch, h.val_macro_f1, ls=ls, lw=2, color=COLORS[k], marker="o", ms=3)
    axes[0].set(title="Training loss (class-weighted CE)", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation macro-F1 (early-stopping criterion)", xlabel="epoch", ylabel="macro-F1")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.text(0.5, -0.02, "solid = selected learning rate, dotted = other grid value", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves.png")
    plt.close(fig)


def architecture_table():
    """Table 1. Parameter counts are measured from the fitted / loaded models."""
    import joblib

    from .common import MODELS

    def n_linear(m):
        clf = m.named_steps["clf"]
        if hasattr(clf, "calibrated_classifiers_"):
            return sum(c.estimator.coef_.size + c.estimator.intercept_.size for c in clf.calibrated_classifiers_)
        return clf.coef_.size + clf.intercept_.size

    rows = []
    desc = {
        "logreg": ("Linear (multinomial)", "none (trained from scratch)", "TF-IDF unigrams, learned on train"),
        "svm": ("Linear max-margin + Platt calibration", "none", "TF-IDF unigrams, learned on train"),
        "rf": ("Bagged decision trees", "none", "TF-IDF uni+bigrams, top 20k"),
        "biobert": ("BERT-base encoder (cased)", "BERT init. + PubMed abstracts", "WordPiece, 28,996 (general BERT vocab)"),
        "clinicalbert": ("BERT-base encoder (cased)", "BioBERT init. + MIMIC-III clinical notes", "WordPiece, 28,996 (general BERT vocab)"),
        "pubmedbert": ("BERT-base encoder (uncased)", "from scratch: PubMed abstracts + PMC full text", "WordPiece, 30,522 (in-domain vocab)"),
        "roberta": ("RoBERTa-base encoder", "general domain (books, Wikipedia, news, web)", "byte-level BPE, 50,265"),
    }
    for m in ["logreg", "svm", "rf"]:
        if not (MODELS / f"{m}.joblib").exists():
            continue
        est = joblib.load(MODELS / f"{m}.joblib")
        if m == "rf":
            nodes = sum(t.tree_.node_count for t in est.named_steps["clf"].estimators_)
            size = f"{len(est.named_steps['clf'].estimators_)} trees, {nodes:,} nodes"
        else:
            size = f"{n_linear(est):,} weights"
        rows.append({"Model": MODEL_NAMES[m], "Architecture": desc[m][0], "Pre-training": desc[m][1],
                     "Features / vocabulary": desc[m][2], "Size": size, "Trainable in this study": "all", "Input": "full note"})
    for m in ["biobert", "clinicalbert", "pubmedbert", "roberta"]:
        f = RESULTS / f"transformer_{m}.json"
        if not f.exists():
            continue
        info = load_json(f)
        rows.append({"Model": MODEL_NAMES[m], "Architecture": desc[m][0] + ", 12 layers, 768-d, 12 heads",
                     "Pre-training": desc[m][1], "Features / vocabulary": desc[m][2],
                     "Size": f"{info['total_params'] / 1e6:.1f} M parameters",
                     "Trainable in this study": f"top 2 layers + head ({info['trainable_params'] / 1e6:.1f} M)",
                     "Input": "first 256 word-pieces"})
    pd.DataFrame(rows).to_csv(TABLES / "architectures.csv", index=False)


def efficiency_table():
    """Merge measured inference efficiency (efficiency.json) with measured training cost."""
    f = RESULTS / "efficiency.json"
    if not f.exists():
        return
    eff = load_json(f)
    ct = load_json(RESULTS / "classical_training.json")
    rows = []
    for key, r in eff.items():
        base = key.replace("_int8", "")
        if base in ct:
            train = ct[base]["tuning_time_s"]
            note = "grid search, 5-fold CV"
        else:
            tj = load_json(RESULTS / f"transformer_{base}.json")
            train = tj["cache_time_s"] + tj["train_time_s"]
            note = "state caching + 2 LR configs"
        rows.append({"model": key, "Model": MODEL_NAMES.get(base, base) + (" (int8)" if key.endswith("_int8") else ""),
                     "training/tuning time (min)": train / 60, "training note": note,
                     "latency median (ms)": r["latency_ms_median"], "latency p95 (ms)": r["latency_ms_p95"],
                     "throughput (docs/s)": r["throughput_docs_per_s"], "peak memory (MB)": r["peak_memory_mb"],
                     "serialized size (MB)": r["serialized_size_mb"],
                     # quantised Linear weights are packed and not reported by .parameters(): not comparable
                     "parameters (M)": (r["n_parameters"] / 1e6) if r.get("n_parameters") and not key.endswith("_int8") else np.nan,
                     "test macro-F1 end-to-end": r.get("test_macro_f1_end_to_end", np.nan),
                     "agreement w/ cached pipeline": r.get("agreement_with_cached_pipeline", np.nan)})
    pd.DataFrame(rows).to_csv(TABLES / "efficiency.csv", index=False)


def tradeoff():
    """Accuracy vs speed vs interpretability, from measured tables only."""
    if not (TABLES / "efficiency.csv").exists():
        return
    eff = pd.read_csv(TABLES / "efficiency.csv")
    perf = pd.read_csv(TABLES / "performance.csv")
    rob = pd.read_csv(TABLES / "robustness.csv")
    df = eff[~eff.model.str.endswith("_int8")].merge(perf[["model", "Macro-F1", "ECE"]], on="model")
    df = df.merge(rob[["model", "keyword_mask", "OOD AUROC (MSP)"]], on="model")
    interp = {"logreg": "intrinsic (exact linear SHAP)", "svm": "intrinsic (linear weights)", "rf": "post-hoc (tree SHAP possible)",
              "clinicalbert": "post-hoc only (LIME/SHAP, approx.)", "pubmedbert": "post-hoc only (LIME/SHAP, approx.)",
              "biobert": "post-hoc only (LIME/SHAP, approx.)", "roberta": "post-hoc only (LIME/SHAP, approx.)"}
    df["interpretability"] = df.model.map(interp)
    cols = ["Model", "Macro-F1", "ECE", "keyword_mask", "OOD AUROC (MSP)", "latency median (ms)", "throughput (docs/s)",
            "peak memory (MB)", "serialized size (MB)", "interpretability"]
    df[cols].rename(columns={"keyword_mask": "Macro-F1 under keyword mask"}).to_csv(TABLES / "tradeoff.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for _, r in df.iterrows():
        ax.scatter(r["latency median (ms)"], r["Macro-F1"], s=60, color=COLORS[r.model], zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(r.Model, (r["latency median (ms)"], r["Macro-F1"]), textcoords="offset points", xytext=(7, -3), fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set(xlabel="median single-note latency on CPU (ms, log scale)", ylabel="test macro-F1",
           title="Accuracy vs. inference latency (same CPU, end-to-end)")
    fig.savefig(FIGS / "tradeoff.png")
    plt.close(fig)


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main(run_cv: bool = True):
    s = load_splits()
    labels = s.labels
    lab2id = {l: i for i, l in enumerate(labels)}
    y = s.test.label.map(lab2id).values
    models = available(MAIN)
    extra = available(["logreg_trunc256"])
    fig_data_overview(s)
    fig_training_curves()
    architecture_table()
    perf, pc, idx = performance(models + extra, y, labels)
    # Select the reference using validation predictions. Artifact consistency alone cannot
    # establish that test data were never inspected during development; intervals are pointwise.
    y_val = s.val.label.map(lab2id).values
    val_f1 = {m: macro_f1(y_val, load_probs(m, "val").argmax(1)) for m in models}
    pd.Series(val_f1, name="validation macro-F1").rename(index=MODEL_NAMES).to_csv(TABLES / "validation_selection.csv")
    best = max(val_f1, key=val_f1.get)
    tfs = [m for m in models if m in ("clinicalbert", "pubmedbert", "biobert", "roberta")]
    best_tf = max(tfs, key=val_f1.get) if tfs else None
    out = perf.copy()
    out["Acc 95% CI"] = out["Acc 95% CI"].apply(fmt_ci)
    out["F1 95% CI"] = out["F1 95% CI"].apply(fmt_ci)
    out.to_csv(TABLES / "performance.csv", index=False)
    pc.to_csv(TABLES / "per_class_f1.csv")
    fig_confusion(models, y, labels)
    fig_roc(models, y, labels, best)
    fig_reliability(models, y)
    mc = mcnemar_tests(models, y)
    mc.to_csv(TABLES / "mcnemar.csv", index=False)
    pbs = paired_bootstrap(models, y, best, idx)
    pbs.to_csv(TABLES / "paired_bootstrap.csv", index=False)
    rob = robustness(models, y, None)
    rob.to_csv(TABLES / "robustness.csv", index=False)
    fig_robustness(rob)
    fair, eo, cf = fairness(models, s.test, y, labels)
    fair.to_csv(TABLES / "fairness.csv", index=False)
    eo.to_csv(TABLES / "equal_opportunity_by_class.csv", index=False)
    cf.to_csv(TABLES / "counterfactual_gender_swap.csv", index=False)
    pairs, err, lb = error_analysis(best, s.test, y, labels)
    pairs.to_csv(TABLES / f"error_pairs_{best}.csv")
    lb.to_csv(TABLES / f"accuracy_by_length_{best}.csv", index=False)
    summary = {"best_overall": best, "best_transformer": best_tf, "error_analysis": err}
    if best_tf and best_tf != best:
        pairs_tf, err_tf, lb_tf = error_analysis(best_tf, s.test, y, labels)
        pairs_tf.to_csv(TABLES / f"error_pairs_{best_tf}.csv")
        lb_tf.to_csv(TABLES / f"accuracy_by_length_{best_tf}.csv", index=False)
        summary["error_analysis_best_transformer"] = err_tf
    if run_cv:
        cvs, cvt, _ = cv_analysis(s, labels)
        cvs.to_csv(TABLES / "cv_summary.csv", index=False)
        cvt.to_csv(TABLES / "cv_tests.csv", index=False)
    efficiency_table()
    tradeoff()
    save_json(summary, RESULTS / "analysis_summary.json")
    print(out.to_string())
    print(summary)


if __name__ == "__main__":
    import sys

    main(run_cv="--no-cv" not in sys.argv)
