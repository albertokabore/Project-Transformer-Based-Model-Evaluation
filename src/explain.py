"""Interpretability: SHAP (global + local), LIME (local) and attention visualisation.

* TF-IDF + Logistic Regression: exact linear SHAP (``shap.LinearExplainer``, interventional,
  background = training matrix) over the whole test set -> global importance per class.
* Best transformer: model-agnostic SHAP Partition explainer with a text masker and LIME, both on
  the same two test documents (one correct, one misclassified), plus last-layer attention.
  Because every explanation costs hundreds of forward passes on CPU, the transformer analysis is
  local only (n=2 documents); this is a stated limitation.
* Attention is shown for completeness but is *not* treated as an explanation (Jain & Wallace, 2019).
"""
from __future__ import annotations

import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import COLORS, INK, INK2, GRID  # noqa: F401  (also applies rcParams)
from .common import FIGS, MODEL_NAMES, MODELS, RESULTS, TABLES, load_probs, save_json
from .data import load_splits

POS, NEG = "#2a78d6", "#eb6834"  # diverging poles (supports / opposes the class)


def lr_shap(s, labels):
    import shap

    model = joblib.load(MODELS / "logreg.joblib")
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    Xtr = vec.transform(s.train.text)
    Xte = vec.transform(s.test.text)
    feats = np.array(vec.get_feature_names_out())
    expl = shap.LinearExplainer(clf, shap.maskers.Independent(Xtr, max_samples=500))
    sv = np.asarray(expl.shap_values(Xte))  # (n, features, classes) in recent shap
    if sv.shape[0] == len(labels) and sv.shape[1] == Xte.shape[0]:
        sv = np.transpose(sv, (1, 2, 0))
    rows = []
    for c, lab in enumerate(labels):
        imp = np.abs(sv[:, :, c]).mean(0)
        signed = sv[:, :, c].mean(0)
        for j in np.argsort(-imp)[:10]:
            rows.append({"class": lab, "feature": feats[j], "mean |SHAP|": imp[j], "mean SHAP": signed[j]})
    top = pd.DataFrame(rows)
    top.to_csv(TABLES / "shap_logreg_top_features.csv", index=False)

    # Global importance (mean |SHAP| over docs and classes)
    g = np.abs(sv).mean((0, 2))
    order = np.argsort(-g)[:20][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(feats[order], g[order], color=POS, height=0.7)
    axes[0].set(title="LR: global SHAP importance (top 20 n-grams)", xlabel="mean |SHAP| (class logits), all classes")
    axes[0].grid(axis="y", visible=False)
    # Beeswarm-style plot for one class
    c = labels.index("Cardiovascular / Pulmonary")
    imp = np.abs(sv[:, :, c]).mean(0)
    idx = np.argsort(-imp)[:12][::-1]
    Xd = Xte.toarray()
    rng = np.random.default_rng(0)
    for k, j in enumerate(idx):
        v = sv[:, j, c]
        val = Xd[:, j]
        colr = np.where(val > 0, POS, "#b7b6b0")
        axes[1].scatter(v, k + rng.uniform(-0.25, 0.25, len(v)), s=8, c=colr, alpha=0.7, linewidths=0)
    axes[1].set_yticks(range(len(idx)), feats[idx])
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set(title="SHAP values for class 'Cardiovascular / Pulmonary'", xlabel="SHAP value (class logit)")
    axes[1].scatter([], [], c=POS, s=12, label="n-gram present in note")
    axes[1].scatter([], [], c="#b7b6b0", s=12, label="n-gram absent")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "shap_logreg.png")
    plt.close(fig)
    return model


def pick_docs(s, labels, best_tf):
    y = s.test.label.map({l: i for i, l in enumerate(labels)}).values
    p_tf = load_probs(best_tf, "test").argmax(1)
    p_lr = load_probs("logreg", "test").argmax(1)
    short = s.test.n_words.values < 400
    right = np.where((p_tf == y) & (p_lr == y) & short)[0]
    wrong = np.where((p_tf != y) & short)[0]
    return [int(right[0]), int(wrong[0])] if len(wrong) else [int(right[0]), int(right[1])]


def unique_labels(names):
    """Keep repeated tokens as separate bars (matplotlib merges identical category labels)."""
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def bar_weights(ax, pairs, title):
    pairs = sorted(pairs, key=lambda x: abs(x[1]))[-12:]
    pairs = list(zip(unique_labels([p[0] for p in pairs]), [p[1] for p in pairs]))
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=[POS if p[1] > 0 else NEG for p in pairs], height=0.7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_title(title, fontsize=8.5)


def redraw(best_tf: str = "pubmedbert") -> None:
    """Rebuild local_explanations.png from the saved SHAP values and LIME weights (no model calls)."""
    meta = __import__("json").loads((RESULTS / "local_explanations.json").read_text())
    labels = load_splits().labels
    fig, axes = plt.subplots(len(meta), 3, figsize=(15, 4.3 * len(meta)))
    for r, m in enumerate(meta):
        c = labels.index(m["pred_transformer"])
        z = np.load(RESULTS / f"shap_{best_tf}_doc{m['test_index']}.npz", allow_pickle=True)
        occ = [(str(t).strip(), float(v)) for t, v in zip(z["tokens"], z["values"][:, c]) if str(t).strip()]
        bar_weights(axes[r, 0], occ, f"SHAP (Partition, 500 evals), {MODEL_NAMES[best_tf]}\nclass '{labels[c]}', per token occurrence")
        bar_weights(axes[r, 1], [tuple(x) for x in m["lime_tf"]], f"LIME, {MODEL_NAMES[best_tf]}\nclass '{labels[c]}'")
        bar_weights(axes[r, 2], [tuple(x) for x in m["lime_lr"]], f"LIME, {MODEL_NAMES['logreg']}\nclass '{labels[c]}'")
        axes[r, 0].set_ylabel(f"Test doc #{m['test_index']}\ntrue: {m['true']}\npred: {labels[c]} (p={m['p']:.2f})", fontsize=8, color=INK)
    fig.suptitle("Local explanations (blue = supports the predicted class, orange = opposes)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "local_explanations.png")
    plt.close(fig)


def local_explanations(s, labels, best_tf):
    import shap
    import torch
    from lime.lime_text import LimeTextExplainer

    from .transformer import MAX_LEN, FullClassifier

    torch.set_num_threads(8)
    docs = pick_docs(s, labels, best_tf)
    tf = FullClassifier(best_tf, len(labels))
    lr = joblib.load(MODELS / "logreg.joblib")
    lime = LimeTextExplainer(class_names=labels, random_state=0)
    masker = shap.maskers.Text(tf.tokenizer)
    sh = shap.Explainer(lambda t: tf.predict_proba(list(t), batch=16), masker, output_names=labels)
    y = s.test.label.values
    fig, axes = plt.subplots(len(docs), 3, figsize=(15, 4.3 * len(docs)))
    meta = []
    for r, i in enumerate(docs):
        text = s.test.text.iloc[i]
        p_tf = tf.predict_proba([text])[0]
        c = int(p_tf.argmax())
        # truncate the text the explainers see to the model's 256-token window, so that LIME/SHAP
        # do not spend samples on words the model never reads
        enc = tf.tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        text_w = text[: max(e for _, e in enc["offset_mapping"])]
        ex_tf = lime.explain_instance(text_w, lambda t: tf.predict_proba(list(t), batch=16), labels=[c], num_features=12, num_samples=300)
        ex_lr = lime.explain_instance(text, lr.predict_proba, labels=[c], num_features=12, num_samples=1000)
        sv = sh([text_w], max_evals=500, batch_size=16)
        toks, vals = np.array(sv.data[0]), sv.values[0][:, c]
        np.savez(RESULTS / f"shap_{best_tf}_doc{i}.npz", tokens=toks, values=sv.values[0], base=sv.base_values[0])
        # individual token occurrences (no summing over repeats, which would inflate frequent tokens)
        occ = [(t.strip(), float(v)) for t, v in zip(toks, vals) if t.strip()]
        bar_weights(axes[r, 0], occ, f"SHAP (Partition, 500 evals), {MODEL_NAMES[best_tf]}\nclass '{labels[c]}', per token occurrence")
        bar_weights(axes[r, 1], ex_tf.as_list(label=c), f"LIME, {MODEL_NAMES[best_tf]}\nclass '{labels[c]}'")
        bar_weights(axes[r, 2], ex_lr.as_list(label=c), f"LIME, {MODEL_NAMES['logreg']}\nclass '{labels[c]}'")
        axes[r, 0].set_ylabel(f"Test doc #{i}\ntrue: {y[i]}\npred: {labels[c]} (p={p_tf[c]:.2f})", fontsize=8, color=INK)
        top_tf = {w for w, _ in ex_tf.as_list(label=c)}
        top_lr = {w for w, _ in ex_lr.as_list(label=c)}
        p_lr = lr.predict_proba([text])[0]
        meta.append({"test_index": i, "true": y[i], "pred_transformer": labels[c], "p": float(p_tf[c]),
                     "pred_logreg": labels[int(p_lr.argmax())], "p_logreg": float(p_lr.max()),
                     "lime_tf": ex_tf.as_list(label=c), "lime_lr": ex_lr.as_list(label=c),
                     "lime_top12_overlap_tf_vs_lr": len(top_tf & top_lr), "text_excerpt": text_w[:400]})
        print("explained doc", i, flush=True)
    fig.suptitle("Local explanations (blue = supports the predicted class, orange = opposes)", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "local_explanations.png")
    plt.close(fig)

    # attention: last layer, averaged over heads; attention *received* by each token
    i = docs[0]
    enc = tf.tokenizer(s.test.text.iloc[i], truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        _, att = tf(enc["input_ids"], enc["attention_mask"], output_attentions=True)
    a = att[-1][0].mean(0)  # (L, L)
    recv = a.mean(0).numpy()
    toks = tf.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    special = set(tf.tokenizer.all_special_tokens)
    cand = [(t, v) for t, v in zip(toks, recv) if t not in special and len(t.strip("#Ġ")) > 2]
    cand = sorted(cand, key=lambda x: -x[1])[:20][::-1]
    n = 40
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].barh(unique_labels([t.strip("Ġ") for t, _ in cand]), [v for _, v in cand], color=POS, height=0.7)
    axes[0].grid(axis="y", visible=False)
    axes[0].set(title=f"Attention received (last layer, head-avg)\n{MODEL_NAMES[best_tf]}, test doc #{i}", xlabel="mean attention weight")
    from .analysis import SEQ

    axes[1].imshow(a[:n, :n].numpy(), cmap=SEQ)
    lbl = [t.strip("Ġ") for t in toks[:n]]
    axes[1].set_xticks(range(n), lbl, rotation=90, fontsize=5.5)
    axes[1].set_yticks(range(n), lbl, fontsize=5.5)
    axes[1].grid(False)
    axes[1].set_title(f"Last-layer attention map, first {n} tokens")
    fig.tight_layout()
    fig.savefig(FIGS / "attention.png")
    plt.close(fig)
    save_json(meta, RESULTS / "local_explanations.json")


if __name__ == "__main__":
    s = load_splits()
    labels = s.labels
    if "--transformer" not in sys.argv:
        lr_shap(s, labels)
        print("LR SHAP done")
    if "--lr-only" not in sys.argv:
        best_tf = sys.argv[sys.argv.index("--transformer") + 1] if "--transformer" in sys.argv else "clinicalbert"
        local_explanations(s, labels, best_tf)
