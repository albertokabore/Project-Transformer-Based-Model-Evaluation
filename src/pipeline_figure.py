"""Figure 1: evaluation pipeline diagram (matplotlib, no data)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import FIGS

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#c3c2b7"
FILL = {"data": "#e6f0fc", "model": "#fdeee6", "eval": "#e3f5ee"}

STAGES = [
    ("data", "MTSamples", ["4,999 rows", "40 category labels"], 0.09),
    ("data", "Audit & labels", ["group duplicates", "(2,357 unique texts)", "exclude 10 labels", "by analyst policy",
                                "keep single-specialty", "documents (1,745)"], 0.275),
    ("data", "Preprocessing", ["PHI scrub", "normalisation and", "abbreviation expansion", "stratified 70/10/20 split",
                               "11 classes (1,466 docs)", "OOD set (279 docs)"], 0.46),
    ("model", "Models", ["TF-IDF + LR / SVM / RF", "(5-fold CV grid search)", "BioBERT, ClinicalBERT,",
                         "PubMedBERT, RoBERTa", "(top-2-layer fine-tuning,", "early stop, grad clip)"], 0.645),
    ("eval", "Evaluation", ["metrics + bootstrap CI", "McNemar, 5x2cv, corr. t", "robustness and OOD",
                            "fairness, efficiency", "SHAP, LIME, attention"], 0.83),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.168, 0.94
    for i, (kind, title, lines, x) in enumerate(STAGES):
        ax.add_patch(FancyBboxPatch((x - w / 2, 0.5 - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc=FILL[kind], ec=EDGE, lw=1))
        ax.text(x, 0.5 + h / 2 - 0.05, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x, 0.5 + h / 2 - 0.2, "\n".join(lines), ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.4)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w / 2 + 0.002, 0.5), (STAGES[i + 1][3] - w / 2 - 0.002, 0.5),
                                         arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1))
    ax.text(0.965, 0.5, "MLflow\ntracking\n+ saved\npredictions\n-> report", ha="center", va="center", fontsize=7.6, color=INK2)
    fig.savefig(FIGS / "pipeline.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
