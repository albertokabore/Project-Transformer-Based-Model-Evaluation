"""Generate an evidence-reuse notebook with the reviewed report's eight-step narrative.

Run report/build_report.py first. Executing this notebook rebuilds and checks data,
recomputes metrics from saved predictions, and displays the existing experiment evidence.
It does not retrain transformers or repeat hardware timing.
"""
from pathlib import Path
import importlib.util
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
target = Path(__file__).with_name("Transformer_Model_Evaluation.ipynb")
previous = nbf.read(target, as_version=4) if target.exists() else None
previous_code = {cell.source: cell for cell in previous.cells if cell.cell_type == "code"} if previous else {}
spec = importlib.util.spec_from_file_location("report_builder", ROOT / "report/build_report.py")
report_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_builder)
report = report_builder.render_markdown()
# The technical report retains its required abstract; the notebook starts with introduction.
report = re.sub(r"(?ms)^## Abstract\n.*?(?=^## Introduction\n)", "", report)
parts = re.split(r"(?m)(?=^## Step [1-8]\.)", report)
assert len(parts) == 9
nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(parts[0] + "\n\nThe cells below load saved predictions and recalculate the evaluation tables. "
                                          "Transformer training and timing measurements are not repeated. "
                                          "Set `RUN_CV = True` to repeat resampling for the classical models and frozen probes.")]


def code(text):
    source = text.strip()
    # Preserve execution records when only the surrounding prose has changed.
    cells.append(previous_code[source] if source in previous_code else nbf.v4.new_code_cell(source))


code('''
import sys, json, sqlite3
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from IPython.display import Image, display
from src.data import build_dataset, load_splits
T, F = ROOT / "results/tables", ROOT / "results/figures"
RUN_CV = False  # True repeats classical/frozen-probe CV, not transformer fine-tuning
pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 4)
print("Evidence-reuse workflow; no transformer retraining or efficiency timing rerun.")
''')

evidence = {
    1: '''
s = load_splits()
rebuilt = build_dataset(save=False)
for name in ("train", "val", "test", "ood"):
    pd.testing.assert_frame_equal(getattr(s, name), getattr(rebuilt, name))
print(json.dumps(s.stats, indent=2))
import itertools
overlaps = {f"{a}/{b}": len(set(getattr(s, a).text) & set(getattr(s, b).text))
            for a, b in itertools.combinations(("train", "val", "test", "ood"), 2)}
assert not any(overlaps.values())
print("Exact normalized text overlaps:", overlaps)
display(Image(filename=str(F / "data_overview.png")))
''',
    2: '''
display(pd.read_csv(T / "tokenizer_stats.csv"))
print("Saved masked-language-model casing check:")
print((ROOT / "results/mlm_casing_check.json").read_text(encoding="utf-8"))
print("Test demographic proxies (unknown values are retained):")
display(pd.crosstab(s.test.sex, s.test.age_group, margins=True))
''',
    3: '''
display(pd.read_csv(T / "architectures.csv"))
''',
    4: '''
training = json.loads((ROOT / "results/classical_training.json").read_text())
display(pd.DataFrame(training).T[["best_params", "cv_macro_f1", "val_macro_f1"]])
rows = []
for model in ("biobert", "clinicalbert", "pubmedbert", "roberta"):
    info = json.loads((ROOT / f"results/transformer_{model}.json").read_text())
    for run in info["grid"]:
        rows.append({"model": model, "lr": run["lr"], "best_epoch": run["best_epoch"],
                     "epochs_run": len(run["history"]), "validation_macro_f1": run["best_val_macro_f1"],
                     "selected": run["lr"] == info["best_lr"]})
display(pd.DataFrame(rows))
db = sqlite3.connect(f"file:{(ROOT / 'mlflow.db').as_posix()}?mode=ro", uri=True)
print("MLflow status counts:", db.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall())
print("MLflow metric records:", db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0])
db.close()
display(Image(filename=str(F / "training_curves.png")))
''',
    5: '''
from src import analysis
analysis.main(run_cv=RUN_CV)
display(pd.read_csv(T / "performance.csv"))
display(pd.read_csv(T / "robustness.csv"))
display(pd.read_csv(T / "fairness.csv"))
display(pd.read_csv(T / "counterfactual_gender_swap.csv"))
for figure in ("confusion_matrices.png", "roc_curves.png", "robustness.png"):
    display(Image(filename=str(F / figure)))
''',
    6: '''
display(pd.read_csv(T / "efficiency.csv"))
display(pd.read_csv(T / "shap_logreg_top_features.csv").groupby("class", sort=False).head(3))
explanations = json.loads((ROOT / "results/local_explanations.json").read_text())
display(pd.DataFrame([{k: item[k] for k in ("test_index", "true", "pred_transformer", "p", "pred_logreg")}
                      for item in explanations]))
for figure in ("shap_logreg.png", "local_explanations.png", "attention.png"):
    display(Image(filename=str(F / figure)))
''',
    7: '''
for name in ("validation_selection", "mcnemar", "paired_bootstrap", "cv_summary", "cv_tests", "tradeoff"):
    print(name)
    display(pd.read_csv(T / f"{name}.csv"))
print("Bootstrap tail fractions are descriptive, not calibrated p-values.")
print("CV transformer rows use frozen representation probes, not fine-tuned models.")
display(Image(filename=str(F / "tradeoff.png")))
''',
    8: '''
from src.review import main as audit_saved_evidence
audit_saved_evidence()
summary = json.loads((ROOT / "results/analysis_summary.json").read_text())
print(json.dumps(summary, indent=2))
display(pd.read_csv(T / "error_pairs_pubmedbert.csv"))
display(pd.read_csv(T / "per_class_f1.csv"))
print("No diagnostic or prospective clinical outcome evaluation was performed.")
'''
}

for step, section in enumerate(parts[1:], 1):
    tail = ""
    if step == 8:
        section, tail = section.split("## Discussion\n", 1)
        tail = "## Discussion\n" + tail
    narrative, observations = section.split("**Observations**", 1)
    cells.append(nbf.v4.new_markdown_cell(narrative.strip()))
    code(evidence[step])
    cells.append(nbf.v4.new_markdown_cell("**Observations**" + observations.rstrip()))
    if tail:
        for subsection in re.split(r"(?m)(?=^## )", tail):
            if subsection.strip():
                cells.append(nbf.v4.new_markdown_cell(subsection.strip()))

insights_cell = next(cell for cell in cells if cell.cell_type == "markdown" and cell.source.startswith("## Final insights\n"))
insights_bullets = [line[2:] for line in insights_cell.source.splitlines() if line.startswith("- ")]
assert len(insights_bullets) == 6 and len(" ".join(insights_bullets).split()) == 300
assert not any(re.search(r"(?im)^#+\s+Abstract\b", cell.source) for cell in cells if cell.cell_type == "markdown")

nb["cells"] = cells
if previous:
    nb["metadata"] = previous.metadata
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.validate(nb)
nbf.write(nb, target)
print("Wrote", target)
