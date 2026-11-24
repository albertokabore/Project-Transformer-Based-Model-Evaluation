# Transformer-Based Model Evaluation — Medical Text Classification

**Albert Kabore, PhD student in AI at the University of the Cumberlands**

Seven-model evaluation of MTSamples specialty classification: TF-IDF Logistic Regression,
linear SVM, Random Forest, BioBERT, Bio_ClinicalBERT, PubMedBERT/BiomedBERT, and RoBERTa.

The report follows the eight assignment steps, with observations after each step.
It includes the results, discussion, final insights, recommendations, conclusions, and
APA references. Tables come from the saved CSV files.

The final report is [Technical_Report.docx](report/Technical_Report.docx).

The notebook, audit records, tables, and figures remain in the project as supporting
analysis files.

PubMedBERT has the highest observed test macro-F1 (0.9347), versus 0.9280 for Logistic
Regression. Its advantage over the linear baselines is statistically inconclusive.
Logistic Regression has faster CPU inference and stronger unseen-specialty detection
in the saved experiment. Specialty classification does not establish diagnostic ability.

## Reuse and check the existing evidence

The local raw data, classical model files, and MLflow database are needed for the full audit.
The notebook recomputes evaluation metrics and verifies artifacts; it does not retrain
transformers or repeat hardware measurements. Additional CV is optional and disabled by default.

```powershell
python -m src.review
python report/build_report.py
python notebooks/build_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Transformer_Model_Evaluation.ipynb --ExecutePreprocessor.timeout=600
```

Edit the authoring source at `src/report_template.md`, then run `python report/build_report.py`
to update the Word report. The builder produces one report file with embedded charts
and editable tables. The report retains its required abstract; the notebook omits that
section and contains six output-focused final-insight bullets totaling 300 words.

## Reproduce the original experiments

The experiments used Python 3.11 and CPU training, which takes hours on the recorded
laptop. A fresh workspace needs the dataset and model downloads before these commands
can complete.

```powershell
pip install -r requirements.txt
python -c "import kagglehub; print(kagglehub.dataset_download('tboyle10/medicaltranscriptions'))"
# Copy the downloaded mtsamples.csv to data/raw/mtsamples.csv.
python -m src.data
python -m src.tokenizer_stats
python -m src.classical
python -m src.transformer
python -m src.analysis --no-cv
python -m src.efficiency --quantize pubmedbert
python -m src.explain --lr-only
python -m src.explain --transformer pubmedbert
python -m src.pipeline_figure
python -m src.analysis
python -m src.review
python report/build_report.py
python notebooks/build_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Transformer_Model_Evaluation.ipynb --ExecutePreprocessor.timeout=600
```

`src.transformer` skips models with existing completed prediction artifacts, so these
commands do not force retraining over an existing run. Checkpoint revisions are not pinned;
the environment lock alone does not guarantee identical future downloads. The separate
`mlm_casing_check.json` records an exploratory prior check without a dedicated regeneration
script. Hardware timing is specific to the supplied records.

## Scope and reproducibility limits

- The corpus has 1,466 main-task documents across 11 specialties and 279 same-source notes
  from unseen specialties. Exact text overlaps between splits were checked; patient-level
  independence, near duplicates, and pre-training overlap remain unverified.
- Transformers train only their final two layers and classification head with 256-token inputs.
  Their results are not a guaranteed lower bound on full fine-tuning.
- Transformer CV rows evaluate frozen embedding probes, not repeated fine-tuning. Classical
  resampling uses previously selected parameters and is not a nested search evaluation.
- Fairness uses noisy demographic proxies inferred from text. Privacy screening is regex based;
  complete de-identification and clinical readiness are not established.
- Raw data, trained model weights, caches, and MLflow artifacts are excluded by `.gitignore`.
  A clean clone needs those artifacts or a new training run. Saved probabilities and tables
  allow substantial evaluation reuse without redoing training.
- MLflow can be inspected with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
  At review, 18 run records were FINISHED and one remained RUNNING.
