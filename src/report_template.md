# Transformer-Based Model Evaluation for Medical Text Classification and Clinical Decision Support

**Albert Kabore**  
**PhD student in AI at the University of the Cumberlands**  
Technical report | November 24, 2026

## Abstract

Seven models were compared on medical specialty classification: Logistic Regression, linear SVM, Random Forest, BioBERT, Bio_ClinicalBERT, PubMedBERT, and RoBERTa. After duplicate grouping and label filtering, the MTSamples file yielded 1,466 documents across 11 specialties; another 279 documents were reserved for testing unseen specialties. The comparison covered classification scores, corrupted text, demographic proxies, inference cost, and explanations. CPU limits restricted transformer training to the final two encoder layers and classification head. PubMedBERT ranked first on validation macro-F1 (0.9358) and test macro-F1 (0.9347), with 94.22% accuracy. Logistic Regression reached macro-F1 0.9280 and accuracy 93.88%. The paired macro-F1 difference was 0.0067, with a 95% bootstrap interval from −0.0300 to 0.0451; McNemar's adjusted p-value was 1.000. These results leave the advantage over Logistic Regression unresolved. The linear model was much faster on CPU and better at detecting unseen specialties. This dataset measures specialty classification; it supplies no evidence about diagnostic accuracy or patient outcomes.

## Introduction

Assigning a medical transcription to a specialty could help staff route documents, organize work queues, and retrieve relevant records. This study tests that task: each classifier receives a note and predicts one specialty label. Diagnoses confirmed by clinicians are unavailable, so diagnosis support cannot be assessed directly.

An accuracy score can conceal several problems. A copied note in both training and test data makes performance look better than it is. Errors in small specialties disappear inside an average, and a wrong answer delivered with high confidence can be difficult to challenge. This comparison checks those weaknesses alongside speed, explanations, and differences between demographic groups.

BERT supplies the starting architecture for the biomedical encoders (Devlin et al., 2019). BioBERT adapts it to biomedical literature, and Bio_ClinicalBERT adds training on clinical notes (Lee et al., 2020; Alsentzer et al., 2019). PubMedBERT uses biomedical pre-training and vocabulary; RoBERTa provides a general-domain comparison (Gu et al., 2021; Liu et al., 2019). Their performance is compared with three TF-IDF baselines on the same documents.

**What was checked.** The audit rebuilt the splits from the raw CSV, recalculated metrics and statistical tests from saved predictions, and checked predictions from the stored classical models. It also inspected training histories, explanations, and timing records. Transformer training and timing measurements were not repeated. Dataset provenance, overlap with pre-training material, and clinical effectiveness remain unverified. The checks are recorded in `results/review_audit.json`; run `python -m src.review` to repeat them.

## Methodology and results by assignment step

Steps 1–4 cover the data and training. Evaluation spans Steps 5 and 6, followed by statistical comparison and reporting in Steps 7 and 8.

![Figure 1. Scripted evaluation pipeline. Exact duplicate grouping precedes stratified splitting; all seven architectures are evaluated from saved predictions.](../results/figures/pipeline.png)

## Step 1. Data collection

The analysis uses `data/raw/mtsamples.csv`, supplied for the assignment's [Kaggle Medical Transcriptions dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (tboyle10, n.d.). The counts below were reproduced from that file. MIMIC-III was not used.

| Audit stage | Documents or rows | Interpretation |
|---|---:|---|
| Raw file | 4,999 | 40 category labels |
| Nonmissing transcriptions | 4,966 | 33 rows lack transcription text |
| Unique stripped transcription texts | 2,357 | 2,609 excess duplicate rows, or 52.54% of nonmissing rows |
| Exactly one retained specialty after label filtering | 1,745 | Candidate single-label documents |
| No retained specialty | 295 | Excluded |
| Two retained specialties | 306 | Excluded as ambiguous |
| Three retained specialties | 11 | Excluded as ambiguous |
| In-distribution corpus | 1,466 | 11 labels with at least 50 unique documents each |
| Unseen-specialty corpus | 279 | 19 less frequent labels; excluded from training |

Many transcriptions appear under several categories. Grouping identical text before splitting keeps copies of a note together. The label policy excludes ten categories: Surgery, Consult - History and Phy., SOAP / Chart / Progress Notes, Discharge Summary, Office Notes, Letters, Emergency Room Reports, IME-QME-Work Comp etc., General Medicine, and Radiology. Surgery, Radiology, and General Medicine are legitimate clinical fields; their exclusion defines this particular benchmark. A note is retained only when one label remains.

Removing ambiguous and uncommon cases narrows the workload and may make it easier than routine clinical routing. The unseen-specialty notes also come from MTSamples, so they test unfamiliar labels within the same source. Patient, institution, and date identifiers are too limited to support reliable patient-level, hospital-level, or chronological splits.

**Observations**

- Duplicate grouping reduced 4,966 nonmissing rows to 2,357 unique transcriptions, removing 2,609 rows. The classification set contains 1,466 documents across 11 specialties after label filtering.
- Another 279 notes belong to 19 unseen specialties. The 317 documents with multiple retained specialty labels were excluded, leaving their classification difficulty outside the scores.

## Step 2. Preprocessing

Preprocessing consists of five operations:

1. **Identifier screening:** regular expressions replace email addresses, phone numbers, dates, selected identifiers, and titled names. Across the 1,745 retained documents, including unseen specialties, the script replaced 880 date strings, 334 titled names, and one phone number. These counts record pattern matches. Complete PHI removal was not verified, and the script alone does not meet a formal de-identification standard.
2. **Normalization:** punctuation artifacts and repeated whitespace are cleaned. A dictionary of 34 abbreviations produces 692 expansions. DM, CVA, and LOC can have different meanings depending on context; clinicians have not reviewed the expansions for accuracy.
3. **Feature extraction:** TF-IDF is fitted within each grid-search training fold, using sublinear term frequency, minimum document frequency 2, maximum document frequency 90%, and unigram or bigram candidates. The token pattern retains tokens of at least two characters beginning with a letter and permits subsequent letters, digits, or hyphens. Transformer models use their checkpoint-specific vocabularies.
4. **Imbalance handling:** Logistic Regression and SVM use balanced class weights; Random Forest uses `balanced_subsample`; transformers use class-weighted cross-entropy. The largest retained class contains 339 documents and the smallest 50, a ratio of 6.78:1. Macro-F1 gives each specialty equal weight.
5. **Splitting:** stratified splitting with seed 42 assigns approximately 70% of notes to training, 10% to validation, and 20% to testing: 1,025, 147, and 294 documents. Rebuilding the splits reproduced every row in its saved order. No exact normalized text appears in more than one split, including the unseen-specialty set. Near duplicates and shared patient histories remain possible.

The tokenizer audit measures the training notes only:

{{table:tokenizer_stats|model,vocab_size,fertility (pieces/word),clinical terms kept whole (/30),median tokens/note,% notes > 256 tokens|2}}

*Table 1. Vocabulary size, token fragmentation, a 30-term vocabulary check, and the share of training notes exceeding the input limit. Fragmentation alone cannot explain performance differences.*

Case is preserved for BioBERT and Bio_ClinicalBERT. In the saved masked-language-model check on 24 notes, preserved case gave lower loss: 1.348 versus 1.655 for BioBERT, and 1.371 versus 1.655 for Bio_ClinicalBERT. Different tokenizations can change the prediction targets themselves, so this small check offers limited support for the choice. It does not establish which setting would classify notes better.

The `sex` field comes from gendered words in raw notes, requiring at least two cues and a 2:1 majority. Age comes from the first matching year-old expression. A cue could describe a spouse or another person. Neither field has been checked against patient records, and the sex field cannot establish biological sex or gender identity.

![Figure 2. Retained specialty counts and document lengths. The main corpus has a median length of 383 words.](../results/figures/data_overview.png)

**Observations**

- Screening replaced 880 date strings, 334 titled names, and one phone number; abbreviation expansion produced 692 replacements. These counts describe matches rather than privacy protection.
- The splits contain 1,025 training, 147 validation, and 294 test notes. Normalized text overlap between splits is zero; the largest class outnumbers the smallest 6.78:1.

## Step 3. Model selection

Three traditional classifiers and four transformers use the same splits and label mapping. Each transformer has 12 encoder layers, a hidden size of 768, and 12 attention heads.

| Model | Architecture and features | Training in this study | Input scope |
|---|---|---|---|
| Logistic Regression | Multinomial linear classifier over TF-IDF | All classifier weights; selected unigram vocabulary | Full note |
| Linear SVM | Linear maximum-margin classifier with sigmoid calibration | Five classifier/calibrator pairs; selected unigrams | Full note |
| Random Forest | Ensemble of decision trees over up to 20,000 TF-IDF features | 300 selected trees; unigram and bigram features | Full note |
| BioBERT v1.2 | Biomedical BERT with cased WordPiece vocabulary | Top two layers and classification head; approximately 14.2 million parameters | First 256 tokens |
| Bio_ClinicalBERT | Clinical BERT adapted from BioBERT | Top two layers and head; approximately 14.2 million parameters | First 256 tokens |
| PubMedBERT / BiomedBERT | Biomedical BERT with in-domain uncased vocabulary | Top two layers and head; approximately 14.2 million parameters | First 256 tokens |
| RoBERTa-base | General-domain encoder with byte-level BPE | Top two layers and head; approximately 14.2 million parameters | First 256 tokens |

*Table 2. Architecture comparison. The 256-token limit includes special tokens; RoBERTa uses BPE rather than WordPiece.*

The exact checkpoints are `dmis-lab/biobert-base-cased-v1.2`, `emilyalsentzer/Bio_ClinicalBERT`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, and `FacebookAI/roberta-base`. Microsoft's [model card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) documents the PubMedBERT-to-BiomedBERT naming change and this abstract-plus-full-text checkpoint.

Input length differs between systems. TF-IDF sees the whole note; transformers see a prefix. To examine that difference, an additional Logistic Regression model receives the prefix covered by Bio_ClinicalBERT's first 256 tokens. PubMedBERT and RoBERTa tokenize differently, so their prefixes do not match this control exactly. Throughout the report, ClinicalBERT refers specifically to Bio_ClinicalBERT.

**Observations**

- The table contains three TF-IDF classifiers and four transformers. Each transformer trains approximately 14.2 million parameters, while its frozen layers remain part of inference cost.
- PubMedBERT represents 23 of 30 terms as single tokens, compared with one for BioBERT. Median training-note lengths are 534 and 672 tokens, exceeding input limits.

## Step 4. Training

The classical models are tuned with five stratified folds from the 1,025 training documents. Each fold refits the complete TF-IDF pipeline. Mean macro-F1 determines the selected settings.

| Model | Search space | Selected configuration | Training CV macro-F1 |
|---|---|---|---:|
| Logistic Regression | C = 1, 10, 100, 1000; unigrams or unigrams plus bigrams | C = 100; unigrams | 0.8888 |
| Linear SVM | C = 0.1, 0.5, 1, 5; same n-gram options | C = 0.5; unigrams | 0.8956 |
| Random Forest | 300 or 600 trees; max_features = sqrt or 0.05; minimum leaf size = 1 or 2 | 300 trees; sqrt; minimum leaf size 2 | 0.8199 |

*Table 3. Search results from `results/classical_training.json`. The SVM CV score evaluates the uncalibrated classifier; final SVM probabilities come from a subsequent calibration ensemble.*

SVM probabilities come from sigmoid calibration with five internal folds. Here, TF-IDF is fitted on the entire training split before calibration, allowing the calibration folds to share vocabulary and inverse document frequencies. Validation and test notes remain outside that fitting. A stricter implementation would place the whole text pipeline inside the calibration wrapper and tune it there.

Training used an Intel Core i3-N305 CPU with eight threads and no GPU. To reduce the cost, the transformers cache lower-layer states in float16, then train the top two layers and a classification head consisting of masked mean pooling, dropout, and a linear layer. The frozen layers run in evaluation mode, with their dropout disabled. Cached float16 states introduce rounding; this setup differs from running the whole network in training mode.

The common transformer settings are maximum length 256 tokens, batch size 16, AdamW, weight decay 0.01, 10% warm-up followed by linear decay, class-weighted cross-entropy, and gradient norm clipping at 1.0. The learning-rate grid is {0.00003, 0.0001}. Validation macro-F1 controls early stopping with patience two, at most six epochs, and restoration of the best checkpoint. Classical models use their own optimization algorithms; gradient clipping and neural-network early stopping do not apply to those estimators.

| Transformer | Selected learning rate | Selected epoch | Validation macro-F1 |
|---|---:|---:|---:|
| BioBERT | 0.0001 | 5 | 0.8863 |
| Bio_ClinicalBERT | 0.00003 | 5 | 0.9096 |
| PubMedBERT | 0.0001 | 6 | 0.9358 |
| RoBERTa | 0.0001 | 4 | 0.8533 |

*Table 4. Selected checkpoints from the saved training histories; seed 42.*

Freezing layers follows an approach studied by Lee et al. (2019). It saves computation, but the effect on this dataset is unknown: full fine-tuning could improve or worsen the scores. The experiment uses one seed and two learning rates, leaving much of that uncertainty unmeasured.

MLflow contains 399 metric entries and 19 run records. Eighteen are marked FINISHED; one is still marked RUNNING. Its status needs to be checked against the logs. The selected models have saved training histories, but those records do not explain why the other run was left open.

Five-by-two and ten-fold resampling use the combined training and validation notes. They cover the classical methods and frozen transformer embeddings fitted with Logistic Regression probes. The fine-tuned transformers were not retrained in each fold, leaving that part of the assignment incomplete.

![Figure 3. Saved training loss and validation macro-F1 histories for both learning-rate settings.](../results/figures/training_curves.png)

**Observations**

- PubMedBERT's checkpoint occurs at epoch six with validation macro-F1 0.9358. Bio_ClinicalBERT reaches 0.9096 at epoch five; BioBERT and RoBERTa reach 0.8863 and 0.8533, respectively.
- Grid-search macro-F1 is 0.8956 for SVM, 0.8888 for Logistic Regression, and 0.8199 for Random Forest. These cross-validation scores describe different samples from the transformer validation scores.

## Step 5. Evaluation: predictive performance, robustness, and fairness

### Predictive performance

The test set contains 294 notes across 11 specialties. Accuracy counts correct labels; macro precision, recall, and F1 give each specialty equal weight. Weighted F1 accounts for class size. ROC-AUC is calculated one-versus-rest with macro and micro averages. Expected calibration error (ECE) uses 15 equal-width confidence bins to compare stated confidence with observed correctness (Guo et al., 2017).

{{table:performance|Model,Accuracy,Macro-P,Macro-R,Macro-F1,Weighted-F1,ROC-AUC (macro OvR),ECE|4}}

*Table 5. Test metrics, as proportions. The eighth row is a length ablation rather than an additional architecture.*

{{table:performance|Model,Acc 95% CI,F1 95% CI|3}}

*Table 6. Percentile 95% bootstrap intervals from 1,000 resamples of test notes. The fitted models stay fixed, so the intervals exclude variation from retraining or model selection.*

PubMedBERT gets 277 labels right. Logistic Regression and SVM each get 276. Rankings change with the metric: Logistic Regression has the higher macro ROC-AUC, 0.9964 versus 0.9915, and Bio_ClinicalBERT has the lowest ECE at 0.0325. Calibration on these notes still leaves open how confidence behaves on unfamiliar inputs.

Restricting Logistic Regression to the 256-token prefix lowers macro-F1 from 0.9280 to 0.8811, a drop of 0.0469. Later parts of a note appear useful to this model. Because the transformer tokenizers produce different prefixes, the ablation cannot isolate architecture effects across all four encoders.

![Figure 4. Row-normalized confusion matrices for all seven architectures; rows are true specialties and columns predicted specialties.](../results/figures/confusion_matrices.png)

![Figure 5. Micro-averaged ROC curves and class-specific ROC curves for the validation-selected PubMedBERT model.](../results/figures/roc_curves.png)

![Figure 6. Reliability plots. The plot uses ten bins and displays bins containing at least five notes; the tabulated ECE uses fifteen bins.](../results/figures/reliability.png)

### Robustness and unseen-specialty detection

The robustness tests introduce typos with probabilities 0.10 and 0.25, delete words with probability 0.20, or mask specialty terms. The masking list contains 164 distinct terms drawn from the top 15 Logistic Regression unigram coefficients for each class. This targets the linear model and tests transfer to the others. Actual corruption rates vary by note. No optimized adversarial search was run, and clinicians did not verify that the altered notes retained their original meaning.

{{table:robustness|Model,clean,typo_10,typo_25,word_drop_20,keyword_mask,OOD AUROC (MSP),OOD FPR@95TPR|4}}

*Table 7. Perturbation macro-F1 and maximum-probability OOD detection. Higher AUROC is better; lower OOD false-positive rate is better.*

Masking keywords lowers PubMedBERT's macro-F1 from 0.9347 to 0.8615 and Logistic Regression's from 0.9280 to 0.8346. Some mild corruptions improve scores slightly on this test set. Each condition has only one generated version, so those small changes could depend on which words were altered. Removing clinical terms can also remove valid evidence for a label.

Unseen-specialty detection uses the largest predicted class probability, following Hendrycks and Gimpel (2017). SVM contributes calibrated probabilities; Random Forest uses its native estimates. In-distribution notes count as positive. A threshold at the fifth percentile of their test confidence retains roughly 95% of them. OOD FPR reports how many unseen-specialty notes also pass that threshold. This test-set calculation describes model behavior; any threshold used in practice would need separate validation data.

Logistic Regression separates known from unseen specialties better, with OOD AUROC 0.9139 against PubMedBERT's 0.8414. Even so, it accepts 45.88% of unseen-specialty notes at this threshold. PubMedBERT accepts 65.95% and assigns those OOD notes a mean maximum probability of 0.8455. Both models can give confident answers when the correct specialty is absent from their label set.

![Figure 7. Macro-F1 under each perturbation condition.](../results/figures/robustness.png)

### Exploratory fairness assessment

Demographic parity difference compares predicted label frequencies between groups A and B, taking the largest absolute class-specific gap. Equal opportunity compares per-class true-positive rates, following Hardt et al. (2016). A class enters that calculation only if both groups have at least five notes. The group labels come from text cues rather than verified demographics.

{{table:fairness|Model,attribute,group A,acc A,group B,acc B,DP diff (max over classes),EO gap max !dTPR!,EO classes evaluated|4}}

*Table 8. Proxy subgroup comparisons. DP denotes demographic parity; EO denotes equal opportunity. EO coverage is only four classes for the sex proxy and three for the adult-age comparison.*

Text cues classify 89 test notes as female-coded and 91 as male-coded; 114 remain unknown. Age extraction identifies 74 notes aged 18–64, 44 aged 65 or older, and 14 younger than 18, leaving 162 unknown. The adult comparison excludes the younger group. Missing values and small class-by-group samples leave much of the population unexamined.

PubMedBERT accuracy is 91.01% on female-coded notes and 93.41% on male-coded notes. Its maximum EO gaps are 0.1667 for sex proxies and 0.1000 for adult age. The sex-proxy DP difference, 0.1913, equals the largest gap in true label frequencies. Matching those two maxima says nothing about whether every class has matching disparities. Specialties linked to sex or age also have different base rates, which complicates interpretation.

{{table:counterfactual_gender_swap|Model,flips (n),flip rate (all),macro-F1 swapped|4}}

*Table 9. Sensitivity to automated gendered-word substitutions, evaluated on all 294 test notes.*

Swapping gendered words changes one PubMedBERT prediction, nine Logistic Regression predictions, and ten RoBERTa predictions. A swap can produce medically inconsistent text, so this measures sensitivity to substitutions rather than causal counterfactual fairness. Small flip rates and nonsignificant subgroup tests leave discrimination unresolved. Race, ethnicity, language, socioeconomic status, and intersections between groups were not assessed.

**Observations**

- PubMedBERT correctly labels 277 of 294 notes, reaching macro-F1 0.9347. Logistic Regression labels 276 correctly with macro-F1 0.9280, a difference of one correct specialty label.
- Unseen-specialty AUROC favors Logistic Regression, 0.9139 versus 0.8414. PubMedBERT accepts 65.95% of unseen notes at the reported threshold, compared with 45.88% for the linear baseline.

## Step 6. Evaluation continued: computational efficiency and interpretability

### Computational efficiency

Each model was measured in a fresh process on the same CPU. After one warm-up call, latency was recorded for 30 individual notes and throughput for a batch of 64. Timing includes feature extraction or tokenization, starting with cleaned text; PHI screening and normalization are excluded. File size and peak memory are expressed in MiB, using division by 2²⁰, despite the CSV columns being labeled MB.

{{table:efficiency|Model,latency median (ms),latency p95 (ms),throughput (docs/s),peak memory (MB),serialized size (MB)|2}}

*Table 10. Hardware-specific saved inference records. Peak memory includes interpreter, libraries, loading, and temporary allocations; it is not isolated resident weight memory.*

Median latency is 1.09 ms for Logistic Regression and 613.52 ms for PubMedBERT, about 562 times longer. Those numbers describe the systems as implemented, including their different input lengths. Size comparisons need care: classical files are compressed joblib pipelines, while transformer files are state dictionaries that exclude tokenizers and software dependencies. Only one machine and one measurement sequence were used; there are no uncertainty intervals or concurrent-load tests.

Caching and tuning took roughly 142.2 minutes for Bio_ClinicalBERT, 110.1 for PubMedBERT, 122.2 for BioBERT, and 84.1 for RoBERTa. These totals include caching evaluation and corrupted inputs. Classical grid searches took 1.45, 0.35, and 5.83 minutes for Logistic Regression, SVM, and Random Forest. The SVM timer excludes later calibration, so these totals cover different amounts of work.

For every transformer, the end-to-end float32 implementation matches the cached pipeline's predicted labels on all 294 test notes. The probabilities can still differ, and agreement on this set does not guarantee agreement on new notes.

Dynamic int8 quantization cuts PubMedBERT's median latency from 613.52 to 343.60 ms and its stored state from 417.74 to 173.08 MiB. The cost is substantial: macro-F1 falls from 0.9347 to 0.8813, and peak process memory rises from 863.86 to 1,170.42 MiB. This version would sacrifice accuracy for speed. Its packed weights also change what `.parameters()` counts, making the apparent parameter reduction unsuitable as evidence of architectural compression.

### Interpretability

Linear SHAP explains Logistic Regression against a background of 500 training documents. The contributions sum in raw class-logit space. They do not describe changes in predicted probability or binary log-odds against all other classes. The chosen background and correlations between TF-IDF features affect their interpretation (Lundberg & Lee, 2017).

![Figure 8. Linear SHAP importance for Logistic Regression, expressed in raw class-logit units.](../results/figures/shap_logreg.png)

Two selected notes illustrate PubMedBERT explanations: one correct prediction and one error. SHAP Partition uses up to 500 evaluations per note. LIME uses 300 perturbations for the transformer and 1,000 for Logistic Regression. Explanations follow each model's input scope: the prefix for PubMedBERT, the full note for the baseline. LIME fits a local approximation to model behavior (Ribeiro et al., 2016).

![Figure 9. Local SHAP and LIME explanations for test notes 0 and 73. Blue supports the displayed class; orange opposes it.](../results/figures/local_explanations.png)

Note 73 shows the risk clearly. Its label is Hematology - Oncology, yet PubMedBERT predicts Urology with probability 0.9956. Logistic Regression also gets it wrong, choosing Gastroenterology at 0.3194. A clinician would need to examine the note, its label, and the explanations to judge the error. These two examples cannot establish explanation quality across the dataset.

![Figure 10. Last-layer attention for one selected note. Attention visualizes model computation but does not establish a faithful explanation.](../results/figures/attention.png)

The attention plot is shown as a view of model computation. Attention weights alone can fail to explain a prediction faithfully (Jain & Wallace, 2019). The SHAP axes use class-logit units.

**Observations**

- Median inference takes 1.09 milliseconds for Logistic Regression and 613.52 for PubMedBERT. Their storage sizes are 0.81 and 417.74 MiB, though serialization formats differ.
- Quantized PubMedBERT reaches 343.60 milliseconds and 173.08 MiB, but macro-F1 drops to 0.8813. Peak memory rises to 1,170.42 MiB, exceeding the original model's process memory usage.

## Step 7. Optimization and statistical comparison

Validation macro-F1 selects the reference model. PubMedBERT leads at 0.9358, followed by SVM at 0.9333 and Logistic Regression at 0.9304. The code follows this rule, but saved files cannot establish whether developers ever inspected the test set. No preregistration is available to support a stronger claim about the comparison protocol.

McNemar's test compares whether the models get the same notes right or wrong. Fewer than 25 discordant pairs trigger an exact binomial test; larger counts use continuity-corrected chi-square. Holm correction covers all 21 pairs. Only three adjusted comparisons fall below p = .05: Logistic Regression, SVM, and PubMedBERT each outperform Random Forest on accuracy. The remaining tests leave differences unresolved; they do not establish equivalence or test macro-F1.

| Accuracy comparison | First model correct / second wrong | First wrong / second correct | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Logistic Regression vs. PubMedBERT | 8 | 9 | 1.0000 | 1.0000 |
| SVM vs. PubMedBERT | 7 | 8 | 1.0000 | 1.0000 |
| Random Forest vs. PubMedBERT | 9 | 33 | 0.000387 | 0.007735 |
| PubMedBERT vs. RoBERTa | 18 | 4 | 0.004344 | 0.078183 |

*Table 11. Selected McNemar comparisons. The complete 21-pair table is `results/tables/mcnemar.csv`.*

{{table:paired_bootstrap|reference,vs,delta macro-F1,95% CI low,95% CI high|4}}

*Table 12. Paired percentile-bootstrap differences: PubMedBERT macro-F1 minus comparator macro-F1, using 1,000 shared resamples. These six intervals are pointwise, without multiplicity adjustment.*

The paired macro-F1 interval for PubMedBERT minus Logistic Regression spans −0.0300 to 0.0451 around a difference of 0.0067. The SVM interval also crosses zero. Intervals against some weaker models exclude zero, but these use macro-F1 without multiplicity adjustment, unlike the McNemar tests. The CSV field `P(delta<=0)` counts bootstrap samples with nonpositive differences. It is neither a calibrated p-value nor a posterior probability and is omitted from Table 12.

Resampling uses the 1,172 training and validation documents, excluding the held-out test set. Dietterich's five-by-two paired t-test and a Nadeau–Bengio-style correction for ten-fold scores account approximately for dependence between fitted comparisons (Dietterich, 1998; Nadeau & Bengio, 2003). Treating correlated fold scores as independent would make the estimates appear too precise.

{{table:cv_summary||4}}

*Table 13. Resampled macro-F1 means and standard deviations. Each `_frozen` row is an unfine-tuned transformer representation with a standardized Logistic Regression probe; it is not the corresponding final transformer classifier.*

SVM leads the ten-fold comparison at 0.9125; Logistic Regression follows at 0.9048. Their raw p-values are 0.5440 for five-by-two and 0.2168 for corrected ten-fold testing, with both adjusted to 1.000. Hyperparameters were selected earlier using overlapping data, so these are not nested estimates of the full tuning procedure. SVM resampling also uses the uncalibrated classifier, whereas the test results use the calibrated ensemble. Frozen transformer probes leave fine-tuning stability untested.

Both PubMedBERT and Logistic Regression merit further testing. The former has the highest observed macro-F1; the latter costs far less to run. The int8 result rules out assuming that compression preserves performance. Longer inputs, full fine-tuning, temperature scaling, ensembles, and alternative OOD detectors remain experiments to run.

![Figure 11. Observed test macro-F1 versus median CPU latency. These measurements support a trade-off, not a statistically established universal winner.](../results/figures/tradeoff.png)

**Observations**

- PubMedBERT exceeds Logistic Regression by 0.0067 macro-F1, with a paired interval from −0.0300 to 0.0451. Their Holm-adjusted McNemar p-value is 1.000, leaving superiority unresolved.
- Ten-fold macro-F1 averages are 0.9125 for SVM and 0.9048 for Logistic Regression. Both adjusted paired-test p-values equal 1.000; these resampling outputs do not establish a difference.

## Step 8. Reporting, error analysis, and clinical implications

### Error analysis

PubMedBERT misclassifies 17 test notes. Its mean maximum probability on errors is 0.7821, and six errors exceed 0.90 confidence; correct predictions average 0.9860. Three confusion directions each occur twice: Cardiovascular / Pulmonary to Neurology, the reverse direction, and Orthopedic to ENT - Otolaryngology. Explaining why requires clinicians to review the individual notes.

{{table:per_class_f1|Unnamed: 0,TF-IDF + Logistic Regression,PubMedBERT (BiomedBERT),support|4}}

*Table 14. Per-specialty F1 for the leading baseline and transformer, with test support. Perfect F1 on small samples does not imply perfect clinical performance.*

Hematology - Oncology has only 12 test notes. F1 is 0.8148 for PubMedBERT and 0.7826 for Logistic Regression. PubMedBERT does well on Gastroenterology; the linear model scores higher on ENT and Urology. Model choice may depend on which specialties a service receives most often and which errors it can tolerate.

### Ethical reliability and clinical relevance

No part of this experiment measures treatment decisions, patient outcomes, or benefit to clinicians. Ambiguous cases were excluded, privacy screening remains incomplete, and demographics were inferred from text. Shared templates, near duplicates, and overlap with pre-training material could affect the scores. The study has no external-site, temporal, prospective, or clinical workflow validation.

The World Health Organization's [ethics and governance guidance](https://www.who.int/publications/i/item/9789240029200) calls for human oversight, transparency, accountability, and equity (World Health Organization, 2021). In a routing service, staff would need to know when a prediction is uncertain, correct misrouted notes, and reject an output when necessary. The present evaluation establishes neither ethical certification nor regulatory compliance.

### Requirement coverage and remaining limitations

| Assignment requirement | Evidence | Review conclusion |
|---|---|---|
| Labeled medical text collection | Raw CSV, reproducible label audit and splits | Specialty classification supported; diagnosis-support outcomes unavailable |
| Preprocessing, imbalance handling, stratified splits | `src/data.py`, class weights, tokenizer audit | Implemented; complete PHI removal and abbreviation correctness unverified |
| At least four architectures | Three classical models and four transformers | Met |
| Hyperparameter search and tracking | Classical grids, two transformer learning rates, MLflow histories | Implemented within limited search budgets |
| Early stopping and clipping | Transformer training code and histories | Met where applicable |
| Cross-validation | Classical grids, five-by-two and ten-fold analyses, frozen probes | Partial for final fine-tuned transformers; nested tuning absent |
| Accuracy, precision, recall, F1, ROC-AUC | Saved probabilities and metric tables | Recomputed and checked |
| Robustness and adversarial examples | Heuristic corruptions, targeted keyword masking, unseen labels | Implemented with limitations; no optimized semantic-preserving attack |
| Fairness | DP, EO, subgroup scores, gendered-word swaps | Exploratory proxy audit; insufficient to establish fairness |
| Efficiency and interpretability | Timing records, sizes, SHAP, LIME, attention | Available; timing not repeated; transformer explanations cover two notes |
| Statistical optimization | McNemar, paired bootstrap, corrected CV tests, quantization | Available; leading-model superiority not established |
| Technical report | Required narrative sections, comparison tables, charts, references | Report format covered; remaining experimental gaps listed above; separate rubric unavailable |

The notebook loads saved predictions and recalculates evaluation results. Its executed cells document that process, without rerunning transformer training. Raw data, model weights, and MLflow records are local files excluded by `.gitignore`; another researcher would need those files or a new training run. `requirements.txt` pins software versions, but checkpoint revisions remain unpinned, which limits exact reproduction after future model-hub changes.

**Observations**

- PubMedBERT makes 17 test errors, including six above 0.90 confidence. Mean prediction confidence is 0.7821 on errors, compared with 0.9860 on correctly classified notes.
- Hematology and Oncology F1 is 0.8148 for PubMedBERT and 0.7826 for Logistic Regression on 12 notes. The linear model scores higher on ENT and Urology specialties.

## Discussion

The main choice is between a small measured gain in classification and a large difference in running cost. PubMedBERT gets one more test label right than either linear baseline. Its paired macro-F1 interval still allows an advantage for Logistic Regression. A service choosing between them would need to decide what an additional correct routing decision is worth and test that assumption on a larger sample.

PubMedBERT handles keyword masking better and has lower ECE than Logistic Regression. The linear model is faster, occupies less storage, and separates unseen specialties more successfully. Neither result settles whether staff could use the model safely: the data contain no observations of actual routing decisions, corrections, delays, or harm.

The largest changes to the study happen before classification. Duplicate grouping reduces 4,966 nonmissing rows to 2,357 unique texts, and label filtering narrows the main task further. What remains is a useful comparison of these systems on selected MTSamples notes. Ambiguous cases, new hospitals, and variation between training runs still need their own evaluation.

## Final insights

- PubMedBERT produced the highest test macro-F1, 0.9347, and accuracy, 94.22%, correctly labeling 277 of 294 notes. Logistic Regression and SVM each classified 276 correctly. The difference amounts to one note, and the paired confidence interval crosses zero, leaving the apparent advantage over the linear baselines statistically unresolved on this dataset.
- Robustness depended on the failure being tested. Keyword masking reduced PubMedBERT's macro-F1 to 0.8615 and Logistic Regression's to 0.8346. For unseen specialties, Logistic Regression achieved AUROC 0.9139 against PubMedBERT's 0.8414. Their acceptance rates for unseen notes, 45.88% and 65.95%, show substantial remaining risk when the correct specialty is unavailable.
- The fairness outputs describe only part of the test population. PubMedBERT accuracy was 91.01% for female-coded notes and 93.41% for male-coded notes. Sex-proxy equal-opportunity gaps reached 0.1667. Unknown age affected 162 notes, and only four classes supported sex comparisons, preventing conclusions about fairness across all specialties or demographic groups.
- Logistic Regression required 1.09 milliseconds per note at the median, compared with 613.52 milliseconds for PubMedBERT. Quantization lowered transformer latency to 343.60 milliseconds and storage to 173.08 MiB, but macro-F1 dropped to 0.8813 and peak memory increased to 1,170.42 MiB. The smaller saved model did not deliver equivalent predictive performance.
- Explanation outputs identify associations and specific mistakes. Logistic Regression's cardiovascular SHAP ranking emphasized chest, artery, and coronary. PubMedBERT assigned Urology to an oncology note with probability 0.9956. Six of its 17 errors exceeded 0.90 confidence. Two local explanation examples cannot establish faithful reasoning, and high confidence did not guarantee a correct specialty.
- Class-level results vary beneath the aggregate ranking. Hematology and Oncology F1 reached 0.8148 for PubMedBERT versus 0.7826 for Logistic Regression on 12 notes. The linear model scored higher for ENT and Urology. These outputs support retaining both candidates for specialty-routing evaluation; they provide no measured diagnostic accuracy or patient benefit.

## Recommendations

1. Retain PubMedBERT and Logistic Regression for the next comparison. Their macro-F1 difference is 0.0067 with an interval crossing zero, so the current ranking cannot justify dropping the linear model. A larger external test set and repeated training seeds would help determine whether the small advantage persists.
2. For a supervised routing trial after external validation, assess Logistic Regression first when CPU latency matters: 1.09 milliseconds per note is much lower than PubMedBERT's 613.52. Measure actual misrouting costs alongside speed before selecting either model for a service.
3. Address unseen-specialty rejection before automating routing. PubMedBERT accepts 65.95% of unseen notes at the reported operating point, and Logistic Regression accepts 45.88%. Use separate validation data to set abstention thresholds and report the error rate among accepted notes as well as the review workload.
4. Do not adopt the tested int8 version on speed alone. Its macro-F1 falls from 0.9347 to 0.8813 while peak memory increases. Any revised compression method needs a fresh accuracy, latency, and memory comparison against the uncompressed model.
5. Prioritize clinical review of the 17 PubMedBERT errors, especially the six above 0.90 confidence and the oncology note assigned to Urology at 0.9956. Hematology - Oncology has only 12 test notes and F1 0.8148, making more labeled examples and class-specific error analysis a priority.
6. Revisit the subgroup results using verified demographics. Age is unknown for 162 notes, and sex-proxy equal-opportunity comparisons cover only four classes. Report uncertainty and missingness before interpreting the observed 91.01% versus 93.41% accuracy gap as evidence about patient equity.
7. Test longer transformer inputs because truncating Logistic Regression reduces macro-F1 from 0.9280 to 0.8811. Keep the input scope explicit and repeat the calibrated SVM pipeline within cross-validation; its 0.9125 resampling mean currently describes an uncalibrated classifier.
8. Check privacy-screening failures alongside its recorded 880 date, 334 name, and one phone substitutions. Those outputs count matches without measuring missed identifiers. The available outputs also contain no diagnostic accuracy or patient outcomes; evaluating diagnosis support requires a separate clinically adjudicated target and prospective evidence.

## Conclusions

PubMedBERT ranks first on test macro-F1, at 0.9347, and reaches 94.22% accuracy under partial fine-tuning. The paired tests do not establish an advantage over Logistic Regression or SVM. Logistic Regression reaches macro-F1 0.9280 with much lower CPU latency and better unseen-specialty detection.

Both models are candidates for further specialty-routing research. Neither has been validated for diagnosis or autonomous clinical use. The next decision should rest on external data and observed workflow errors, with privacy and demographic checks completed before patient care is affected.

## References

Alsentzer, E., Murphy, J., Boag, W., Weng, W.-H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. In *Proceedings of the 2nd Clinical Natural Language Processing Workshop* (pp. 72–78). Association for Computational Linguistics. https://doi.org/10.18653/v1/W19-1909

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). *Domain-specific language model pretraining for biomedical natural language processing* (arXiv:2007.15779, Version 6) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2007.15779

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning* (arXiv:1610.02413) [Preprint]. arXiv. https://arxiv.org/abs/1610.02413

Hendrycks, D., & Gimpel, K. (2017). *A baseline for detecting misclassified and out-of-distribution examples in neural networks*. International Conference on Learning Representations. https://arxiv.org/abs/1610.02136

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. In *Proceedings of NAACL-HLT 2019* (pp. 3543–3556). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1357

Lee, J., Tang, R., & Lin, J. (2019). *What would Elsa do? Freezing layers during transformer fine-tuning* (arXiv:1911.03090) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1911.03090

Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics, 36*(4), 1234–1240. https://doi.org/10.1093/bioinformatics/btz682

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A robustly optimized BERT pretraining approach* (arXiv:1907.11692) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1907.11692

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*. https://arxiv.org/abs/1705.07874

Microsoft. (n.d.). *BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext* [Model card]. Hugging Face. Retrieved November 24, 2026, from https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*, 239–281. https://doi.org/10.1023/A:1024068626366

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144). https://doi.org/10.1145/2939672.2939778

tboyle10. (n.d.). *Medical transcriptions* [Data set]. Kaggle. https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

World Health Organization. (2021). *Ethics and governance of artificial intelligence for health: WHO guidance*. https://www.who.int/publications/i/item/9789240029200
