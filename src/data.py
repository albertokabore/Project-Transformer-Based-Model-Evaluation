"""Data loading, label resolution, PHI scrubbing, normalization and splitting for MTSamples.

Design decisions (see report, Section 3.1):
* MTSamples files the *same* transcription under several categories (e.g. "Surgery" and
  "Orthopedic"). Of 4,966 non-empty rows only 2,357 transcriptions are unique. A naive row-level
  split therefore leaks identical documents between train and test. We group by text first.
* Several categories describe a *document type* or *modality* rather than a clinical specialty
  (Surgery, Consult, SOAP, Discharge Summary, Radiology, ...). They are removed from each
  document's label set; a document is kept only if exactly one specialty remains.
* Specialties with fewer than ``MIN_DOCS`` unique documents are excluded from the in-distribution
  task and their documents form an out-of-distribution (OOD) evaluation set.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mtsamples.csv"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
MIN_DOCS = 50

GENERIC_LABELS = {
    "Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
    "Office Notes", "Letters", "Emergency Room Reports", "IME-QME-Work Comp etc.",
    "General Medicine", "Radiology",
}

# --------------------------------------------------------------------------------------------
# Identifier screening is a defensive safeguard; complete de-identification has not been
# independently established. Counts of replacements are reported, not certified PHI counts.
# --------------------------------------------------------------------------------------------
PHI_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("[PHONE]", re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("[DATE]", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("[DATE]", re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b")),
    ("[ID]", re.compile(r"\b(?:MRN|medical record number|account number|SSN)[:#\s]*\d+\b", re.I)),
    ("[ID]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[NAME]", re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?")),
]

# Fixed clinical abbreviation dictionary; expansions can still be context-dependent.
ABBREVIATIONS = {
    "htn": "hypertension", "dm": "diabetes mellitus", "cad": "coronary artery disease",
    "chf": "congestive heart failure", "copd": "chronic obstructive pulmonary disease",
    "gerd": "gastroesophageal reflux disease", "uti": "urinary tract infection",
    "cva": "cerebrovascular accident", "tia": "transient ischemic attack",
    "dvt": "deep vein thrombosis", "afib": "atrial fibrillation", "bph": "benign prostatic hyperplasia",
    "sob": "shortness of breath", "nkda": "no known drug allergies", "hpi": "history of present illness",
    "pmh": "past medical history", "ckd": "chronic kidney disease", "esrd": "end stage renal disease",
    "aki": "acute kidney injury", "ekg": "electrocardiogram", "ecg": "electrocardiogram",
    "cabg": "coronary artery bypass graft", "ptca": "percutaneous transluminal coronary angioplasty",
    "turp": "transurethral resection of the prostate", "egd": "esophagogastroduodenoscopy",
    "ercp": "endoscopic retrograde cholangiopancreatography", "orif": "open reduction internal fixation",
    "acl": "anterior cruciate ligament", "tah": "total abdominal hysterectomy",
    "bso": "bilateral salpingo oophorectomy", "iol": "intraocular lens", "iop": "intraocular pressure",
    "uri": "upper respiratory infection", "loc": "loss of consciousness",
}
_ABBR_RE = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r")\b", re.I)


def scrub_phi(text: str, counter: dict[str, int] | None = None) -> str:
    for token, pat in PHI_PATTERNS:
        text, n = pat.subn(token, text)
        if counter is not None and n:
            counter[token] = counter.get(token, 0) + n
    return text


def normalize(text: str, counter: dict[str, int] | None = None) -> str:
    # MTSamples encodes section breaks as ",HEADER:," and list items as ",1.  "
    text = re.sub(r"\s*,\s*,", ". ", text)
    text = re.sub(r",(?=[A-Z][A-Z /&-]{2,}:)", " ", text)  # ",HISTORY:" -> " HISTORY:"
    text = re.sub(r"\.,", ". ", text)

    def _expand(m: re.Match) -> str:
        if counter is not None:
            counter["abbreviations_expanded"] = counter.get("abbreviations_expanded", 0) + 1
        return ABBREVIATIONS[m.group(1).lower()]

    text = _ABBR_RE.sub(_expand, text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------------------------
# Demographic proxies (MTSamples has no structured demographics). Inferred from text cues; this
# is a known limitation and is stated in the report.
# --------------------------------------------------------------------------------------------
_FEMALE = re.compile(r"\b(she|her|hers|herself|female|woman|lady|girl|mrs|ms|miss)\b", re.I)
_MALE = re.compile(r"\b(he|him|his|himself|male|man|gentleman|boy|mr)\b", re.I)
_AGE = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.I)


def infer_sex(text: str) -> str:
    f, m = len(_FEMALE.findall(text)), len(_MALE.findall(text))
    # require >=2 cues and a 2:1 majority to reduce mislabelling (e.g. mentions of a spouse)
    if f >= 2 and f >= 2 * m:
        return "female"
    if m >= 2 and m >= 2 * f:
        return "male"
    return "unknown"


def infer_age_group(text: str) -> str:
    m = _AGE.search(text)
    if not m:
        return "unknown"
    age = int(m.group(1))
    if age > 110:
        return "unknown"
    return "<18" if age < 18 else ("18-64" if age < 65 else "65+")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    ood: pd.DataFrame
    labels: list[str]
    stats: dict


def build_dataset(save: bool = True) -> Splits:
    df = pd.read_csv(RAW)
    stats: dict = {"raw_rows": int(len(df)), "raw_categories": int(df.medical_specialty.nunique())}
    df = df.dropna(subset=["transcription"]).copy()
    stats["rows_with_text"] = int(len(df))
    df["medical_specialty"] = df.medical_specialty.str.strip()
    df["raw_text"] = df.transcription.str.strip()

    grouped = df.groupby("raw_text").medical_specialty.agg(lambda s: sorted(set(s))).reset_index()
    stats["unique_transcriptions"] = int(len(grouped))
    grouped["specialties"] = grouped.medical_specialty.apply(lambda v: [x for x in v if x not in GENERIC_LABELS])
    n_spec = grouped.specialties.str.len()
    stats["docs_by_n_specialties"] = {int(k): int(v) for k, v in n_spec.value_counts().sort_index().items()}

    single = grouped[n_spec == 1].copy()
    single["label"] = single.specialties.str[0]
    counts = single.label.value_counts()
    keep = sorted(counts[counts >= MIN_DOCS].index)
    stats["kept_classes"] = {k: int(counts[k]) for k in keep}

    phi_counter: dict[str, int] = {}
    norm_counter: dict[str, int] = {}

    def _prep(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["text"] = [normalize(scrub_phi(t, phi_counter), norm_counter) for t in frame.raw_text]
        frame["sex"] = frame.raw_text.apply(infer_sex)
        frame["age_group"] = frame.raw_text.apply(infer_age_group)
        frame["n_words"] = frame.text.str.split().str.len()
        return frame

    ind = _prep(single[single.label.isin(keep)])
    # After normalisation re-check exact duplicates (none expected, verified below)
    stats["post_norm_duplicates"] = int(ind.text.duplicated().sum())
    ind = ind.drop_duplicates("text")
    ood = _prep(single[~single.label.isin(keep)])
    stats["ood_docs"] = int(len(ood))
    stats["ood_classes"] = {k: int(v) for k, v in ood.label.value_counts().items()}
    stats["phi_replacements"] = phi_counter
    stats["normalization"] = norm_counter

    train, test = train_test_split(ind, test_size=0.20, stratify=ind.label, random_state=SEED)
    train, val = train_test_split(train, test_size=0.125, stratify=train.label, random_state=SEED)  # 70/10/20
    stats["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    stats["class_counts_train"] = {k: int(v) for k, v in train.label.value_counts().items()}

    cols = ["text", "label", "sex", "age_group", "n_words"]
    out = Splits(train[cols].reset_index(drop=True), val[cols].reset_index(drop=True),
                 test[cols].reset_index(drop=True), ood[cols].reset_index(drop=True), keep, stats)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test", "ood"):
            getattr(out, name).to_csv(PROCESSED / f"{name}.csv", index=False)
        (PROCESSED / "labels.json").write_text(json.dumps(keep, indent=2))
        (PROCESSED / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return out


def load_splits() -> Splits:
    if not (PROCESSED / "train.csv").exists():
        return build_dataset()
    labels = json.loads((PROCESSED / "labels.json").read_text())
    stats = json.loads((PROCESSED / "dataset_stats.json").read_text())
    rd = lambda n: pd.read_csv(PROCESSED / f"{n}.csv")
    return Splits(rd("train"), rd("val"), rd("test"), rd("ood"), labels, stats)


if __name__ == "__main__":
    s = build_dataset()
    print(json.dumps(s.stats, indent=2))
    print(s.train.sex.value_counts(), s.train.age_group.value_counts())
    print(s.train.text.iloc[0][:500])
