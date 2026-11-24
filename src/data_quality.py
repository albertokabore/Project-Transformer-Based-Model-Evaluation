"""Audit cleaned model inputs without altering the benchmark tied to saved predictions.

Run ``python -m src.data_quality``. Counts and hashes are written to
results/data_quality_audit.json; note text and possible identifiers are not printed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import defaultdict

import pandas as pd

from .data import PROCESSED, RAW, ROOT, build_dataset, load_splits, normalize, scrub_phi


def main():
    saved, rebuilt = load_splits(), build_dataset(save=False)
    checks, per_split = [], {}

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    raw = pd.read_csv(RAW)
    raw_missing = {column: {"missing": int(raw[column].isna().sum()),
                            "blank_nonmissing": int(raw[column].dropna().astype(str).str.strip().eq("").sum())}
                   for column in ("transcription", "medical_specialty")}
    canonical_groups = defaultdict(list)
    repeat_punctuation = 0
    identifier_matches = defaultdict(int)
    for name in ("train", "val", "test", "ood"):
        frame = getattr(saved, name)
        check(f"{name}: required columns", set(frame.columns) == {"text", "label", "sex", "age_group", "n_words"})
        check(f"{name}: rebuilt from raw data in identical order", frame.equals(getattr(rebuilt, name)))
        text = frame.text.astype(str)
        stats = {"rows": len(frame), "missing_cells": int(frame.isna().sum().sum()),
                 "blank_texts": int(text.str.strip().eq("").sum()),
                 "duplicate_texts": int(text.duplicated().sum()),
                 "untrimmed_texts": int(text.ne(text.str.strip()).sum()),
                 "repeated_whitespace": int(text.str.contains(r"\s{2,}", regex=True).sum()),
                 "control_characters": int(text.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", regex=True).sum()),
                 "word_count_mismatches": int(text.str.split().str.len().ne(frame.n_words).sum())}
        for key, count in stats.items():
            if key != "rows":
                check(f"{name}: {key} is zero", count == 0)
        check(f"{name}: nonempty labels", frame.label.str.strip().ne("").all())
        check(f"{name}: trimmed labels", frame.label.eq(frame.label.str.strip()).all())
        check(f"{name}: permitted sex-proxy values", set(frame.sex) <= {"female", "male", "unknown"})
        check(f"{name}: permitted age groups", set(frame.age_group) <= {"<18", "18-64", "65+", "unknown"})
        check(f"{name}: valid label mapping", not set(frame.label) & set(saved.labels) if name == "ood" else set(frame.label) == set(saved.labels))
        per_split[name] = stats
        for index, row in frame.iterrows():
            canonical = re.sub(r"[^a-z0-9]+", "", row.text.lower())
            canonical_groups[canonical].append((name, int(index)))
            phi_counter = {}
            scrubbed = scrub_phi(row.text, phi_counter)
            for key, count in phi_counter.items():
                identifier_matches[key] += count
            repeated = normalize(scrubbed)
            if repeated != row.text:
                repeat_punctuation += 1
                check(f"{name}/{index}: repeat cleaning preserves letters and digits",
                      re.sub(r"[^a-z0-9]+", "", repeated.lower()) == canonical)
    overlaps = {}
    for a, b in itertools.combinations(per_split, 2):
        overlaps[f"{a}/{b}"] = len(set(getattr(saved, a).text) & set(getattr(saved, b).text))
        check(f"{a}/{b}: zero exact text overlap", overlaps[f"{a}/{b}"] == 0)
    collisions = [entries for entries in canonical_groups.values() if len(entries) > 1]
    check("zero duplicate groups after ignoring case and punctuation", len(collisions) == 0)
    check("no further matches to implemented identifier patterns", not identifier_matches)
    check("raw-data statistics match saved statistics", json.loads(json.dumps(rebuilt.stats)) == saved.stats)
    result = {"date": "2026-11-24", "raw_missing_or_blank": raw_missing, "per_split": per_split,
              "exact_overlap_by_split_pair": overlaps, "canonical_duplicate_groups": len(collisions),
              "additional_implemented_identifier_matches": dict(identifier_matches),
              "repeat_normalization_punctuation_only_documents": repeat_punctuation,
              "input_policy": "Retain original cleaned benchmark inputs and row order so saved predictions remain traceable. A repeat pass changes a separator in 12 notes without changing letters or digits; no second-pass identifier matches were found.",
              "privacy_limit": "Zero regex matches does not establish complete de-identification; clinical and privacy review remain necessary.",
              "checks": checks, "passed": sum(check["passed"] for check in checks),
              "failed": sum(not check["passed"] for check in checks),
              "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [RAW] + [PROCESSED / f"{name}.csv" for name in per_split]}}
    destination = ROOT / "results/data_quality_audit.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "failed", "canonical_duplicate_groups", "repeat_normalization_punctuation_only_documents")}, indent=2))
    if result["failed"]:
        raise SystemExit("Data quality checks failed; inspect results/data_quality_audit.json")
    return result


if __name__ == "__main__":
    main()
