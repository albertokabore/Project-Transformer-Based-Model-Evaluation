"""Tokenizer / vocabulary statistics for each transformer (measured, not quoted).

* vocabulary size and model configuration (layers, hidden size, parameters)
* fertility = word-pieces per whitespace word on the training notes (lower = vocabulary fits
  the domain better), share of words split into >1 piece, and share of a curated list of
  clinical terms kept as a single token
* tokens per document -> share of notes truncated at 256 tokens
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common import TABLES
from .data import load_splits
from .transformer import CHECKPOINTS, MAX_LEN, get_tokenizer

CLINICAL_TERMS = ["hypertension", "cholecystectomy", "colonoscopy", "echocardiogram", "laparoscopic", "cataract",
                  "tonsillectomy", "hematuria", "creatinine", "dialysis", "lymphoma", "chemotherapy", "arthroscopy",
                  "meniscus", "epidural", "radiculopathy", "seizure", "migraine", "hysterectomy", "cesarean",
                  "prostatectomy", "cystoscopy", "tympanostomy", "sinusitis", "glaucoma", "vitrectomy", "anemia",
                  "esophagogastroduodenoscopy", "angioplasty", "myocardial"]


def main() -> pd.DataFrame:
    from transformers import AutoConfig, AutoTokenizer

    s = load_splits()
    texts = s.train.text.tolist()
    words = [w for t in texts for w in re.findall(r"[A-Za-z]+", t)]
    rng = np.random.default_rng(0)
    sample = list(rng.choice(words, size=min(50000, len(words)), replace=False))
    rows = []
    for key, ckpt in CHECKPOINTS.items():
        tok = get_tokenizer(key)
        cfg = AutoConfig.from_pretrained(ckpt)
        # leading space for byte-level BPE (RoBERTa) so words are tokenised as mid-sentence words
        pieces = [len(tok.tokenize((" " if "roberta" in key else "") + w)) for w in sample]
        terms_single = [len(tok.tokenize((" " if "roberta" in key else "") + w)) == 1 for w in CLINICAL_TERMS]
        ntok = [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
        rows.append({"model": key, "checkpoint": ckpt, "vocab_size": tok.vocab_size, "lowercase input": bool(tok.backend_tokenizer.normalizer.__getstate__().find(b'"lowercase":true') >= 0) if tok.backend_tokenizer.normalizer else False,
                     "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
                     "fertility (pieces/word)": float(np.mean(pieces)), "% words split": 100 * float(np.mean(np.array(pieces) > 1)),
                     f"clinical terms kept whole (/{len(CLINICAL_TERMS)})": int(np.sum(terms_single)),
                     "median tokens/note": float(np.median(ntok)), f"% notes > {MAX_LEN} tokens": 100 * float(np.mean(np.array(ntok) > MAX_LEN))})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "tokenizer_stats.csv", index=False)
    print(df.to_string())
    return df


if __name__ == "__main__":
    main()
