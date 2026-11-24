"""Transformer models: partial fine-tuning (top-K encoder layers + classification head).

Compute constraint: the study ran on an 8-core/8-thread Intel i3-N305 CPU without a GPU. Full
fine-tuning of a 110M-parameter encoder at 256 tokens measured 40-67 s per batch of 16 (~11 h for
four models, one configuration each). We therefore freeze the embeddings and the bottom
``12-K`` layers, cache their output once, and fine-tune only the top ``K=2`` layers and a
mean-pooling linear head. The lower layers run in evaluation mode; float16 cached states add
rounding. This approximates the same network with fixed lower layers in evaluation mode,
and cut a training step to ~5-6 s. It is not a guarantee about full fine-tuning quality.

Training protocol (per model): class-weighted cross-entropy, AdamW (wd=0.01), linear warm-up (10%)
+ linear decay, gradient clipping (max-norm 1.0), batch 16, early stopping on validation macro-F1
(patience 2, max 6 epochs), learning-rate grid {3e-5, 1e-4}; best configuration by validation
macro-F1. Each epoch is logged to MLflow.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .common import CACHE, MODELS, ROOT, RESULTS, mlflow_setup, save_json, save_probs
from .data import SEED, load_splits

CHECKPOINTS = {
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "roberta": "FacebookAI/roberta-base",
}
# The HF tokenizer configs of these two *cased* checkpoints lowercase their input by default, which
# leaves 7,964 capitalised vocabulary entries unused. A masked-LM check on training notes gave lower
# loss with cased input (BioBERT 1.348 vs 1.655; Bio_ClinicalBERT 1.371 vs 1.655; results/
# mlm_casing_check.json), so casing is preserved for them.
TOKENIZER_KWARGS = {"clinicalbert": {"do_lower_case": False}, "biobert": {"do_lower_case": False}}
MAX_LEN = 256
TOP_K = 2
N_LAYERS = 12
BATCH = 16
MAX_EPOCHS = 6
PATIENCE = 2
LR_GRID = [3e-5, 1e-4]
PERTURBED = ["typo_10", "typo_25", "word_drop_20", "keyword_mask", "gender_swap"]
torch.set_num_threads(8)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "transformer_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_tokenizer(key: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CHECKPOINTS[key], **TOKENIZER_KWARGS.get(key, {}))


def ext_mask(mask: torch.Tensor) -> torch.Tensor:
    return (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min


class TopClassifier(nn.Module):
    """Top-K transformer layers + masked mean pooling + dropout + linear classifier."""

    def __init__(self, layers: list[nn.Module], hidden: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def encode(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        em = ext_mask(mask)
        for lyr in self.layers:
            o = lyr(h, attention_mask=em)
            h = o[0] if isinstance(o, tuple) else o
        return h

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.pool(self.encode(h, mask), mask)))


class FullClassifier(nn.Module):
    """Deployable end-to-end model: pretrained encoder with fine-tuned top layers + head."""

    def __init__(self, key: str, n_classes: int):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = get_tokenizer(key)
        self.encoder = AutoModel.from_pretrained(CHECKPOINTS[key], attn_implementation="eager")
        top = torch.load(MODELS / f"{key}_top.pt", map_location="cpu", weights_only=True)
        for i in range(TOP_K):
            sd = {k.split(".", 2)[2]: v for k, v in top.items() if k.startswith(f"layers.{i}.")}
            self.encoder.encoder.layer[N_LAYERS - TOP_K + i].load_state_dict(sd)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)
        self.head.load_state_dict({"weight": top["head.weight"], "bias": top["head.bias"]})
        self.eval()

    def forward(self, input_ids, attention_mask, output_attentions: bool = False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.head(pooled)
        return (logits, out.attentions) if output_attentions else logits

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch: int = 8) -> np.ndarray:
        res = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(list(texts[i:i + batch]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            res.append(F.softmax(self(enc["input_ids"], enc["attention_mask"]), -1).numpy())
        return np.concatenate(res)


# ------------------------------------------------------------------------------------------
# Caching frozen states
# ------------------------------------------------------------------------------------------
def split_texts() -> dict[str, list[str]]:
    s = load_splits()
    pert = pd.read_csv(ROOT / "data" / "processed" / "test_perturbed.csv")
    d = {"train": s.train.text.tolist(), "val": s.val.text.tolist(), "test": s.test.text.tolist(), "ood": s.ood.text.tolist()}
    d.update({p: pert[p].tolist() for p in PERTURBED})
    return d


@torch.no_grad()
def cache_states(key: str) -> None:
    from transformers import AutoModel, AutoTokenizer

    out_dir = CACHE / key
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(key)
    model = AutoModel.from_pretrained(CHECKPOINTS[key]).eval()
    for split, texts in split_texts().items():
        if (out_dir / f"{split}_pooled.npy").exists():
            continue
        t0 = time.time()
        n, H = len(texts), model.config.hidden_size
        hs = np.lib.format.open_memmap(out_dir / f"{split}_h.npy", mode="w+", dtype=np.float16, shape=(n, MAX_LEN, H))
        masks = np.zeros((n, MAX_LEN), dtype=np.uint8)
        pooled = np.zeros((n, H), dtype=np.float32)
        ntok = np.zeros(n, dtype=np.int32)
        for i in range(0, n, BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            full = tok(texts[i:i + BATCH], truncation=False, add_special_tokens=True)["input_ids"]
            out = model(**{k: enc[k] for k in ("input_ids", "attention_mask")}, output_hidden_states=True)
            mk = enc["attention_mask"]
            hs[i:i + BATCH] = out.hidden_states[N_LAYERS - TOP_K].numpy().astype(np.float16)
            masks[i:i + BATCH] = mk.numpy()
            m = mk.unsqueeze(-1).float()
            pooled[i:i + BATCH] = ((out.last_hidden_state * m).sum(1) / m.sum(1)).numpy()
            ntok[i:i + BATCH] = [len(f) for f in full]
        hs.flush()
        del hs
        np.save(out_dir / f"{split}_mask.npy", masks)
        np.save(out_dir / f"{split}_ntok.npy", ntok)
        np.save(out_dir / f"{split}_pooled.npy", pooled)  # frozen pretrained embedding (for CV probes)
        log(f"{key}: cached {split} ({n} docs) in {time.time() - t0:.0f}s")


def load_cached(key: str, split: str):
    d = CACHE / key
    return np.load(d / f"{split}_h.npy", mmap_mode="r"), np.load(d / f"{split}_mask.npy")


# ------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model: TopClassifier, h, mask, batch: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(mask), batch):
        hb = torch.from_numpy(np.asarray(h[i:i + batch], dtype=np.float32))
        out.append(F.softmax(model(hb, torch.from_numpy(mask[i:i + batch]).long()), -1).numpy())
    return np.concatenate(out)


def train_one(key: str, lr: float, data: dict, n_classes: int, class_w: torch.Tensor, mlflow) -> dict:
    from transformers import AutoModel

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = AutoModel.from_pretrained(CHECKPOINTS[key])
    model = TopClassifier([copy.deepcopy(l) for l in base.encoder.layer[N_LAYERS - TOP_K:]], base.config.hidden_size, n_classes)
    del base
    htr, mtr, ytr = data["train"]
    hva, mva, yva = data["val"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(ytr) / BATCH))
    total, warm = steps_per_epoch * MAX_EPOCHS, int(0.1 * steps_per_epoch * MAX_EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    rng = np.random.default_rng(SEED)
    best = {"val_macro_f1": -1.0}
    bad, step = 0, 0
    history = []
    with mlflow.start_run(run_name=f"{key}_lr{lr:g}"):
        mlflow.log_params({"model": key, "checkpoint": CHECKPOINTS[key], "lr": lr, "batch_size": BATCH, "max_len": MAX_LEN,
                           "trainable_top_layers": TOP_K, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                           "grad_clip": 1.0, "weight_decay": 0.01, "warmup_frac": 0.1, "loss": "class-weighted CE",
                           "trainable_params": sum(p.numel() for p in params)})
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            t0, losses, gnorms = time.time(), [], []
            for idx in np.array_split(rng.permutation(len(ytr)), steps_per_epoch):
                idx = np.sort(idx)
                hb = torch.from_numpy(np.asarray(htr[idx], dtype=np.float32))
                loss = F.cross_entropy(model(hb, torch.from_numpy(mtr[idx]).long()), torch.from_numpy(ytr[idx]), weight=class_w)
                opt.zero_grad()
                loss.backward()
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
                opt.step()
                sched.step()
                step += 1
                losses.append(loss.item())
            pva = predict(model, hva, mva)
            vf1 = f1_score(yva, pva.argmax(1), average="macro")
            vloss = float(F.nll_loss(torch.log(torch.from_numpy(pva) + 1e-12), torch.from_numpy(yva)))
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vloss, "val_macro_f1": vf1,
                   "val_acc": float((pva.argmax(1) == yva).mean()), "grad_norm_mean": float(np.mean(gnorms)),
                   "epoch_time_s": time.time() - t0}
            history.append(rec)
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            log(f"{key} lr={lr:g} epoch {epoch}: loss={rec['train_loss']:.4f} val_f1={vf1:.4f} val_loss={vloss:.4f} ({rec['epoch_time_s']:.0f}s)")
            if vf1 > best["val_macro_f1"]:
                best = {"val_macro_f1": vf1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    log(f"{key} lr={lr:g}: early stopping at epoch {epoch} (best epoch {best['epoch']})")
                    break
        mlflow.log_metrics({"best_val_macro_f1": best["val_macro_f1"], "best_epoch": best["epoch"]})
    best.update({"lr": lr, "history": history})
    return best


def run(key: str) -> None:
    if (RESULTS / "preds" / key / "gender_swap.npy").exists():
        log(f"{key}: already done, skipping")
        return
    s = load_splits()
    lab2id = {l: i for i, l in enumerate(s.labels)}
    C = len(s.labels)
    t0 = time.time()
    cache_states(key)
    cache_s = time.time() - t0
    y = {"train": s.train.label.map(lab2id).values, "val": s.val.label.map(lab2id).values}
    data = {sp: (*load_cached(key, sp), y[sp]) for sp in ("train", "val")}
    data["train"] = (np.asarray(data["train"][0]), data["train"][1], data["train"][2])  # into RAM for speed
    counts = np.bincount(y["train"], minlength=C)
    class_w = torch.tensor(len(y["train"]) / (C * counts), dtype=torch.float32)
    mlflow = mlflow_setup()
    t1 = time.time()
    runs = [train_one(key, lr, data, C, class_w, mlflow) for lr in LR_GRID]
    train_s = time.time() - t1
    best = max(runs, key=lambda r: r["val_macro_f1"])
    from transformers import AutoModel

    base = AutoModel.from_pretrained(CHECKPOINTS[key])
    model = TopClassifier([copy.deepcopy(l) for l in base.encoder.layer[N_LAYERS - TOP_K:]], base.config.hidden_size, C)
    n_params = sum(p.numel() for p in base.parameters()) + model.head.weight.numel() + model.head.bias.numel()
    del base
    model.load_state_dict(best["state"])
    torch.save(best["state"], MODELS / f"{key}_top.pt")
    del data
    for split in ["val", "test", "ood"] + PERTURBED:
        h, m = load_cached(key, split)
        save_probs(key, split, predict(model, h, m))
    ntok = np.load(CACHE / key / "test_ntok.npy")
    save_json({"checkpoint": CHECKPOINTS[key], "best_lr": best["lr"], "best_epoch": best["epoch"],
               "best_val_macro_f1": best["val_macro_f1"], "grid": [{"lr": r["lr"], "best_val_macro_f1": r["val_macro_f1"],
                                                                    "best_epoch": r["epoch"], "history": r["history"]} for r in runs],
               "cache_time_s": cache_s, "train_time_s": train_s, "total_params": int(n_params),
               "trainable_params": int(sum(p.numel() for p in model.parameters())),
               "test_tokens_median": float(np.median(ntok)), "test_frac_truncated": float((ntok > MAX_LEN).mean())},
              RESULTS / f"transformer_{key}.json")
    log(f"{key}: done. best lr={best['lr']:g} epoch={best['epoch']} val_f1={best['val_macro_f1']:.4f}")


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(CHECKPOINTS)):
        run(k)
