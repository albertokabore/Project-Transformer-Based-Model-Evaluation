"""Deterministic text perturbations for robustness and counterfactual-fairness testing.

* ``typo``      : character-level noise (swap / delete / insert / substitute) on a fraction of words,
                  simulating transcription / OCR / speech-recognition errors.
* ``word_drop`` : random word deletion, simulating incomplete or terse documentation.
* ``keyword_mask``: masks the most class-discriminative unigrams learned by the TF-IDF Logistic
                  Regression on the training set. It is a *white-box* attack for that model and a
                  *transfer (black-box)* attack for all other models -- this asymmetry is reported.
* ``gender_swap``: swaps gendered words (he<->she, male<->female, Mr<->Mrs ...). Used to measure the
                  counterfactual prediction-flip rate (a fairness probe), not accuracy.
"""
from __future__ import annotations

import random
import re
import string

SEED = 13


def _typo_word(w: str, rng: random.Random) -> str:
    if len(w) < 4 or not w.isalpha():
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(("swap", "delete", "insert", "substitute"))
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "delete":
        return w[:i] + w[i + 1:]
    c = rng.choice(string.ascii_lowercase)
    if op == "insert":
        return w[:i] + c + w[i:]
    return w[:i] + c + w[i + 1:]


def typo(text: str, rate: float, seed: int = SEED) -> str:
    rng = random.Random(f"{seed}-{rate}-{len(text)}-{text[:40]}")
    return " ".join(_typo_word(w, rng) if rng.random() < rate else w for w in text.split())


def word_drop(text: str, rate: float, seed: int = SEED) -> str:
    rng = random.Random(f"{seed}-drop-{rate}-{len(text)}-{text[:40]}")
    kept = [w for w in text.split() if rng.random() >= rate]
    return " ".join(kept) if kept else text


def keyword_mask(text: str, keywords: set[str]) -> str:
    def _sub(m: re.Match) -> str:
        return "___" if m.group(0).lower() in keywords else m.group(0)

    return re.sub(r"[A-Za-z]+", _sub, text)


_SWAP = {
    "he": "she", "she": "he", "him": "her", "his": "her", "her": "his", "hers": "his",
    "himself": "herself", "herself": "himself", "male": "female", "female": "male",
    "man": "woman", "woman": "man", "men": "women", "women": "men", "boy": "girl", "girl": "boy",
    "gentleman": "lady", "lady": "gentleman", "mr": "mrs", "mrs": "mr", "ms": "mr",
    "father": "mother", "mother": "father", "son": "daughter", "daughter": "son",
    "husband": "wife", "wife": "husband", "brother": "sister", "sister": "brother",
}
_SWAP_RE = re.compile(r"\b(" + "|".join(_SWAP) + r")\b", re.I)


def gender_swap(text: str) -> str:
    def _sub(m: re.Match) -> str:
        w = m.group(0)
        r = _SWAP[w.lower()]
        if w.isupper():
            return r.upper()
        if w[0].isupper():
            return r.capitalize()
        return r

    return _SWAP_RE.sub(_sub, text)


def build_perturbations(texts: list[str], keywords: set[str]) -> dict[str, list[str]]:
    return {
        "typo_10": [typo(t, 0.10) for t in texts],
        "typo_25": [typo(t, 0.25) for t in texts],
        "word_drop_20": [word_drop(t, 0.20) for t in texts],
        "keyword_mask": [keyword_mask(t, keywords) for t in texts],
        "gender_swap": [gender_swap(t) for t in texts],
    }
