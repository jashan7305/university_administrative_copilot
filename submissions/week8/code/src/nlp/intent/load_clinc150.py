"""Load the CLINC150 out-of-scope queries used for the out_of_scope class.

Source: CLINC150 (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and Out-of-Scope
Prediction", EMNLP-IJCNLP), as distributed in https://github.com/jianguoz/Few-Shot-Intent-Detection
(Datasets/CLINC150/oos/{train,valid,test}/seq.in).

Split use (official splits are kept, so no CLINC150 test query is ever trained on):
  official train (100)   -> our train
  official valid (100)   -> first 40 to our val, remaining 60 to our test
  official test  (1,000) -> test_clinc_oos, evaluation only

Usage, from the repo root:
  python -m nlp.intent.load_clinc150            # download if needed, write data/external/clinc150_oos/*.txt
  python -m nlp.intent.load_clinc150 --summary  # also print split sizes
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import random

from nlp import DATA

BASE_URL = "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/CLINC150/oos"
DOMAIN_URL = "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/CLINC150"
OUT_DIR = DATA / "external" / "clinc150_oos"
DOMAIN_DIR = DATA / "external" / "clinc150_other_domains"
SPLITS = {"train": "train", "valid": "valid", "test": "test"}  # local file name -> official split
VAL_TAKE = 40
# CLINC150's regular (in-domain) queries cover banking, travel, home, auto, work and small talk - all outside
# university administration - so a sample of them is real out-of-scope data for this project. Only the official
# train/valid splits are sampled; the official out-of-scope test set above stays untouched for evaluation.
OTHER_DOMAIN_TAKE = {"train": 1500, "val": 200}


def download(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for local, official in SPLITS.items():
        dest = OUT_DIR / f"{local}.txt"
        if dest.exists() and not force:
            continue
        with urllib.request.urlopen(f"{BASE_URL}/{official}/seq.in", timeout=30) as r:
            dest.write_bytes(r.read())
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        dest = DOMAIN_DIR / f"{split}.txt"
        if dest.exists() and not force:
            continue
        with urllib.request.urlopen(f"{DOMAIN_URL}/{split}/seq.in", timeout=60) as r:
            dest.write_bytes(r.read())


def read(split: str) -> list[str]:
    return [line.strip() for line in (OUT_DIR / f"{split}.txt").read_text().splitlines() if line.strip()]


def rows(other_domains: bool = True) -> list[dict]:
    """CLINC150 out-of-scope rows, assigned to our splits as described in the module docstring."""
    row = lambda text, split, src="clinc150": {"text": text, "intent": "out_of_scope", "split": split,
                                               "source": src, "group": f"clinc::{text}", "entities": []}
    valid = read("valid")
    out = ([row(t, "train") for t in read("train")]
           + [row(t, "val") for t in valid[:VAL_TAKE]]
           + [row(t, "test") for t in valid[VAL_TAKE:]]
           + [row(t, "test_clinc_oos") for t in read("test")])
    if other_domains and DOMAIN_DIR.exists():
        rng = random.Random(42)
        for split, file in (("train", "train"), ("val", "valid")):
            lines = [l.strip() for l in (DOMAIN_DIR / f"{file}.txt").read_text().splitlines() if l.strip()]
            out += [row(t, split, "clinc150_other_domain") for t in rng.sample(lines, OTHER_DOMAIN_TAKE[split])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--summary", action="store_true", help="print rows per split")
    args = parser.parse_args()
    download(args.force)
    if args.summary:
        from collections import Counter

        print(dict(Counter(r["split"] for r in rows())))


if __name__ == "__main__":
    main()
