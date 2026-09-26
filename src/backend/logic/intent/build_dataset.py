"""Build train/test CSVs from the seed utterances.

Order matters: split the seeds first, then augment only the training side, so no
augmented copy of a test sentence can leak into training.

Run from src/backend:  python -m logic.intent.build_dataset
"""
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from logic.intent.preprocess import clean_text
from logic.intent.seed_data import SEED_UTTERANCES

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLINC_DIR = DATA_DIR / "external" / "clinc150_oos"
SEED = 42
CLINC_TRAIN_SAMPLE = 70  # keeps out_of_scope roughly the size of the other classes
AUGMENT_PER_SEED = 5

PREFIXES = ["hi, ", "hello, ", "please help: ", "excuse me, ", "sir, ", "madam, ", "hey, ", "quick question: ", "urgent: "]
SUFFIXES = [" please", " thanks", " thank you", " asap", " kindly guide me", " any help is appreciated"]


def add_typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["swap", "drop", "dup"])
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def noisy(text: str, rng: random.Random, typo_rate: float = 0.12) -> str:
    words = [add_typo(w, rng) if rng.random() < typo_rate else w for w in text.split()]
    out = " ".join(words)
    if rng.random() < 0.5:
        out = out.lower()
    if rng.random() < 0.5:
        out = out.rstrip("?.!")
    return out


def augment(text: str, rng: random.Random) -> str:
    out = text
    if rng.random() < 0.5:
        out = rng.choice(PREFIXES) + out[0].lower() + out[1:]
    if rng.random() < 0.4:
        out = out.rstrip("?.!") + rng.choice(SUFFIXES)
    return noisy(out, rng, typo_rate=0.08)


def read_clinc(split: str) -> list[str]:
    return [line.strip() for line in (CLINC_DIR / f"{split}.txt").read_text().splitlines() if line.strip()]


def main() -> None:
    rng = random.Random(SEED)
    rows = [(t, label) for label, texts in SEED_UTTERANCES.items() for t in texts]
    df = pd.DataFrame(rows, columns=["text", "intent"]).drop_duplicates("text")

    train_seed, test = train_test_split(df, test_size=0.25, stratify=df["intent"], random_state=SEED)

    train_rows = []
    for text, label in train_seed.itertuples(index=False):
        train_rows.append((text, label, "original"))
        for _ in range(AUGMENT_PER_SEED):
            train_rows.append((augment(text, rng), label, "augmented"))
    # Real out-of-scope queries from CLINC150 (train + valid pools); its test pool is kept for evaluation only.
    clinc_pool = read_clinc("train") + read_clinc("valid")
    train_rows += [(t, "out_of_scope", "clinc150") for t in rng.sample(clinc_pool, CLINC_TRAIN_SAMPLE)]
    train = pd.DataFrame(train_rows, columns=["text", "intent", "source"]).drop_duplicates("text")

    test = test.assign(source="original")
    test_noisy = pd.DataFrame(
        [(noisy(t, rng, typo_rate=0.2), label, "noisy") for t, label in test[["text", "intent"]].itertuples(index=False)],
        columns=["text", "intent", "source"],
    )

    test_clinc_oos = pd.DataFrame(
        [(t, "out_of_scope", "clinc150") for t in read_clinc("test")], columns=["text", "intent", "source"]
    )

    for name, frame in [("train", train), ("test", test), ("test_noisy", test_noisy), ("test_clinc_oos", test_clinc_oos)]:
        frame = frame.assign(clean_text=frame["text"].map(clean_text))
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")

    print("\nTrain class counts:\n", train["intent"].value_counts().to_string())
    print("\nTest class counts:\n", test["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
