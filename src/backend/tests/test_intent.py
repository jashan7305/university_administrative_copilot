"""Checks for the preprocessing pipeline and the integrity of the generated splits."""
import re

import pandas as pd
import pytest

from logic.intent.preprocess import DEFAULT, Preprocessor, SpellCorrector

DATA = "data/processed"


@pytest.fixture(scope="module")
def prep():
    return Preprocessor(DEFAULT, SpellCorrector(["certificate certificate hostel hostel attendance attendance"]))


@pytest.mark.parametrize("text, expected", [
    ("I can't pay", "i cannot pay"),
    ("pls send NOC", "please send no objection certificate"),
    ("attendance is 68%", "attendance is <percent>"),
    ("paid Rs. 45,000 on 12/08/2026", "pay <money> on <date>"),
    ("call 9876543210", "call <phone>"),
    ("roll no J054", "roll no <rollno>"),
    ("hostle certficate", "hostel certificate"),
    ("has it been credited as promised", "has it been credit as promise"),
])
def test_preprocessing(prep, text, expected):
    assert prep(text) == expected


def test_spell_corrector_leaves_valid_english(prep):
    assert prep("call") == "call"


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_csv(f"{DATA}/{s}.csv") for s in ("train", "val", "test", "test_seed", "test_noisy")}


def key(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


@pytest.mark.parametrize("split", ["val", "test", "test_seed"])
def test_no_text_leakage(splits, split):
    assert not set(splits["train"]["text"].map(key)) & set(splits[split]["text"].map(key))


def test_frames_are_disjoint(splits):
    tpl = lambda df: set(df.loc[df["source"] == "template", "group"])
    assert not tpl(splits["train"]) & tpl(splits["test"])
    assert not tpl(splits["train"]) & tpl(splits["val"])


def test_all_classes_present(splits):
    for name in ("train", "val", "test", "test_seed"):
        assert splits[name]["intent"].nunique() == 10, name
