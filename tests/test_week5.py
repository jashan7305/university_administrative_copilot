"""Week 5 checks: challenge set labels, deterministic perturbations, evaluation statistics, typo normaliser."""
import numpy as np

from dataset.config import SUB_TO_INTENT, TAXONOMY
from nlp.pipeline.evaluate_week5 import PERTURBATIONS, SpellNormalizer, bootstrap, ece, load_challenge, mcnemar, perturb


def test_challenge_labels_are_valid():
    ch = load_challenge()
    assert len(ch) >= 90
    for intents, subs in zip(ch["intents"], ch["subs"]):
        assert intents and all(i == "out_of_scope" or i in TAXONOMY for i in intents)
        assert all(SUB_TO_INTENT[s] in intents for s in subs)


def test_perturbations_are_deterministic_and_change_text():
    q = "How do I apply for revaluation of my examination paper?"
    for name in PERTURBATIONS:
        assert perturb(name, q) == perturb(name, q)
        assert perturb(name, q) != q


def test_bootstrap_ci_contains_mean():
    v = np.array([1] * 80 + [0] * 20)
    lo, hi = bootstrap(v)
    assert lo <= 0.8 <= hi and hi - lo < 0.25


def test_ece_and_mcnemar():
    assert ece([1.0, 1.0, 0.5, 0.5], [1, 1, 1, 0]) == 0.0
    assert ece([0.9] * 10, [0] * 10) > 0.8
    assert mcnemar([1, 1, 0, 0], [1, 1, 0, 0])["p_value"] == 1.0


def test_spell_normalizer_fixes_typos_only():
    norm = SpellNormalizer(["apply for revaluation"] * 3 + ["bonafide certificate"] * 3)
    assert norm("aply for revalutaion") == "apply for revaluation"
    assert norm("bonafide certificate") == "bonafide certificate"
