"""Week 4 checks: gold NER set integrity, joint intent adapter, span scoring, and the improved copilot (if trained)."""
import numpy as np
import pandas as pd
import pytest
import yaml

from nlp import DATA
from nlp.intent.finetune import JointIntentAdapter
from nlp.pipeline.compare_week4 import span_scores


def test_gold_ner_spans_exist_in_test_queries():
    gold = yaml.safe_load((DATA / "annotations" / "ner_gold.yaml").read_text())["gold"]
    test = pd.read_csv(DATA / "final" / "test.csv").set_index("query_id")["query"]
    assert len(gold) == 120
    assert all(qid in test.index for qid in gold)
    assert all(str(span) in test[qid] for qid, spans in gold.items() for span, _ in spans)


def test_joint_adapter_sums_sub_intents():
    class Joint:
        classes_ = np.array(["A1", "A2", "B1", "out_of_scope"])

        def predict_proba(self, X):
            return np.array([[0.3, 0.3, 0.35, 0.05]] * len(X))

    adapter = JointIntentAdapter(Joint(), {"A1": "A", "A2": "A", "B1": "B"})
    assert list(adapter.classes_) == ["A", "B", "out_of_scope"]
    assert adapter.predict(["q"])[0] == "A"            # 0.6 for A beats 0.35 for B
    assert np.allclose(adapter.predict_proba(["q"]).sum(1), 1)


def test_span_scores_strict_and_relaxed():
    gold = [[(0, 10, "DOCUMENT"), (15, 20, "ISSUE")]]
    pred = [[(0, 10, "DOCUMENT"), (14, 20, "ISSUE")]]
    s = span_scores(pred, gold)
    assert s["strict_f1"] == 0.5 and s["relaxed_f1"] == 1.0


@pytest.fixture(scope="module")
def copilot_v3():
    from nlp.pipeline.copilot import V3_DIR, Copilot
    if not (V3_DIR / "meta.json").exists():
        pytest.skip("week 4 models not trained; run python -m nlp.pipeline.compare_week4")
    return Copilot.from_disk("v3")


def test_improved_copilot_end_to_end(copilot_v3):
    r = copilot_v3.run("how do i apply for revaluation of my TEE paper")
    assert r.intent == "EXAMINATIONS" and r.service_request["department"] and r.sources
    assert copilot_v3.run("can you recommend a good movie to watch tonight").intent == "out_of_scope"
