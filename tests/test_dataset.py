"""Dataset checks: every phase-4 validation check passes on the released files, and the release has the agreed shape."""
import pandas as pd
import pytest

from dataset.config import FINAL, load_facts
from dataset.phase4_validate import run_checks


@pytest.fixture(scope="module")
def release():
    return {n: pd.read_csv(FINAL / f"{n}.csv") for n in
            ("nmims_admin_queries", "nmims_ner_annotations", "source_inventory", "train", "validation", "test")}


def test_all_validation_checks_pass(release):
    checks = run_checks(release["nmims_admin_queries"], release["nmims_ner_annotations"], release["source_inventory"], load_facts())
    failed = [(name, n) for name, n, _ in checks if n]
    assert not failed, failed


def test_split_files_partition_the_dataset(release):
    q = release["nmims_admin_queries"]
    assert sum(len(release[s]) for s in ("train", "validation", "test")) == len(q)
    assert not set(release["train"]["group_id"]) & set(release["test"]["group_id"])


def test_every_verified_query_cites_a_fact(release):
    q = release["nmims_admin_queries"]
    assert q.loc[q["source_status"] == "VERIFIED", "fact_ids"].notna().all()
