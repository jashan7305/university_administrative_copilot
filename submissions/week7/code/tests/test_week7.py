"""Week 7: elective advisor (catalogue, syllabus and identity parsing, eligibility rules, recommendation), its API,
the partner integration and the demo UI."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from nlp import DATA
from nlp.electives import catalog
from nlp.electives.advisor import build_profile, get_advisor
from nlp.electives.schemas import AdviseInput, StudentInput
from nlp.electives.syllabus import parse_syllabus, semesters_in

AI = DATA / "electives" / "syllabi" / "artificial_intelligence"
client = TestClient(app)


def syllabi(folder=AI):
    return [{"name": f.name, "text": f.read_text(encoding="utf-8")} for f in sorted(folder.glob("*.*"))]


# --------------------------------------------------------------------------- catalogue

def test_catalogue_has_official_elective_slots():
    rows = catalog.semester_rows("B Tech Computer Science", 5)
    assert set(rows[rows["kind"] == "program_elective_slot"]["course"]) == {"Program Elective-I", "Program Elective-II"}
    assert len(rows[rows["kind"] == "open_elective_slot"]) == 2
    assert int(rows["page"].iloc[0]) > 0 and len(catalog.programs()) >= 20


def test_catalogue_reads_en_dash_slots():
    rows = catalog.semester_rows("B Tech Civil Engineering", 6)
    assert (rows["kind"] == "open_elective_slot").sum() == 2


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("text, sems", [("Semester VI", [6]), ("Semesters V and VI", [5, 6]), ("taken in Semester V or VI", [5, 6]),
                                        ("Sem 5-7", [5, 6, 7]), ("Year / Semester: III / VI", [6])])
def test_semesters(text, sems):
    assert semesters_in(text) == sems


def test_parse_formal_outline():
    s = parse_syllabus((AI / "natural_language_processing.md").read_text(encoding="utf-8"))
    assert s.title == "Natural Language Processing" and s.course_type == "program_elective"
    assert set(s.programs) == {"B Tech Computer Engineering", "B Tech Information Technology"}
    assert s.semesters == [6] and s.prerequisites == ["Machine Learning", "Probability and Statistics"]
    assert s.parse_confidence == 1.0 and set(s.found.values()) == {"labelled"}


def test_parse_free_text_new_course():
    s = parse_syllabus((AI / "NEW_generative_ai_and_llm_applications.txt").read_text(encoding="utf-8"))
    assert s.is_new and s.course_type == "program_elective" and s.semesters == [6] and s.credits == 3
    assert set(s.programs) == {"B Tech Computer Science", "B Tech Computer Engineering", "B Tech Artificial Intelligence"}
    assert sorted(s.prerequisites) == ["Data Structures and Algorithms", "Machine Learning"]   # not split at "and"
    assert s.found["programs"] == "inferred" and len(s.topics) >= 5


@pytest.mark.parametrize("identity, program, sems", [
    ("third year student, computer science program", "B Tech Computer Science", [5, 6]),
    ("comps engineering sem 6", "B Tech Computer Engineering", [6]),
    ("B.Tech. IT, 7th semester", "B Tech Information Technology", [7]),
    ("MBA Tech Data Science, 3rd year, sem VI", "MBA Tech Data Science", [6]),
])
def test_identity(identity, program, sems):
    p, _ = build_profile(StudentInput(identity=identity))
    assert p.program == program and p.semesters == sems


def test_unknown_programme_is_not_guessed():
    p, _ = build_profile(StudentInput(identity="third year student, astrophysics program"))
    assert p.program is None and p.confidence.level == "low" and p.notes


# --------------------------------------------------------------------------- advisor

@pytest.fixture(scope="module")
def demo():
    return get_advisor().advise(AdviseInput(student={"identity": "third year student, computer science program"},
                                            syllabi=syllabi()))


def test_demo_recommends_new_course(demo):
    st = {a.name.split(".")[0]: a.status for a in demo.assessments}
    assert demo.recommendation.course.startswith("NEW_generative_ai") and demo.recommendation.semester == 6
    assert st == {"NEW_generative_ai_and_llm_applications": "eligible", "deep_learning": "eligible",
                  "introduction_to_machine_learning_open": "not_recommended", "natural_language_processing": "not_eligible",
                  "reinforcement_learning": "not_eligible"}
    assert [r.semester for r in demo.per_semester] == [5, 6]


def test_every_decision_is_explained_and_cited(demo):
    for a in demo.assessments:
        assert a.reasons and all(c.citations for c in a.checks)
    nmims = [c for a in demo.assessments for c in a.checks if c.rule == "nmims"]
    assert nmims and all(c.citations[0].source == "SRC052" for c in nmims)
    assert demo.human_review and "coordinator" in demo.disclaimer


def test_first_year_and_level_rules():
    out = get_advisor().advise(AdviseInput(student={"identity": "first year mechanical engineering student"},
                                           syllabi=syllabi(DATA / "electives" / "syllabi" / "management")))
    assert out.recommendation is None
    pg = next(a for a in out.assessments if a.name.startswith("business_analytics_pg"))
    assert {c.id for c in pg.checks if c.status == "fail"} >= {"level", "first_year"}


def test_personal_cues_do_not_change_advice(demo):
    for cue in ("My name is Priya.", "I belong to the SC category.", "I come from a small village in Bihar."):
        out = get_advisor().advise(AdviseInput(student={"identity": f"{cue} third year student, computer science program"},
                                               syllabi=syllabi()))
        assert [(a.name, a.status) for a in out.assessments] == [(a.name, a.status) for a in demo.assessments]
        assert out.recommendation.course == demo.recommendation.course


# --------------------------------------------------------------------------- API, partners, UI

def test_api_advise_and_parse_file():
    r = client.post("/v1/electives/advise", json={"student": {"identity": "B Tech Computer Engineering, semester V"},
                                                  "syllabi": syllabi()})
    assert r.status_code == 200 and r.json()["recommendation"]["title"] == "Deep Learning"
    b64 = base64.b64encode((AI / "deep_learning.md").read_bytes()).decode()
    p = client.post("/v1/electives/parse", json={"name": "deep_learning.md", "file_base64": b64}).json()
    assert p["title"] == "Deep Learning" and p["semesters"] == [5]


def test_api_validation():
    assert client.post("/v1/electives/advise", json={"student": {}, "syllabi": []}).status_code == 422
    assert client.post("/v1/electives/parse", json={"name": "x"}).status_code == 422


def test_partner_flow_uses_contract():
    r = client.post("/v1/electives/advise-from-partners", json={"topic": "artificial_intelligence", "student_id": "S-1002"})
    body = r.json()
    assert r.status_code == 200 and [c["service"] for c in body["partner_calls"]] == ["syllabus", "student_records"]
    assert body["advice"]["student"]["program"] == "B Tech Computer Engineering"
    assert client.get("/partner-mock/syllabi", params={"topic": "cybersecurity"}).json()["courses"]
    assert client.get("/partner-mock/students/S-9999").status_code == 404


def test_partner_unreachable_is_reported(monkeypatch):
    monkeypatch.setenv("PARTNER_SYLLABUS_URL", "http://127.0.0.1:9")      # nothing listens there
    r = client.post("/v1/electives/advise-from-partners", json={"topic": "artificial_intelligence", "student_id": "S-1001"})
    assert r.status_code == 502 and "partner service error" in r.json()["detail"]


def test_manifest_and_ui():
    m = client.get("/v1/manifest").json()
    assert m["group"] == "Group 10" and any(o["endpoint"] == "POST /v1/electives/advise" for o in m["offers"])
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "Elective advisor" in page.text
    assert client.get("/ui/app.js").status_code == 200
