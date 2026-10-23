"""Stand-in for other groups' components, implementing the contracts in nlp.integrations.partners.

Used in-process when no partner URL is configured, and served over HTTP at /partner-mock (backend.api.partner_mock)
so the cross-component call can be shown end to end. Syllabi come from data/electives/syllabi (illustrative, written
by Group 10); the student records are three fictional students.
"""
from nlp import DATA

SYLLABI = DATA / "electives" / "syllabi"
STUDENTS = {
    "S-1001": {"program": "B Tech Computer Science", "year": 3, "semester": None, "completed_courses": []},
    "S-1002": {"program": "B Tech Computer Engineering", "year": 3, "semester": 6,
               "completed_courses": ["Coursera: Deep Learning Specialization"]},
    "S-1003": {"program": "B Tech Mechanical Engineering", "year": 3, "semester": 5, "completed_courses": []},
}


def topics() -> list[str]:
    return sorted(p.name for p in SYLLABI.iterdir() if p.is_dir())


def mock_syllabi(topic: str) -> dict:
    folder = SYLLABI / topic
    if topic not in topics():
        raise KeyError(f"unknown topic {topic!r}")
    courses = [{"course_id": f.stem, "title": f.stem.replace("NEW_", "").replace("_", " ").title(),
                "syllabus_text": f.read_text(encoding="utf-8")} for f in sorted(folder.glob("*.*"))]
    return {"service": "mock syllabus service (stand-in for a partner group)", "topic": topic, "courses": courses}


def mock_student(student_id: str) -> dict:
    if student_id not in STUDENTS:
        raise KeyError(f"unknown student {student_id!r}")
    return {"service": "mock student-record service (stand-in for a partner group)", "student_id": student_id,
            **STUDENTS[student_id]}
