"""HTTP API of the elective advisor (/v1/electives), the partner-integration flow and the component manifest."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nlp.component import COMPONENT_VERSION, SCHEMA_VERSION
from nlp.electives import catalog
from nlp.electives.advisor import get_advisor, read_syllabus
from nlp.electives.schemas import AdviseInput, AdviseOutput, ParsedSyllabus, StudentInput, SyllabusInput
from nlp.electives.syllabus import parse_syllabus
from nlp.integrations import mock
from nlp.integrations.partners import PartnerClient, PartnerError

router = APIRouter(prefix="/v1", tags=["elective advisor"])


@router.get("/electives/programs")
def programs() -> dict:
    """Programmes in the official MPSTME structures (the names the advisor recognises)."""
    return {"programs": catalog.programs(), "source": f"{catalog.SOURCE_ID} {catalog.SOURCE_TITLE}"}


@router.get("/electives/samples")
def samples() -> dict:
    """Demo syllabus sets (illustrative, written by Group 10) for the UI."""
    return {"topics": {t: mock.mock_syllabi(t)["courses"] for t in mock.topics()},
            "note": "Illustrative syllabi in the MPSTME course-outline format; not official NMIMS documents."}


@router.post("/electives/parse", response_model=ParsedSyllabus)
def parse(body: SyllabusInput):
    """Structured fields of one syllabus, with how each was found."""
    try:
        name, text = read_syllabus(body)
    except ValueError as e:
        raise HTTPException(422, str(e))
    s = parse_syllabus(text, name)
    return ParsedSyllabus(name=name, title=s.title, code=s.code, course_type=s.course_type, programs=s.programs,
                          open_to_all=s.open_to_all, level=s.level, semesters=s.semesters, credits=s.credits,
                          prerequisites=s.prerequisites, topics=s.topics, is_new=s.is_new, fields_found=s.found,
                          parse_confidence=s.parse_confidence)


@router.post("/electives/advise", response_model=AdviseOutput)
def advise(body: AdviseInput):
    """Which of the given electives the student can enrol in, and which fits best."""
    try:
        return get_advisor().advise(body)
    except ValueError as e:
        raise HTTPException(422, str(e))


class PartnerAdviseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(..., examples=["artificial_intelligence"], description="Topic requested from the syllabus service.")
    student_id: str | None = Field(None, examples=["S-1001"], description="Fetched from the student-record service.")
    student: StudentInput | None = Field(None, description="Used instead of student_id (or to add interests).")
    extra_syllabi: list[SyllabusInput] = Field(default_factory=list, max_length=10,
                                               description="E.g. a new course syllabus not yet in the partner service.")


@router.post("/electives/advise-from-partners")
def advise_from_partners(body: PartnerAdviseInput) -> dict:
    """Cross-component flow: syllabi from the partner syllabus service and the academic identity from the partner
    student-record service (bundled mocks until PARTNER_*_URL is set), then the advisor."""
    client = PartnerClient()
    try:
        feed = client.syllabi(body.topic)
        record = client.student(body.student_id) if body.student_id else None
    except PartnerError as e:
        raise HTTPException(502, f"partner service error: {e}")
    if record:
        st = StudentInput(program=record["program"], year=record.get("year"), semester=record.get("semester"),
                          completed_courses=record.get("completed_courses", []),
                          interests=body.student.interests if body.student else None)
    elif body.student:
        st = body.student
    else:
        raise HTTPException(422, "give student_id or student")
    syllabi = [SyllabusInput(name=c["course_id"], text=c["syllabus_text"], source=feed.get("service"))
               for c in feed["courses"]] + list(body.extra_syllabi)
    advice = get_advisor().advise(AdviseInput(student=st, syllabi=syllabi[:20]))
    return {"partner_calls": [c.__dict__ for c in client.calls], "student_record": record,
            "advice": advice.model_dump()}


@router.get("/manifest")
def manifest() -> dict:
    """Component card for the University AI application's service registry: what this component offers and needs."""
    return {
        "component": "University Administrative Copilot", "group": "Group 10", "version": COMPONENT_VERSION,
        "schema_version": SCHEMA_VERSION, "layer": "Student services / administrative NLP",
        "offers": [
            {"capability": "elective_recommendation", "endpoint": "POST /v1/electives/advise",
             "input": "student academic identity + syllabi", "output": "eligibility per course, recommendation, citations"},
            {"capability": "syllabus_parsing", "endpoint": "POST /v1/electives/parse"},
            {"capability": "admin_query_understanding", "endpoint": "POST /v1/analyze",
             "output": "intent, department routing, entities, grounded answer, service request, decision"},
            {"capability": "intent", "endpoint": "POST /v1/intent"}, {"capability": "entities", "endpoint": "POST /v1/entities"},
            {"capability": "knowledge_retrieval", "endpoint": "POST /v1/retrieve"},
        ],
        "consumes": [
            {"capability": "syllabus_feed", "contract": "GET /syllabi?topic=", "configured_by": "PARTNER_SYLLABUS_URL"},
            {"capability": "student_academic_record", "contract": "GET /students/{id}", "configured_by": "PARTNER_STUDENT_URL"},
        ],
        "docs": ["/docs", "docs/elective_advisor.md", "docs/integration.md", "docs/responsible_ai.md"],
    }
