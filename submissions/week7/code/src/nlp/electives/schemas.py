"""Structured input and output of the elective advisor (pydantic; same conventions as nlp.component.schemas)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nlp.component.schemas import SCHEMA_VERSION, Confidence

MAX_SYLLABI = 20
MAX_SYLLABUS_CHARS = 30000

CheckStatus = Literal["pass", "fail", "warn", "unknown"]
CourseStatus = Literal["eligible", "not_recommended", "needs_review", "not_eligible"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- input

class SyllabusInput(_Model):
    name: str | None = Field(None, max_length=200, description="File name or label, shown in the output.")
    text: str | None = Field(None, max_length=MAX_SYLLABUS_CHARS, description="Syllabus text (any format).")
    file_base64: str | None = Field(None, description="A .pdf, .docx, .txt or .md file, base64-encoded, instead of text.")
    source: str | None = Field(None, max_length=200, description="Where it came from, e.g. a partner service.")


class StudentInput(_Model):
    identity: str | None = Field(None, max_length=500, examples=["third year student, computer science program"],
                                 description="Free-text academic identity; parsed into the fields below.")
    program: str | None = Field(None, max_length=200, description="Overrides the programme found in `identity`.")
    year: int | None = Field(None, ge=1, le=5)
    semester: int | None = Field(None, ge=1, le=10)
    completed_courses: list[str] = Field(default_factory=list, max_length=100,
                                         description="Extra courses taken (e.g. MOOCs); the programme's earlier "
                                                     "compulsory courses are assumed completed.")
    interests: str | None = Field(None, max_length=500, description="Optional: topics or career goals, used for ranking.")

    @field_validator("identity", "program", "interests")
    @classmethod
    def _strip(cls, v):
        return v.strip() or None if isinstance(v, str) else v


class AdviseInput(_Model):
    student: StudentInput
    syllabi: list[SyllabusInput] = Field(..., min_length=1, max_length=MAX_SYLLABI)
    request_id: str | None = Field(None, max_length=128)


# --------------------------------------------------------------------------- output

class Citation(_Model):
    source: str = Field(..., description="Source id (SRC052 = MPSTME Student Resource Book 2026), fact id, or syllabus.")
    title: str
    page: int | None = None
    url: str | None = None
    quote: str | None = None


class ParsedSyllabus(_Model):
    name: str
    title: str
    code: str | None
    course_type: Literal["program_elective", "open_elective", "core", "unknown"]
    programs: list[str]
    open_to_all: Literal["UG", "PG"] | None
    level: Literal["UG", "PG"] | None
    semesters: list[int]
    credits: int | None
    prerequisites: list[str]
    topics: list[str]
    is_new: bool
    fields_found: dict[str, Literal["labelled", "inferred", "missing"]]
    parse_confidence: float = Field(..., ge=0, le=1, description="Share of required fields found.")


class StudentProfile(_Model):
    program: str | None
    program_alternatives: list[str] = Field(..., description="Other programmes the identity could mean.")
    program_match: Literal["exact", "partial", "none", "given"]
    degree_level: Literal["UG", "PG"] | None
    year: int | None
    semesters: list[int] = Field(..., description="Semesters considered (the given one, or both of the year).")
    completed_courses: list[str] = Field(..., description="Courses treated as completed.")
    interests: str | None
    confidence: Confidence
    notes: list[str]


class Check(_Model):
    id: Literal["level", "program", "semester", "slot", "first_year", "prerequisites", "duplicate"]
    label: str
    status: CheckStatus
    detail: str
    rule: Literal["nmims", "syllabus", "advisor_policy"] = Field(
        ..., description="nmims = published NMIMS rule; syllabus = stated in the syllabus; advisor_policy = our own "
                         "recommendation logic (not a university rule).")
    citations: list[Citation] = Field(default_factory=list)


class PrerequisiteMatch(_Model):
    prerequisite: str
    matched_course: str | None
    semester: int | None
    score: float = Field(..., ge=0, le=1)
    status: CheckStatus


class CourseAssessment(_Model):
    name: str
    title: str
    is_new: bool
    status: CourseStatus
    semester: int | None = Field(..., description="Semester in which the student can take it.")
    slot: str | None = Field(..., description="Programme-structure slot it fills, e.g. 'Program Elective - III'.")
    checks: list[Check]
    prerequisites: list[PrerequisiteMatch]
    fit_score: float | None = Field(..., description="Ranking score among eligible courses (0-1).")
    interest_match: float | None
    novelty: float | None = Field(..., description="1 - similarity to the closest course already in the programme.")
    reasons: list[str]


class Recommendation(_Model):
    semester: int | None
    course: str
    title: str
    slot: str | None
    confidence: Confidence
    why: list[str]
    runner_up: str | None


class EnrolmentStep(_Model):
    department: str
    next_action: str
    rules: list[Citation]


class AdviseOutput(_Model):
    request_id: str | None
    student: StudentProfile
    syllabi: list[ParsedSyllabus]
    assessments: list[CourseAssessment] = Field(..., description="One per syllabus, best first.")
    recommendation: Recommendation | None = Field(..., description="The elective to enrol in; null if none is eligible.")
    per_semester: list[Recommendation] = Field(..., description="Best eligible elective for each semester considered.")
    overlaps: list[dict] = Field(..., description="Pairs of eligible syllabi with heavily overlapping content.")
    how_to_enrol: EnrolmentStep
    human_review: list[str] = Field(..., description="What a course coordinator should confirm before enrolment.")
    disclaimer: str
    schema_version: str = SCHEMA_VERSION
