"""HTTP face of the partner mock (nlp.integrations.mock), at /partner-mock. Point PARTNER_SYLLABUS_URL or
PARTNER_STUDENT_URL here to exercise the real HTTP path, or at a partner group's service to integrate with it."""
from fastapi import APIRouter, HTTPException

from nlp.integrations.mock import mock_student, mock_syllabi

router = APIRouter(prefix="/partner-mock", tags=["partner mock (stand-in for other groups' components)"])


@router.get("/syllabi")
def syllabi(topic: str):
    try:
        return mock_syllabi(topic)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/students/{student_id}")
def student(student_id: str):
    try:
        return mock_student(student_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
