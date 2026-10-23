"""Connections to other groups' components of the University AI application.

Two inbound contracts (what we consume) - any service that implements them can be plugged in with one environment
variable; until a partner is configured, the bundled mock (nlp.integrations.mock, same contract) is used:

  SYLLABUS SERVICE   GET {PARTNER_SYLLABUS_URL}/syllabi?topic=<topic>
                     -> {"service": str, "topic": str, "courses": [{"course_id": str, "title": str,
                         "syllabus_text": str}]}
  STUDENT RECORDS    GET {PARTNER_STUDENT_URL}/students/<student_id>
                     -> {"service": str, "student_id": str, "program": str, "year": int, "semester": int | null,
                         "completed_courses": [str]}

Outbound, other groups call our /v1 API (docs/integration.md). Only what the advisor needs is requested from the
student-record service (programme, year, semester, completed courses): no name, contact details or grades.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

TIMEOUT_S = float(os.getenv("PARTNER_TIMEOUT_S", "8"))


class PartnerError(RuntimeError):
    """The partner service could not be reached or returned an unexpected response."""


@dataclass
class PartnerCall:
    service: str
    url: str
    ms: float
    ok: bool
    mock: bool


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "nmims-admin-copilot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise PartnerError(f"{url}: {e}") from e


class PartnerClient:
    """base_url None -> the bundled mock is called in-process (no network), with the same contract."""

    def __init__(self, syllabus_url: str | None = None, student_url: str | None = None):
        self.syllabus_url = (syllabus_url or os.getenv("PARTNER_SYLLABUS_URL") or "").rstrip("/") or None
        self.student_url = (student_url or os.getenv("PARTNER_STUDENT_URL") or "").rstrip("/") or None
        self.calls: list[PartnerCall] = []

    def _call(self, service: str, base: str | None, path: str, mock_fn) -> dict:
        import time

        t0 = time.perf_counter()
        url = f"{base}{path}" if base else f"mock:{path}"
        try:
            data = _get(url) if base else mock_fn()
            self.calls.append(PartnerCall(service, url, round((time.perf_counter() - t0) * 1000, 1), True, base is None))
            return data
        except (PartnerError, KeyError) as e:
            self.calls.append(PartnerCall(service, url, round((time.perf_counter() - t0) * 1000, 1), False, base is None))
            raise PartnerError(str(e)) from e
        except Exception:
            self.calls.append(PartnerCall(service, url, round((time.perf_counter() - t0) * 1000, 1), False, base is None))
            raise

    def syllabi(self, topic: str) -> dict:
        from nlp.integrations.mock import mock_syllabi

        data = self._call("syllabus", self.syllabus_url, f"/syllabi?topic={urllib.parse.quote(topic)}",
                          lambda: mock_syllabi(topic))
        if not isinstance(data.get("courses"), list) or not all("syllabus_text" in c for c in data["courses"]):
            raise PartnerError("syllabus service response does not follow the contract (courses[].syllabus_text)")
        return data

    def student(self, student_id: str) -> dict:
        from nlp.integrations.mock import mock_student

        data = self._call("student_records", self.student_url, f"/students/{urllib.parse.quote(student_id)}",
                          lambda: mock_student(student_id))
        if "program" not in data:
            raise PartnerError("student-record response does not follow the contract (program)")
        return data
