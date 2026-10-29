# Final demonstration script

## Start

```bash
uv sync
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/ui/>.

## Main scenario (about 4 minutes)

1. In **Elective advisor**, leave the identity as
   `third year student, computer science program`.
2. Select the **Artificial intelligence** sample set.
3. Press **Find my elective**.
4. Point out the recommended new course, semester, slot, and confidence.
5. Expand **Every course checked**. Show that each course has a status,
   reason, rule classification, and citation.
6. Show **Before you register** and explain that the coordinator still checks
   capacity, timetable, prerequisites, and approval.
7. Open the JSON disclosure to show the typed response available to another
   application.

## Integration scenario (about 2 minutes)

1. Open **Partner integration**.
2. Keep student `S-1001` and topic `artificial_intelligence`.
3. Press **Fetch from partners and advise**.
4. Show the syllabus-service and student-record calls, their status and
   latency, followed by the same advisor output.
5. Explain that setting `PARTNER_SYLLABUS_URL` and `PARTNER_STUDENT_URL`
   replaces the bundled mocks without changing the advisor.

## Administrative NLP scenario (about 1 minute)

1. Open **Ask the copilot**.
2. Ask: `I lost my ID card, roll no J054. What should I do?`
3. Show intent, department, extracted entities, grounded answer, sources, and
   next action.

## Recovery / boundary demonstration

- Replace the identity with an unsupported programme. The system reports that
  it cannot recognise the programme instead of guessing.
- Remove the semester. The result uses the possible year semesters and
  exposes the uncertainty.
- Use **Upload files** or **Paste a syllabus** to show the additional-course
  path.

## Stop criteria

The demonstration is successful when the browser loads, the main scenario
returns a recommendation, the partner scenario reports two successful calls,
and the copilot scenario returns a grounded response.
