"""Elective advisor: given several syllabi (including new courses) and a student's academic identity, decide which
electives the student can enrol in, and which one fits best.

    identity ("third year student, computer science program")  ──►  StudentProfile (programme, year, semesters,
                                                                     courses completed per the official structure)
    syllabi (outline / PDF / free text)  ──►  ParsedSyllabus (type, programmes, semester, prerequisites, topics)
                                     │
                                     ▼
    checks per syllabus (each cited):  level (CBCS UG->UG, PG->PG) · programme · semester · elective slot in the
                                       programme structure · first-year rule · prerequisites · duplicate content
                                     │
                                     ▼
    rank eligible courses (prerequisite fit, novelty, interests)  ──►  recommendation + confidence + how to enrol

Rules come from the MPSTME Student Resource Book 2026 (SRC052) and the verified facts; "duplicate content" is our own
advising policy and is labelled as such. The output is advice: the course coordinator confirms every enrolment.
"""
from __future__ import annotations

import base64
import io
import re
from functools import lru_cache

import numpy as np

from nlp.component.schemas import Confidence
from nlp.electives import catalog
from nlp.electives.schemas import (
    AdviseInput, AdviseOutput, Check, Citation, CourseAssessment, EnrolmentStep, ParsedSyllabus, PrerequisiteMatch,
    Recommendation, StudentInput, StudentProfile, SyllabusInput,
)
from nlp.electives.syllabus import (
    AFTER_DEGREE, ANYWHERE, DEGREES, Syllabus, degree_of, find_programs, level_of, norm, parse_syllabus, program_index,
    semesters_in, tokens, year_in,
)

PREREQ_PASS, PREREQ_WARN = 0.75, 0.6     # match score for a prerequisite to count as completed / to need review
DUPLICATE = 0.85                         # title similarity above which an elective repeats a programme course
GENERIC = {"introduction", "intro", "fundamental", "basic", "principle", "applied", "advanced", "everyone",
           "foundation", "elements", "concept", "essential", "topic", "selected", "special", "engineer"}
DURATION = {"b tech": 4, "mba tech": 5, "m tech": 2, "master of": 2, "diploma in": 3}

SRB = dict(source=catalog.SOURCE_ID, title=catalog.SOURCE_TITLE, url=catalog.SOURCE_URL)
RULE_CBCS_LEVEL = Citation(**SRB, page=14, quote="The CBCS offerings of courses will be opted from a UG Program to a UG "
                                                 "Program and a PG Program to a PG Program")
RULE_FIRST_YEAR = Citation(**SRB, page=14, quote="The UG students of 1st year and PG students of Term I of 1st year are "
                                                 "not eligible to opt CBCS courses")
RULE_MIN_ENROL = Citation(**SRB, page=13, quote="The minimum 15 number of students or 40% of the total number of "
                                                "students, to offer an elective course/s")
RULE_ROUTE = Citation(source="F0009", title=catalog.SOURCE_TITLE, page=6, url=catalog.SOURCE_URL,
                      quote="They must route their option through Course Co-ordinators or Program Chairs")
RULE_PORTAL = Citation(source="F0089", title=catalog.SOURCE_TITLE, page=34, url=catalog.SOURCE_URL,
                       quote="options are available for MPSTME (Mumbai / Shirpur) students to choose Open Electives")
DISCLAIMER = ("Advice generated from the published MPSTME programme structures and rules and from the syllabi provided. "
              "It is not an enrolment: elective choices are confirmed by the course coordinator or programme chair, "
              "and an elective runs only if enough students register.")
ROMAN = {v: k for k, v in catalog.ROMAN.items()}


# --------------------------------------------------------------------------- text similarity

@lru_cache(maxsize=1)
def _encoder():
    from nlp.retrieval.search import get_encoder

    return get_encoder()


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    return _encoder().encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def lexical(a: str, b: str) -> float:
    """Share of a's content words found in b (generic words like 'introduction' ignored)."""
    ta, tb = tokens(a) - GENERIC, tokens(b) - GENERIC
    return len(ta & tb) / len(ta) if ta else 0.0


def best_match(query: str, candidates: list[str], cand_emb: np.ndarray | None = None,
               both_ways: bool = False) -> tuple[int, float]:
    """Closest candidate by max(embedding cosine, word containment). both_ways also counts a candidate of two or more
    content words contained in the query ('Generative AI' inside 'Generative AI and LLM Applications')."""
    if not candidates:
        return -1, 0.0
    cand_emb = embed(candidates) if cand_emb is None else cand_emb
    cos = cand_emb @ embed([query])[0]
    lex = np.array([max(lexical(query, c), lexical(c, query) if both_ways and len(tokens(c) - GENERIC) >= 2 else 0)
                    for c in candidates])
    score = np.maximum(cos, lex)
    i = int(score.argmax())
    return i, float(min(1.0, max(0.0, score[i])))


# --------------------------------------------------------------------------- inputs

def read_syllabus(s: SyllabusInput) -> tuple[str, str]:
    """(name, text) from text or a base64 .pdf / .docx / .txt file."""
    name = s.name or "syllabus"
    if s.text:
        return name, s.text
    if not s.file_base64:
        raise ValueError(f"{name}: give either text or file_base64")
    data = base64.b64decode(s.file_base64)
    low = name.lower()
    if low.endswith(".pdf") or data[:4] == b"%PDF":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return name, "\n".join(page.get_text() for page in doc)
    if low.endswith(".docx") or data[:2] == b"PK":
        from docx import Document

        return name, "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return name, data.decode("utf-8", errors="replace")


def match_program(text: str) -> tuple[str | None, list[str], str]:
    """Programme named in free text -> (best, alternatives, 'exact' | 'partial' | 'none')."""
    # upper-case branch codes are safe to expand ("IT student"); lower-case "it" / "ai" may be ordinary words
    text = re.sub(r"\b(IT|CE|CS|AI|DS)\b", lambda m: " " + AFTER_DEGREE[m.group(1).lower()] + " ", text)
    t = norm(text)
    for a, b in ANYWHERE.items():
        t = re.sub(rf"\b{a}\b", b, t)
    full = find_programs(t)
    if full:
        return full[0], full[1:], "exact"
    degree = next((d for d in DEGREES if re.search(rf"\b{d}\b", t)), None)
    exact, partial = [], []
    for name, f, body in program_index():
        if degree and not f.startswith(degree):
            continue
        if re.search(rf"\b{re.escape(body)}\b", t):
            exact.append(name)
        elif len(tokens(body) & tokens(t)) >= max(1, len(tokens(body)) - 1) and tokens(body) & tokens(t) - {"engineering"}:
            partial.append(name)
    # prefer the longest body found ("computer science and business systems" over "computer science"), then B Tech
    exact = sorted(exact, key=lambda n: (-len(next(b for x, _, b in program_index() if x == n)),
                                         not norm(n).startswith("b tech")))
    if exact:
        best = exact[0]
        body = next(b for x, _, b in program_index() if x == best)
        # "did you mean": same degree, longer names starting the same ("computer science and business systems"), then
        # other exact or partial matches
        same_deg = degree or degree_of(best)
        longer = [x for x, f, b in program_index() if x != best and b.startswith(body + " ") and f.startswith(same_deg)]
        alts = list(dict.fromkeys(longer + [n for n in exact[1:] + partial if n != best]))
        return best, alts[:5], "exact"
    if partial:
        partial.sort(key=lambda n: (not norm(n).startswith("b tech"), len(n)))
        return partial[0], partial[1:5], "partial"
    return None, [], "none"


def build_profile(s: StudentInput) -> tuple[StudentProfile, dict]:
    text = " ".join(x for x in (s.identity, s.program) if x)
    notes = []
    if s.program:
        prog, alts, how = match_program(s.program)
        how = "given" if how == "exact" else how
    else:
        prog, alts, how = match_program(s.identity or "")
    if how == "none":
        notes.append("Programme not recognised; name it as in the MPSTME structures (e.g. 'B Tech Computer Science').")
    elif how == "partial":
        notes.append(f"Programme matched approximately to {prog}; please confirm.")
    elif alts:
        notes.append(f"Read as {prog}. If you meant another programme ({', '.join(alts[:3])}), say so.")
    sems = [s.semester] if s.semester else semesters_in(s.identity or "")
    year = s.year or (year_in(s.identity or "") if s.identity else None)
    if year == -1:
        year = DURATION.get(degree_of(prog or ""), 4) if prog else None
    if sems and not year:
        year = (sems[0] + 1) // 2
    if year and not sems:
        sems = [2 * year - 1, 2 * year]
        notes.append(f"Semester not given: considering both semesters of year {year} "
                     f"({ROMAN[sems[0]]} and {ROMAN[sems[1]]}).")
    if year and sems and any((x + 1) // 2 != year for x in sems):
        notes.append(f"Semester {sems} does not match year {year}; using the semester.")
        year = (sems[0] + 1) // 2
    if not year:
        notes.append("Year or semester not found; give e.g. 'third year' or 'semester VI'.")
    completed = []
    if prog and sems:
        completed = [c.course for c in catalog.courses_before(prog, min(sems))]
    completed += [c for c in s.completed_courses if c.strip()]
    score = {"exact": 0.95, "given": 1.0, "partial": 0.6, "none": 0.0}[how] * (1.0 if year else 0.5)
    if how == "exact" and alts and not s.program:
        score *= 0.9
    level = "high" if score >= 0.85 else "medium" if score >= 0.6 else "low"
    prof = StudentProfile(program=prog, program_alternatives=alts, program_match=how,
                          degree_level=level_of(prog) if prog else None, year=year, semesters=sems,
                          completed_courses=completed, interests=s.interests,
                          confidence=Confidence(score=round(score, 2), level=level,
                                                method="programme match (exact / partial) x year or semester found"),
                          notes=notes)
    return prof, {"text": text}


# --------------------------------------------------------------------------- checks

def _slot_rows(program: str, semester: int, course_type: str):
    rows = catalog.semester_rows(program, semester)
    want = {"program_elective": ["program_elective_slot"], "open_elective": ["open_elective_slot"]}.get(
        course_type, ["program_elective_slot", "open_elective_slot"])
    return rows, rows[rows["kind"].isin(want)]


def assess(syl: Syllabus, name: str, prof: StudentProfile, programme_courses: list) -> CourseAssessment:
    checks: list[Check] = []
    reasons: list[str] = []
    src = Citation(source=name, title=f"Syllabus: {syl.title}")
    is_open = syl.course_type == "open_elective"

    # level (CBCS: UG -> UG, PG -> PG)
    if syl.level and prof.degree_level:
        ok = syl.level == prof.degree_level
        checks.append(Check(id="level", label="Programme level", status="pass" if ok else "fail", rule="nmims",
                            detail=f"{syl.level} course; you are in a {prof.degree_level} programme."
                            + ("" if ok else " CBCS courses are taken UG to UG and PG to PG only."),
                            citations=[RULE_CBCS_LEVEL, src]))
    # programme
    if prof.program:
        if syl.programs:
            ok = prof.program in syl.programs
            st = "pass" if ok else ("unknown" if is_open and not syl.open_to_all else "fail")
            detail = (f"Offered to {prof.program}." if ok else
                      f"Offered to {', '.join(syl.programs)}; not to {prof.program}.")
        elif syl.open_to_all:
            st, detail = ("pass", f"Open to all {syl.open_to_all} programmes.")
        else:
            st, detail = "unknown", "The syllabus does not say which programmes it is offered to."
        checks.append(Check(id="program", label="Offered to your programme", status=st, detail=detail,
                            rule="syllabus", citations=[src]))
    # semester
    matched = [s for s in syl.semesters if s in prof.semesters]
    if not syl.semesters:
        checks.append(Check(id="semester", label="Semester", status="unknown", rule="syllabus", citations=[src],
                            detail="The syllabus does not say in which semester it is taught."))
        matched = list(prof.semesters)
    elif prof.semesters:
        sem_txt = "/".join(ROMAN[s] for s in syl.semesters)
        you = "/".join(ROMAN[s] for s in prof.semesters)
        checks.append(Check(id="semester", label="Semester", status="pass" if matched else "fail", rule="syllabus",
                            citations=[src], detail=f"Taught in Semester {sem_txt}; you are in Semester {you}."))
    # elective slot in the official programme structure
    semester, slot = None, None
    if prof.program and matched:
        for s in matched:
            rows, slots = _slot_rows(prof.program, s, syl.course_type)
            if len(slots):
                semester, slot = s, slots.iloc[0]["course"]
                page = int(slots.iloc[0]["page"])
                kinds = sorted(set(slots["course"]))
                checks.append(Check(id="slot", label="Elective slot in your programme structure", status="pass",
                                    rule="nmims", citations=[Citation(**SRB, page=page)],
                                    detail=f"{prof.program}, Semester {ROMAN[s]} has {', '.join(kinds)}."))
                break
        else:
            any_rows = any(len(_slot_rows(prof.program, s, syl.course_type)[0]) for s in matched)
            kind = {"program_elective": "programme-elective", "open_elective": "open-elective"}.get(
                syl.course_type, "elective")
            checks.append(Check(id="slot", label="Elective slot in your programme structure",
                                status="fail" if any_rows else "unknown", rule="nmims",
                                citations=[Citation(**SRB, page=None)],
                                detail=(f"{prof.program} has no {kind} slot in Semester "
                                        f"{'/'.join(ROMAN[s] for s in matched)}." if any_rows else
                                        "Programme structure for this semester not found in the Student Resource Book.")))
    # first-year rule (CBCS courses are open / cross-programme electives)
    if is_open and prof.year == 1 and prof.degree_level == "UG":
        checks.append(Check(id="first_year", label="First-year students", status="fail", rule="nmims",
                            citations=[RULE_FIRST_YEAR],
                            detail="First-year UG students are not eligible to opt CBCS (open) courses."))
    # prerequisites
    pre_matches: list[PrerequisiteMatch] = []
    if syl.prerequisites:
        # compulsory courses of the semesters before the one the elective is taken in, plus courses the student added
        limit = semester or (min(matched) if matched else min(prof.semesters or [1]))
        sem_of = {c.course: c.semester for c in programme_courses}
        extra = [c for c in prof.completed_courses if c not in sem_of]
        done = list(dict.fromkeys([c.course for c in programme_courses if c.semester < limit] + extra))
        # without the programme and semester the course history is unknown: unmatched means unknown, not failed
        history_known = bool(prof.program and prof.semesters)
        emb = embed(done) if done else None
        for p in syl.prerequisites:
            i, sc = best_match(p, done, emb) if done else (-1, 0.0)
            st = "pass" if sc >= PREREQ_PASS else "warn" if sc >= PREREQ_WARN else "fail" if history_known else "unknown"
            pre_matches.append(PrerequisiteMatch(prerequisite=p, matched_course=done[i] if i >= 0 and st in ("pass", "warn")
                                                 else None, semester=sem_of.get(done[i]) if i >= 0 else None,
                                                 score=round(sc, 3), status=st))
        worst = "fail" if any(m.status == "fail" for m in pre_matches) else \
            "unknown" if any(m.status in ("warn", "unknown") for m in pre_matches) else "pass"
        missing = "not found in your completed courses" if history_known else \
            "cannot check without your programme and semester"
        detail = "; ".join(f"{m.prerequisite}: " + (f"{m.matched_course} (Sem {ROMAN.get(m.semester, '?')})"
                                                    if m.matched_course else missing)
                           for m in pre_matches)
        checks.append(Check(id="prerequisites", label="Prerequisites completed", status=worst, detail=detail,
                            rule="syllabus", citations=[src]))
    # duplicate content (advisor policy)
    novelty = None
    if programme_courses:
        titles = [c.course for c in programme_courses]
        i, sc = best_match(syl.title, titles, both_ways=True)
        novelty = round(1 - sc, 3)
        if sc >= DUPLICATE:
            c = programme_courses[i]
            checks.append(Check(id="duplicate", label="Not already in your programme", status="warn",
                                rule="advisor_policy", citations=[Citation(**SRB, page=c.page)],
                                detail=f"Very similar to '{c.course}', a compulsory course in your Semester "
                                       f"{ROMAN[c.semester]} (similarity {sc:.2f}); it would add little."))

    statuses = [c.status for c in checks if c.id != "duplicate"]
    if "fail" in statuses:
        status = "not_eligible"
        reasons = [c.detail for c in checks if c.status == "fail"]
    elif "unknown" in statuses or not prof.program or not prof.semesters:
        status = "needs_review"
        reasons = [c.detail for c in checks if c.status == "unknown"] or ["Student programme or semester unknown."]
    elif any(c.id == "duplicate" for c in checks):
        status = "not_recommended"
        reasons = [c.detail for c in checks if c.id == "duplicate"]
    else:
        status = "eligible"
        reasons = [c.detail for c in checks if c.id in ("slot", "prerequisites", "program")]
    return CourseAssessment(name=name, title=syl.title, is_new=syl.is_new, status=status, semester=semester,
                            slot=slot, checks=checks, prerequisites=pre_matches, fit_score=None, interest_match=None,
                            novelty=novelty, reasons=reasons)


# --------------------------------------------------------------------------- advisor

class ElectiveAdvisor:
    def __init__(self):
        from nlp.knowledge.store import load_services

        self.service = load_services().get("ELECTIVE_SELECTION")

    def advise(self, inp: AdviseInput) -> AdviseOutput:
        prof, _ = build_profile(inp.student)
        programme_courses = (catalog.courses_in(prof.program, list(range(1, 11))) if prof.program else [])
        parsed, assessments, syls = [], [], []
        for s in inp.syllabi:
            name, text = read_syllabus(s)
            syl = parse_syllabus(text, re.sub(r"\.\w+$", "", name))
            syls.append(syl)
            parsed.append(ParsedSyllabus(name=name, title=syl.title, code=syl.code, course_type=syl.course_type,
                                         programs=syl.programs, open_to_all=syl.open_to_all, level=syl.level,
                                         semesters=syl.semesters, credits=syl.credits, prerequisites=syl.prerequisites,
                                         topics=syl.topics, is_new=syl.is_new, fields_found=syl.found,
                                         parse_confidence=syl.parse_confidence))
            assessments.append(assess(syl, name, prof, programme_courses))

        # rank eligible courses
        content = [f"{s.title}. " + "; ".join(s.topics) for s in syls]
        emb = embed(content)
        interest = embed([prof.interests])[0] if prof.interests else None
        for a, e, s in zip(assessments, emb, syls):
            if a.status not in ("eligible", "not_recommended"):
                continue
            fit = float(np.mean([m.score for m in a.prerequisites])) if a.prerequisites else 1.0
            nov = a.novelty if a.novelty is not None else 0.5
            if interest is not None:
                a.interest_match = round(float(max(0.0, e @ interest)), 3)
                a.fit_score = round(0.5 * a.interest_match + 0.3 * fit + 0.2 * nov, 3)
            else:
                a.fit_score = round(0.6 * fit + 0.4 * nov, 3)
        order = {"eligible": 0, "not_recommended": 1, "needs_review": 2, "not_eligible": 3}
        idx = sorted(range(len(assessments)), key=lambda i: (order[assessments[i].status],
                                                             -(assessments[i].fit_score or 0)))
        assessments = [assessments[i] for i in idx]
        parsed_by = {p.name: p for p in parsed}

        per_sem = []
        for sem in prof.semesters:
            pool = [a for a in assessments if a.status == "eligible" and a.semester == sem]
            if pool:
                per_sem.append(self._recommend(pool, parsed_by, prof))
        fit = {a.name: a.fit_score for a in assessments}
        rec = max(per_sem, key=lambda r: fit[r.course]).model_copy(deep=True) if per_sem else None
        if rec and len(per_sem) > 1:
            rec.why.append("Semester not given: this is the best fit overall; the best course for each semester is "
                           "listed separately.")

        overlaps = []
        elig = [orig for pos, orig in enumerate(idx) if assessments[pos].status == "eligible"]   # original indices
        for x in range(len(elig)):
            for y in range(x + 1, len(elig)):
                i, j = elig[x], elig[y]
                sim = float(emb[i] @ emb[j])
                if sim >= 0.55:
                    overlaps.append({"a": syls[i].title, "b": syls[j].title, "similarity": round(sim, 3),
                                     "note": "Content overlaps; taking both repeats material."})

        human = ["Confirm the choice with your course coordinator or programme chair before registering (F0009)."]
        if prof.program_match != "given" and prof.program_alternatives:
            human.append(f"Check the programme: read as {prof.program}.")
        if any(a.is_new for a in assessments):
            human.append("New courses must be approved and offered in your semester; check with the department.")
        if any(a.status == "needs_review" for a in assessments):
            human.append("Some syllabi are missing information (see needs_review); a coordinator should decide those.")
        svc = self.service
        how = EnrolmentStep(department=svc.department if svc else "Course coordinator / Programme chair",
                            next_action=svc.next_action if svc else "Route your choice through the course coordinator",
                            rules=[RULE_ROUTE, RULE_PORTAL, RULE_MIN_ENROL])
        return AdviseOutput(request_id=inp.request_id, student=prof, syllabi=parsed, assessments=assessments,
                            recommendation=rec, per_semester=per_sem, overlaps=overlaps, how_to_enrol=how,
                            human_review=human, disclaimer=DISCLAIMER)

    @staticmethod
    def _recommend(pool: list[CourseAssessment], parsed_by: dict, prof: StudentProfile) -> Recommendation:
        top = pool[0]
        second = pool[1] if len(pool) > 1 else None
        p = parsed_by[top.name]
        margin = (top.fit_score - second.fit_score) if second else 1.0
        certainty = 0.5 * p.parse_confidence + 0.5 * prof.confidence.score
        score = round(min(1.0, certainty * (0.7 + 0.3 * min(1.0, margin / 0.15))), 3)
        level = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
        why = [r for r in top.reasons]
        if top.interest_match is not None:
            why.append(f"Matches your interests (similarity {top.interest_match:.2f}).")
        if second:
            why.append(f"Ahead of '{second.title}' (fit {top.fit_score:.2f} vs {second.fit_score:.2f}).")
        if top.is_new:
            why.append("This is a new course: its syllabus was parsed from the proposal provided.")
        return Recommendation(semester=top.semester, course=top.name, title=top.title, slot=top.slot,
                              confidence=Confidence(score=score, level=level,
                                                    method="syllabus parse confidence x identity confidence x "
                                                           "margin over the runner-up"),
                              why=why, runner_up=second.title if second else None)


@lru_cache(maxsize=1)
def get_advisor() -> ElectiveAdvisor:
    return ElectiveAdvisor()
