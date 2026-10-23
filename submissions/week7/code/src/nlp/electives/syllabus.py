"""Turn a syllabus (course outline or free-text new-course proposal) into structured fields.

Two layers, so that both formal outlines and unseen formats work:
  1. labelled fields   "Course Title:", "Course Type:", "Program(s):", "Year / Semester:", "Credits:", "Prerequisites:"
  2. prose fallback    patterns such as "open to third-year (Semester VI) students of ...", "expected to have
                       completed ...", "programme elective", "3-credit"; programme and course names are recognised
                       against the official MPSTME catalogue, so "Data Structures and Algorithms" is not split at "and"

Every field records how it was found (labelled / inferred / missing); parse_confidence is the share of the fields
the advisor needs that were found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from nlp.electives import catalog

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
SEM_WORDS = {w: i for i, w in enumerate(["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                                         "ninth", "tenth"]) if w}
ORDINAL = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
           "5th": 5}
DEGREES = ["mba tech", "b tech", "m tech", "diploma in", "master of"]
# abbreviations expanded only right after a degree ("BTech IT"), where they cannot be ordinary words
AFTER_DEGREE = {"cs": "computer science", "ce": "computer engineering", "it": "information technology",
                "ai": "artificial intelligence", "ds": "data science", "aiml": "artificial intelligence machine learning",
                "csbs": "computer science and business systems", "mech": "mechanical engineering",
                "extc": "electronics and telecommunication engineering", "cse": "computer science and engineering"}
# abbreviations safe to expand anywhere (in a student's description of themselves)
ANYWHERE = {"cse": "computer science and engineering", "csbs": "computer science and business systems",
            "aiml": "artificial intelligence machine learning", "extc": "electronics and telecommunication engineering",
            "mech": "mechanical engineering", "comps": "computer engineering", "cs": "computer science"}
STOP = {"and", "of", "the", "in", "for", "with", "to", "a", "an", "program", "programme", "&"}
REQUIRED = ("title", "course_type", "programs", "semesters", "prerequisites")


def norm(text: str) -> str:
    t = text.lower().replace("&", " and ").replace("programme", "program")
    t = re.sub(r"\bb\.?\s?tech\.?", "b tech", t)
    t = re.sub(r"\bm\.?\s?tech\.?", "m tech", t)
    t = re.sub(r"\bmba\s*\(?\s*tech\.?\s*\)?", "mba tech ", t)
    t = re.sub(r"\bengg\b\.?", "engineering", t)
    t = re.sub(r"[()]", " ", t)
    t = re.sub(r"[^a-z0-9;,.:/\- ]", " ", t)
    for deg in ("b tech", "mba tech", "m tech"):
        t = re.sub(rf"\b{deg}\s+({'|'.join(AFTER_DEGREE)})\b", lambda m: f"{deg} {AFTER_DEGREE[m.group(1)]}", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> set[str]:
    return {w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", norm(text))} - STOP


@lru_cache(maxsize=1)
def program_index() -> list[tuple[str, str, str]]:
    """(catalogue name, normalised full name, normalised name without the degree), longest first."""
    out = []
    for p in catalog.programs():
        full = norm(p)
        body = full
        for d in DEGREES:
            if full.startswith(d + " "):
                body = full[len(d) + 1:]
        out.append((p, full, body.replace(" program", "").strip()))
    return sorted(out, key=lambda x: -len(x[1]))


@lru_cache(maxsize=1)
def course_index() -> list[str]:
    names = {norm(c) for c in catalog.load()["course"] if not catalog.SLOT.match(c)}
    return sorted((n for n in names if len(n) > 3), key=len, reverse=True)


def degree_of(program: str) -> str:
    n = norm(program)
    return next((d for d in DEGREES if n.startswith(d)), "")


def level_of(program: str) -> str:
    return "PG" if degree_of(program) in ("m tech", "master of") else "UG"


def find_programs(text: str) -> list[str]:
    """Catalogue programmes named in `text` ("B.Tech. Computer Science, B.Tech. Computer Engineering and ...")."""
    t = norm(text)
    found, taken = [], []
    for name, full, body in program_index():
        for m in re.finditer(rf"\b{re.escape(full)}\b", t):
            if not any(a <= m.start() < b for a, b in taken):
                found.append(name)
                taken.append((m.start(), m.end()))
                break
    # "B.Tech. Computer Science, Computer Engineering and Artificial Intelligence": a degree followed by a list
    for m in re.finditer(r"\b(b tech|mba tech|m tech)\s+([a-z ,]+?(?:\band\b [a-z ]+?)?)(?=[.;:]|$|\bas\b|\bprogram)", t):
        deg = m.group(1)
        for part in re.split(r",|\band\b(?! business| telecommunication| engineering \w)", m.group(2)):
            part = part.strip()
            for name, full, body in program_index():
                if full.startswith(deg) and part == body and name not in found:
                    found.append(name)
    return found


@lru_cache(maxsize=1)
def _course_names() -> dict[str, str]:
    return {norm(c): c for c in catalog.load()["course"]}


def course_name(n: str) -> str:
    """Official spelling of a normalised catalogue course name (capitalised as given if not in the catalogue)."""
    return _course_names().get(n, n[:1].upper() + n[1:])


def find_courses(text: str) -> list[str]:
    """Catalogue course names in `text`, longest first, without overlaps; then leftover comma/'and' phrases."""
    t, out = norm(text), []
    for c in course_index():
        if re.search(rf"\b{re.escape(c)}\b", t):
            out.append(c)
            t = re.sub(rf"\b{re.escape(c)}\b", " | ", t)
    for rest in re.split(r"[,;|.]|\band\b", t):
        rest = re.sub(r"^\s*(a course in|courses? in|basic|knowledge of|completed|done)\s+", "", rest.strip())
        if len(rest.split()) >= 2 and not re.search(r"\b(should|expected|students?|have|you)\b", rest):
            out.append(rest)
    return out


def _num(s: str) -> int | None:
    return int(s) if s.isdigit() else ROMAN.get(s)


def semesters_in(text: str) -> list[int]:
    """'Semester VI', 'Sem 5', 'Semesters V and VI', 'Semester V or VI', 'Semester V-VI', 'III / VI'."""
    t = text.lower()
    if m := re.search(r"year\s*/\s*sem(?:ester)?\s*[-:]?\s*([ivx]+|\d)\s*/\s*([ivx]+|\d+)\b", t):   # "III / VI"
        return [s] if (s := _num(m.group(2))) else []
    sems = []
    for m in re.finditer(r"\bsem(?:ester)?s?\s*[-:]?\s*([ivx]+|\d{1,2})\b((?:\s*(?:,|or|and|to|-)\s*(?:sem(?:ester)?\s*)?"
                         r"(?:[ivx]+|\d{1,2})\b)*)", t):
        first, tail = _num(m.group(1)), m.group(2) or ""
        rest = [_num(x) for x in re.findall(r"\b([ivx]+|\d{1,2})\b", tail)]
        if re.search(r"\bto\b|-", tail) and first and rest and rest[-1]:
            sems += list(range(first, rest[-1] + 1))
        else:
            sems += [first, *rest]
    for m in re.finditer(r"\b(\w+)\s+sem(?:ester)?\b", t):                  # "sixth semester", "6th sem"
        w = m.group(1)
        d = re.sub(r"(st|nd|rd|th)$", "", w)
        sems.append(SEM_WORDS.get(w) or (int(d) if d.isdigit() else None))
    if m := re.search(r"\b([ivx]+|\d)\s*/\s*([ivx]+|\d+)\b", t):        # "Year / Semester: III / VI"
        y = ROMAN.get(m.group(2)) or (int(m.group(2)) if m.group(2).isdigit() else None)
        sems.append(y)
    return sorted({s for s in sems if s and 1 <= s <= 10})


def year_in(text: str) -> int | None:
    t = text.lower()
    if m := re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)[\s-]+(?:year|yr)s?\b", t):
        return ORDINAL[m.group(1)]
    if m := re.search(r"\b(?:year|yr)\s*[-:]?\s*([1-5]|i{1,3}|iv|v)\b", t):
        g = m.group(1)
        return int(g) if g.isdigit() else ROMAN[g]
    if re.search(r"\b(final|last)[\s-]+(?:year|yr)", t):
        return -1                                     # resolved against the programme length
    return None


# --------------------------------------------------------------------------- syllabus

@dataclass
class Syllabus:
    title: str
    text: str
    code: str | None = None
    course_type: str = "unknown"            # program_elective | open_elective | core | unknown
    programs: list[str] = field(default_factory=list)     # catalogue names
    open_to_all: str | None = None          # "UG" / "PG" when offered to all programmes of that level
    level: str | None = None
    semesters: list[int] = field(default_factory=list)
    year: int | None = None
    credits: int | None = None
    prerequisites: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    is_new: bool = False
    found: dict[str, str] = field(default_factory=dict)   # field -> labelled | inferred | missing

    @property
    def parse_confidence(self) -> float:
        return round(sum(self.found.get(f) != "missing" for f in REQUIRED) / len(REQUIRED), 2)


_LABEL = {
    "title": r"course\s+title|course\s+name|title",
    "code": r"course\s+code|code",
    "course_type": r"course\s+type|type|category",
    "programs": r"program\(s\)|programs?|programmes?|offered\s+to|open\s+to",
    "semester": r"year\s*/\s*semester|semester|term",
    "credits": r"credits?",
    "prerequisites": r"pre-?requisites?|pre-?requisite\s+courses?",
}


def _labelled(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        for key, pat in _LABEL.items():
            if key not in out and (m := re.match(rf"^\s*(?:{pat})\s*[:\-–]\s*(.+?)\s*$", line, re.I)):
                out[key] = m.group(1)
    return out


def _type(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bopen\s+elective", t):
        return "open_elective"
    if re.search(r"\b(program|programme|professional|department(al)?)\s+elective", t):
        return "program_elective"
    if re.search(r"\b(compulsory|core)\s+course", t):
        return "core"
    return None


def _title(text: str) -> str | None:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:6]:
        if m := re.match(r"^(?:proposed\s+)?new\s+(?:open\s+|programme\s+|program\s+)?(?:course|elective)\s*"
                         r"(?:\(proposed\))?\s*[:\-–]\s*(.+?)(?:\s*\(\d+\s*credits?\))?$", l, re.I):
            return m.group(1).strip()
    for l in lines[:6]:
        if (len(l.split()) <= 10 and not re.search(r"proposal|outline|nmims|school|board|^course\b", l, re.I)
                and not l.endswith(".")):
            return l
    return None


def parse_syllabus(text: str, name: str | None = None) -> Syllabus:
    lab = _labelled(text)
    prose = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # re-join lines wrapped inside a paragraph
    prose = re.sub(r"\b([BM])\.\s?Tech\.?", r"\1Tech", prose, flags=re.I)   # "B.Tech." is not a sentence end
    sentences = re.split(r"(?<=[.!?])\s+|\n+", prose)
    found: dict[str, str] = {}

    def pick(key, labelled_val, inferred_val):
        if labelled_val:
            found[key] = "labelled"
            return labelled_val
        if inferred_val:
            found[key] = "inferred"
            return inferred_val
        found[key] = "missing"
        return None

    title = pick("title", lab.get("title"), _title(text) or (name and re.sub(r"[_-]+", " ", name).strip()))
    lab_type = lab.get("course_type", "")
    lab_type = _type(lab_type) or ("core" if re.search(r"core|compulsory", lab_type, re.I) else None)
    ctype = pick("course_type", lab_type, _type(text))

    prog_text = lab.get("programs") or ""
    scope_sents = " ".join(s for s in sentences
                           if re.search(r"\b(open to|students of|who can take|offered to|for .*students)\b", s, re.I))
    programs = find_programs(prog_text) if prog_text else find_programs(scope_sents or text)
    all_lvl = None
    scope = (prog_text or scope_sents or text).lower()
    if re.search(r"\ball\s+(?:ug|undergraduate)|undergraduate\s+students\s+of\s+all|ug\s+students\s+of\s+all", scope):
        all_lvl = "UG"
    elif re.search(r"\ball\s+(?:pg|postgraduate)|postgraduate\s+students\s+of\s+all", scope):
        all_lvl = "PG"
    pick("programs", (programs or all_lvl) and prog_text, programs or all_lvl)

    sem_text = lab.get("semester") or ""
    sems = semesters_in(sem_text) if sem_text else semesters_in(scope_sents or text)
    year = year_in(sem_text or scope_sents or text)
    if not sems and year and year > 0:
        sems = [2 * year - 1, 2 * year]
    pick("semesters", sems and sem_text, sems)

    credits = None
    if m := re.search(r"(\d+)", lab.get("credits", "")) or re.search(r"\b(\d+)[\s-]+credits?\b", text, re.I):
        credits = int(m.group(1))

    if "prerequisites" in lab:
        raw = lab["prerequisites"]
        prereqs = [] if re.match(r"^\s*(none|nil|no|-)\b", raw, re.I) else \
            [p.strip(" .") for p in re.split(r";" if ";" in raw else r",", raw) if p.strip(" .")]
        found["prerequisites"] = "labelled"
    else:
        sent = next((s for s in sentences
                     if re.search(r"prerequisite|expected to have|should have (done|completed)|must have (done|completed)"
                                  r"|before taking", s, re.I)), "")
        listed = re.search(r"pre-?requisites?\s*(?:are|is|:)\s*(.+)", sent, re.I)   # "Prerequisites: X; Y."
        if re.search(r"\bno\s+prerequisites?\b", text, re.I) or (
                listed and re.match(r"^\s*(none|nil|no|-)\b", listed.group(1), re.I)):
            prereqs, found["prerequisites"] = [], "inferred"
        elif sent:
            prereqs = [course_name(c) for c in find_courses(listed.group(1) if listed else sent)]
            found["prerequisites"] = "inferred"
        else:
            prereqs, found["prerequisites"] = [], "missing"

    topics = [re.sub(r"^\s*(unit|module)\s*\d+\s*[:.\-]?\s*", "", l, flags=re.I).strip()
              for l in text.splitlines() if re.match(r"^\s*(unit|module|co)\s*\d+", l, re.I)]
    if not topics and (m := re.search(r"(?:topics|covers|introduces[^:.]*|teaches[^:.]*)\s*:\s*(.+?)(?:\.\s|\n\n|$)",
                                      prose, re.I | re.S)):
        topics = [t.strip(" .\n") for t in re.split(r";|,(?![^()]*\))", m.group(1)) if len(t.strip()) > 3]

    level = all_lvl or (level_of(programs[0]) if programs else None)
    return Syllabus(title=title or "Untitled course", text=text, code=lab.get("code"), course_type=ctype or "unknown",
                    programs=programs, open_to_all=all_lvl, level=level, semesters=sems,
                    year=year if year and year > 0 else (sems and (sems[0] + 1) // 2) or None, credits=credits,
                    prerequisites=prereqs, topics=topics,
                    is_new=bool(re.search(r"\bnew\s+(course|elective|open elective)|\bproposed\b", text[:300], re.I)),
                    found=found)
