"""Official programme structures (MPSTME Student Resource Book 2026, SRC052) as a course catalogue.

For every programme and semester the book lists each course with lecture / practical / tutorial hours, credits and
a category: COMP (compulsory), ELEC (programme elective), INTD (interdisciplinary = open elective), COUC (Coursera).
Elective *slots* ("Program Elective - II", "Open Elective - I") say when a student may take an elective; named ELEC
rows under "Program Elective (Select any One)" are the fixed options some programmes list.

  python -m nlp.electives.catalog      # parse data/interim/extracted/SRC052.txt -> data/final/program_structures.csv
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from nlp import DATA

SOURCE_ID = "SRC052"
SOURCE_URL = "https://engineering.nmims.edu/wp-content/uploads/2026/08/SRB_2026_compressed.pdf"
SOURCE_TITLE = "MPSTME Student Resource Book 2026 (Part I + Part II)"
TEXT = DATA / "interim" / "extracted" / f"{SOURCE_ID}.txt"
CSV = DATA / "final" / "program_structures.csv"

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
_PROGRAM = re.compile(r"(?:proposed\s+)?(?:program\s+)?structure\s+of\s+(.+?)\s*$", re.I)
_SEMESTER = re.compile(r"Academic Year\s*[–-]\s*(\d{4}-\d{2}),\s*Semester\s*-\s*([IVX]+)", re.I)
_PAGE = re.compile(r"^=== page (\d+) ===")
_NUM = r"(\d+|[Xx])"
_ROW = re.compile(rf"^(\d+)\s+(.+?)\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+(\d+)\s+(COMP|ELEC|INTD|COUC)\b\s*(.*)$")
_TAIL = re.compile(rf"^(.+?)\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+(\d+)\s+(COMP|ELEC|INTD|COUC)\b\s*(.*)$")
_STARTS_ROW = re.compile(r"^(\d+)\s+([A-Za-z].*)$")
_SELECT_ONE = re.compile(r"program elective\s*\(select any", re.I)
SLOT = re.compile(r"^(program|open)\s+elective\s*[-–—]?\s*([IVX]+|\d+)?\s*$", re.I)


def _clean_program(name: str) -> str:
    name = re.sub(r"\s+", " ", name.replace("(", " (").replace(" )", ")")).strip(" .")
    name = re.sub(r"^B\.?\s?Tech\.?", "B Tech", name)
    return re.sub(r"\s+\(", " (", name)


def parse(text: str) -> pd.DataFrame:
    rows, program, year, sem, page, pending, select_group = [], None, None, None, None, None, False
    for raw in text.splitlines():
        line = raw.strip()
        if m := _PAGE.match(line):
            page = int(m.group(1))
            continue
        if (m := _PROGRAM.search(line)) and "batch" not in line.lower():
            program, sem, pending = _clean_program(m.group(1)), None, None
            continue
        if m := _SEMESTER.search(line):
            year, sem, pending, select_group = m.group(1), ROMAN[m.group(2).upper()], None, False
            continue
        if program is None or sem is None or not line:
            continue
        if _SELECT_ONE.search(line):
            select_group = True
            continue
        if line.startswith("Total"):
            pending, select_group = None, False
            continue
        m = _ROW.match(line)
        if not m and pending and (t := _TAIL.match(line)):          # second half of a wrapped course name
            groups = (pending[0], f"{pending[1]} {t.group(1)}", *t.groups()[1:])
        elif m:
            groups = m.groups()
        else:
            s = _STARTS_ROW.match(line)
            pending = (s.group(1), s.group(2)) if s else None
            continue
        pending = None
        sno, name, lec, prac, tut, credits, cat, remark = groups
        name = re.sub(r"\s+", " ", name).strip()
        slot = SLOT.match(name)
        rows.append({"program": program, "semester": sem, "year_of_study": (sem + 1) // 2, "academic_year": year,
                     "s_no": int(sno), "course": name, "lecture": lec, "practical": prac, "tutorial": tut,
                     "credits": int(credits), "category": cat, "remark": remark.strip(),
                     "kind": ("program_elective_slot" if slot and slot.group(1).lower() == "program" else
                              "open_elective_slot" if slot else
                              "named_elective" if cat == "ELEC" else "course"),
                     "select_one_group": select_group and cat == "ELEC" and not slot,
                     "source_id": SOURCE_ID, "page": page})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["program", "semester", "s_no", "course"]).reset_index(drop=True)


def build() -> pd.DataFrame:
    df = parse(TEXT.read_text(encoding="utf-8"))
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


# --------------------------------------------------------------------------- lookup

@dataclass(frozen=True)
class CatalogCourse:
    program: str
    semester: int
    course: str
    credits: int
    category: str
    kind: str
    page: int


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, keep_default_na=False)
    return df


def programs() -> list[str]:
    return sorted(load()["program"].unique())


def semester_rows(program: str, semester: int) -> pd.DataFrame:
    df = load()
    return df[(df["program"] == program) & (df["semester"] == semester)]


def courses_before(program: str, semester: int) -> list[CatalogCourse]:
    """Compulsory courses of every earlier semester: what a student in `semester` has normally completed."""
    df = load()
    d = df[(df["program"] == program) & (df["semester"] < semester) & (df["kind"] == "course")]
    return [CatalogCourse(r.program, int(r.semester), r.course, int(r.credits), r.category, r.kind, int(r.page))
            for r in d.itertuples()]


def courses_in(program: str, semesters: list[int], kinds=("course",)) -> list[CatalogCourse]:
    df = load()
    d = df[(df["program"] == program) & df["semester"].isin(semesters) & df["kind"].isin(kinds)]
    return [CatalogCourse(r.program, int(r.semester), r.course, int(r.credits), r.category, r.kind, int(r.page))
            for r in d.itertuples()]


if __name__ == "__main__":
    out = build()
    print(f"{len(out)} rows, {out['program'].nunique()} programmes -> {CSV}")
    print(out.groupby("program")["semester"].agg(["min", "max", "count"]))
    print(out["kind"].value_counts())
