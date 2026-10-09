"""Phase 2 - core annotations: turn the curated facts and queries (from official NMIMS sources) into the labelled query dataset + NER.

Inputs (curated from the official NMIMS sources):
  data/annotations/facts.yaml    verified facts, one per source passage, each with a verbatim quote
  data/annotations/queries.yaml  queries from the official NMIMS sources, keyed by the fact(s) that answer them, plus ambiguous,
                                 multi-intent and unverified queries
The queries are written by the project team from official NMIMS sources; they are not collected student queries.

Steps:
  queries   labels (intent / sub-intent / department) from the linked facts; required documents and next action
            only as stated by the facts; source metadata from the inventory; program / semester / user type,
            question type and language style from the text; ambiguity, multi-intent, source status, confidence
  ner       rule- and dictionary-based entity spans over each query (silver labels: only entities literally
            present in the text are marked; overlaps resolved by longest match, then TYPE_PRIORITY)

Outputs: data/final/core_queries.csv, data/final/core_ner_annotations.csv
Run:  python -m dataset.phase2_annotate
"""
from __future__ import annotations

import json
import re

import pandas as pd

from dataset.config import (CONTRADICTED_FACTS, FINAL, OCR_SOURCES, SUB_TO_INTENT, TIME_SENSITIVE_SUBS, annotations,
                            load_facts, next_actions)

CORE_QUERIES = FINAL / "core_queries.csv"
CORE_NER = FINAL / "core_ner_annotations.csv"

# ----------------------------------------------------------------------------- text-derived metadata

PROGRAM_PATTERNS = [
    ("MBA Tech", r"\bmba[ -]?tech\b|\bmbatech\b"), ("B Tech", r"\bb[ .-]?tech\b|\bbtech\b|\bengineering\b"),
    ("M Tech", r"\bm[ .-]?tech\b|\bmtech\b"), ("MCA", r"\bmca\b"), ("Diploma", r"\bdiploma\b|\bd\.?voc\b"),
    ("Ph.D", r"\bph\.?\s?d\b"), ("MBA", r"\bmba\b(?![ -]?tech)"),
]
SEM_PATTERN = re.compile(r"\b(?:sem(?:ester)?\s*(\d{1,2}|[ivx]+)|(\d)(?:st|nd|rd|th)\s+sem(?:ester)?)\b", re.I)
YEAR_PATTERN = re.compile(r"\b(first|second|third|fourth|final|1st|2nd|3rd|4th)\s+year\b", re.I)
HINGLISH = re.compile(r"\b(hai|hain|kaise|kya|kab|kitna|kitne|mein|gaya|gayi|ho|kare|karu|karte|chahiye|kaha|milega|phir|bhi|laga|tha|aayega|ke liye|pe|batao|bhai)\b", re.I)
FORMAL = re.compile(r"\b(dear|respected|kindly|could you please|please advise|please let me know|good afternoon|sir/madam)\b", re.I)
URGENT = re.compile(r"\b(urgent|urgently|asap|right now|today|tonight|immediately|in \d+ min)", re.I)
ABBREV = re.compile(r"\b(attnd|attendence|reval|kt|atkt|tee|ica|mtt|cgpa|sem|lib|pyqs?|lor|som|noc|id|sap|cert|certi|dvoc|nsp|asap|pls|plz|abt|wat|hw|reg|dept|docs)\b", re.I)

QUESTION_TYPES = [
    ("REPORT_COMPLAINT", r"\b(complain|report|harass|ragg|discriminat|threaten|hacked|misusing)"),
    ("FEE_COST", r"\b(fee|fees|cost|charges|how much|price|refund|kitne ka|kitna)\b"),
    ("DEADLINE_TIMING", r"\b(when|last date|deadline|till when|how long|how many days|timing|timeline|kab|dates?|time)\b"),
    ("CONTACT_ROUTING", r"\b(who|whom|which office|which department|which committee|contact|email|where (do|should|can) i (send|submit|go))\b"),
    ("ELIGIBILITY", r"\b(can i|am i|eligible|allowed|possible|will i|do i get|is it ok|compulsory|mandatory)\b"),
    ("DOCUMENT_REQUIREMENT", r"\b(documents?|attach|proof|form|format|what to upload)\b"),
    ("PROCEDURE", r"\b(how|process|procedure|steps|apply|kaise|get)\b"),
    ("STATUS_FOLLOWUP", r"\b(still|yet|not received|no update|pending)\b"),
    ("DEFINITION", r"\b(what is|what's|what does|meaning|difference)\b"),
]


def program_of(q: str) -> str:
    return next((name for name, pat in PROGRAM_PATTERNS if re.search(pat, q.lower())), "")


def semester_of(q: str) -> str:
    m = SEM_PATTERN.search(q)
    if m:
        return f"Semester {(m.group(1) or m.group(2)).upper()}"
    y = YEAR_PATTERN.search(q)
    return f"{y.group(1).title()} year" if y else ""


def question_type(q: str) -> str:
    return next((t for t, pat in QUESTION_TYPES if re.search(pat, q.lower())), "STATEMENT_OR_KEYWORD")


_VOCAB: set[str] | None = None


def _vocab() -> set[str]:
    global _VOCAB
    if _VOCAB is None:
        from nltk.corpus import words as nltk_words

        domain = {"nmims", "mpstme", "revaluation", "reexam", "examination", "examinations", "transcripts", "hostel",
                  "hostels", "attendance", "bonafide", "ragging", "scholarship", "digilocker", "biometric", "portal",
                  "readmission", "migration", "verification", "photocopy", "synoptic", "internship", "placements",
                  "electives", "semester", "semesters", "credited", "deducted", "detained", "onedrive", "webopac",
                  "turnitin", "jstor", "github", "aadhaar", "hackathon", "canada", "germany", "mumbai", "andheri",
                  "thane", "bandra", "jaipur", "virar", "diwali", "outlook", "subjects", "grades", "receipts"}
        _VOCAB = {w.lower() for w in nltk_words.words()} | domain
    return _VOCAB


def language_style(q: str) -> str:
    n = len(q.split())
    tags = ["formal" if FORMAL.search(q) else "informal"]
    if n <= 4:
        tags.append("very_short")
    elif n >= 30:
        tags.append("long_detailed")
    for tag, cond in (("hinglish", HINGLISH.search(q)), ("urgent", URGENT.search(q)),
                      ("no_punctuation", not re.search(r"[?.!]$", q.strip())), ("lowercase", q == q.lower() and n > 2),
                      ("abbreviation", ABBREV.search(q)),
                      ("possible_typo_or_slang", any(w not in _vocab() for w in re.findall(r"[a-z]{5,}", q.lower())))):
        if cond:
            tags.append(tag)
    return ";".join(tags)


# ----------------------------------------------------------------------------- NER (silver labels)

W = r"(?<![\w])"
E = r"(?![\w])"

LEXICON: dict[str, list[str]] = {
    "CERTIFICATE": [r"bonafide(?: certificate| letter)?", r"migration cert(?:ificate|i)?", r"provisional (?:degree )?(?:certificate|certi|degree)",
                    r"degree certificate", r"leaving certificate", r"clearance cert(?:ificate)?", r"no dues(?: form)?",
                    r"medical fitness (?:certificate|form)", r"medical cert(?:ificate|i)?", r"caste certificate", r"income certificate",
                    r"letter of recommendation", r"lor", r"percentage letter", r"c?gpa (?:letter|certificate)", r"statement of marks", r"som(?: letter)?",
                    r"duplicate degree(?: certificate)?", r"diploma certificate", r"visa letter", r"railway concession(?: pass| form)?"],
    "DOCUMENT": [r"e-?transcripts?", r"transcripts?", r"grade ?sheets?", r"grade cards?", r"mark ?sheets?", r"answer (?:book|sheet|script)s?",
                 r"fee receipts?", r"hostel (?:application )?form", r"indemnity bond", r"cancelled cheque", r"hall ticket", r"refund form",
                 r"passport", r"visa", r"aadhaar", r"srb", r"student resource book", r"undertaking(?: form)?", r"discharge summary",
                 r"cancellation form", r"absenteeism form", r"(?:id|identity|college|university|student) cards?", r"\bid\b(?= (?:lost|kho|gone|stolen))", r"(?:application|duplicate application) form", r"degree", r"degrees",
                 r"offer letter", r"tuition fee receipt", r"deposit receipt", r"cv"],
    "EXAMINATION": [r"tee(?: lab exam)?", r"end ?sems?(?: exams?)?", r"term end exams?", r"semester end exams?", r"re[- ]?exams?",
                    r"re-?examinations?", r"mid ?term tests?(?: \d| i+)?", r"mtts?(?: \d)?", r"lab exams?", r"viva", r"practicals?",
                    r"online mcq exams?", r"exams?", r"papers?"],
    "ACTION": [r"revaluation", r"re-?valuation", r"reval", r"verification of marks", r"verification", r"re-?totall?ing", r"photocopy",
               r"cancel(?:lation)?", r"withdraw(?:al)?", r"deferment", r"defer", r"correction", r"re-?admission", r"year back",
               r"renew(?:al)?", r"reserve", r"register(?:ation)?", r"apply", r"opt(?:ing)? out", r"night ?outs?", r"academic break", r"gap year"],
    "ISSUE": [r"lost", r"stolen", r"damaged", r"misplaced", r"not working", r"wrong(?:ly)?", r"marked absent", r"absent", r"failed",
              r"locked", r"blocked", r"late", r"short attend[ae]nce", r"attendance (?:is )?short", r"detained", r"ragg(?:ing|ed)",
              r"harassment", r"discrimination", r"stressed", r"anxious", r"panic attack", r"feeling (?:very )?low", r"misspel+t?",
              r"spell(?:ed|ing) (?:wrong|mistake)", r"payment failed", r"not received", r"error", r"glitch", r"hacked", r"expelled",
              r"kho gay[ai]", r"overdue", r"broken", r"fracture", r"hospitali[sz]ed", r"sick", r"unwell"],
    "DEPARTMENT": [r"exam(?:ination)? (?:cell|department|dept|office|section)", r"accounts?(?: department| office)?", r"admission (?:department|dept|office)",
                   r"library", r"hostel office", r"course coordinator", r"\bar\b", r"\bdr\b", r"registrar", r"dean", r"hod",
                   r"placement (?:office|cell)", r"it helpdesk", r"icc", r"internal complaints committee", r"anti[- ]ragging (?:committee|squad)",
                   r"counsell?or", r"psychologist", r"international (?:linkages|office)", r"security", r"grievance (?:cell|committee)",
                   r"women grievance (?:cell|committee)", r"equal opportunity cell", r"ombudsman", r"controller of examinations", r"warden"],
    "SCHOOL": [r"mpstme", r"sbm", r"stme", r"school of business management"],
    "UNIVERSITY": [r"nmims", r"svkm", r"usc", r"cmu", r"iits?", r"university abroad"],
    "PROGRAM": [r"mba[ -]?tech", r"b\.? ?tech", r"m\.? ?tech", r"mca", r"diploma", r"ph\.?d", r"\bmba\b", r"\bms\b", r"masters?", r"d\.?voc",
                r"computer engineering", r"data science", r"electronics and telecommunication", r"engineering"],
    "SUBJECT": [r"data structures(?: and algorithms)?", r"dsa", r"dbms", r"operating systems", r"engineering mathematics", r"maths?(?: \d)?",
                r"physics", r"computer networks", r"machine learning", r"engineering ethics", r"product realization", r"organizational behaviour"],
    "SEMESTER": [r"sem(?:ester)? ?\d{1,2}", r"(?:1st|2nd|3rd|[4-8]th|first|second|third|fourth|fifth|sixth|seventh|eighth) sem(?:ester)?", r"odd sem(?:ester)?", r"even sem(?:ester)?",
                 r"(?:first|second|third|fourth|final|1st|2nd|3rd|4th) year", r"this sem(?:ester)?", r"next sem(?:ester)?"],
    "ACADEMIC_YEAR": [r"20\d\d-\d\d(?:\d\d)?", r"ay 20\d\d-\d\d"],
    "DATE": [r"\d{1,2}(?:st|nd|rd|th)? (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*(?: 20\d\d)?",
             r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?(?:,? 20\d\d)?",
             r"(?:january|february|march|april|june|july|august|september|october|november|december)", r"20[12]\d",
             r"today", r"tomorrow", r"yesterday", r"tonight", r"this weekend", r"weekends?", r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
             r"diwali", r"summer (?:vacation|break)", r"winter (?:vacation|break)"],
    "DEADLINE": [r"last date", r"deadline", r"till when", r"due date", r"time ?line", r"by when"],
    "STUDENT_ID": [r"\b7\d{10}\b", r"\b[a-z]\d{3}\b"],
    "STUDENT_NUMBER": [r"sap (?:id|number|no\.?)", r"student number", r"roll (?:no|number)", r"11-digit (?:sap )?(?:student )?number"],
    "CGPA": [r"cgpa (?:of )?\d(?:\.\d+)?", r"\d\.\d+ cgpa", r"cgpa"],
    "PERCENTAGE": [r"\d{1,3}(?:\.\d+)? ?(?:%|percent)"],
    "GRADE": [r"\b[fd] grades?\b", r"a\+", r"\bo grade\b", r"\bp grade\b", r"\bc grade\b", r"\bab\b", r"a?tkts?", r"\bkts?\b", r"\bf\b(?= grade| in)"],
    "FEE": [r"re[- ]?exam fees?", r"late fees?", r"hostel fees?", r"semester fees?", r"tuition fees?", r"security deposit", r"hostel deposit",
            r"library deposit", r"fees?", r"fines?", r"deposit", r"readmission fee", r"verification (?:fee|charges)", r"courier charges"],
    "AMOUNT": [r"rs\.? ?\d[\d,]*", r"\d[\d,]* ?(?:rs|rupees)\b", r"\b\d{4,6}\b(?= late fee| fee)", r"\b5000\b"],
    "SCHOLARSHIP": [r"nsp(?: scholarship)?", r"pragati scholarship", r"central sector scholarship", r"post matric scholarship",
                    r"minority scholarship", r"merit scholarship", r"pm scholarship", r"scholarships?", r"dean'?s list"],
    "HOSTEL": [r"g\.? ?r\.? jani(?: boys)?(?: hostel)?", r"jani hostel", r"mkm sanghvi(?: girls)?(?: hostel)?", r"bansi villa(?: flats)?",
               r"anand premises(?: boys)?(?: hostel)?", r"girls hostel", r"boys hostel", r"residential flats?"],
    "LOCATION": [r"navi mumbai", r"mumbai", r"vile parle", r"andheri", r"thane", r"bandra", r"virar", r"jaipur", r"delhi", r"pune"],
    "CAMPUS": [r"vile parle campus", r"mumbai campus", r"shirpur"],
    "ORGANIZATION": [r"wes", r"digilocker", r"eca", r"aicte", r"ugc", r"tele-?manas", r"ieee", r"jstor", r"nptel", r"turnitin", r"drillbit",
                     r"springer", r"coursera", r"smart india hackathon", r"sap(?= portal| student)", r"background verification agency"],
    "COUNTRY": [r"canada", r"germany", r"\bus\b", r"usa", r"\buk\b", r"india"],
    "PURPOSE": [r"for (?:my )?(?:masters|ms|mba|phd|ph\.d|higher studies)(?: application)?", r"for (?:a |my )?(?:job|placement|internship)(?: application| purpose)?",
                r"for (?:my )?(?:visa|passport|immigration|pr|canadian pr)", r"for (?:my )?(?:income tax|education loan)(?: filing)?",
                r"for (?:my )?(?:govt|government) job(?: application)?"],
}

TYPE_PRIORITY = ["STUDENT_ID", "AMOUNT", "PERCENTAGE", "CGPA", "PURPOSE", "CERTIFICATE", "SCHOLARSHIP", "HOSTEL", "FEE", "DOCUMENT",
                 "EXAMINATION", "GRADE", "SUBJECT", "PROGRAM", "SEMESTER", "ACADEMIC_YEAR", "DATE", "DEADLINE", "CAMPUS", "LOCATION",
                 "SCHOOL", "UNIVERSITY", "ORGANIZATION", "COUNTRY", "DEPARTMENT", "STUDENT_NUMBER", "ACTION", "ISSUE"]

COMPILED = [(label, re.compile(W + p + E, re.I)) for label, pats in LEXICON.items() for p in pats]


def annotate(text: str) -> list[dict]:
    cands = [(m.start(), m.end(), label) for label, pat in COMPILED for m in pat.finditer(text) if m.end() > m.start()]
    cands.sort(key=lambda c: (-(c[1] - c[0]), TYPE_PRIORITY.index(c[2]), c[0]))
    taken, out = [False] * len(text), []
    for s, e, label in cands:
        if any(taken[s:e]):
            continue
        taken[s:e] = [True] * (e - s)
        out.append({"entity_text": text[s:e], "entity_type": label, "start_position": s, "end_position": e})
    return sorted(out, key=lambda x: x["start_position"])


def entities_json(text: str) -> str:
    return json.dumps([{"text": s["entity_text"], "type": s["entity_type"], "start": s["start_position"],
                        "end": s["end_position"]} for s in annotate(text)], ensure_ascii=False)


# ----------------------------------------------------------------------------- query assembly

def _raw_queries() -> list[tuple]:
    """(query, fact_ids, user_type, kind, ambiguity, alternatives, unverified_sub_intent)."""
    _, qdoc = annotations()
    raw = []
    for key, items in qdoc["core"].items():
        fids = key.split("+")
        for it in items:
            q, role = (it, "student") if isinstance(it, str) else (it["query"], it["user_type"])
            raw.append((q, fids, role, "multi_fact" if len(fids) > 1 else "single", "LOW", [], None))
    raw += [(a["query"], [a["fact"]], "student", "ambiguous", a["ambiguity"], a["alternatives"], None) for a in qdoc["ambiguous"]]
    raw += [(m["query"], m["facts"], "student", "multi_intent", "MEDIUM", [], None) for m in qdoc["multi_intent"]]
    raw += [(u["query"], [], "student", "unverified", "LOW", [], u["sub_intent"]) for u in qdoc["unverified"]]
    return raw


def build_queries() -> pd.DataFrame:
    facts, actions = load_facts(), next_actions()
    inv = pd.read_csv(FINAL / "source_inventory.csv").set_index("source_id")
    rows, seen = [], set()
    for q, fids, role, kind, amb, alts, usub in _raw_queries():
        q = re.sub(r"\s+", " ", q).strip()
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        missing = [f for f in fids if f not in facts]
        if missing:
            raise KeyError(f"unknown fact ids {missing} for query: {q}")
        linked = [facts[f] for f in fids]
        if usub:
            intent, sub, dept, docs, action, src_ids = SUB_TO_INTENT[usub], usub, "", [], "", []
            status, conf, rcv = "UNVERIFIED", "LOW", True
        else:
            subs = list(dict.fromkeys(f["sub_intent"] for f in linked))
            intent = "|".join(dict.fromkeys(f["intent"] for f in linked))
            sub = "|".join(subs)
            dept = "|".join(dict.fromkeys(f["department"] for f in linked))
            docs = list(dict.fromkeys(d for f in linked for d in f["documents"]))
            action = " / ".join(actions[s] for s in subs)
            src_ids = list(dict.fromkeys(f["source_id"] for f in linked))
            status = "VERIFIED"
            flagged = any(f["historical"] or f["source_id"] in OCR_SOURCES or f["fact_id"] in CONTRADICTED_FACTS for f in linked)
            conf = "MEDIUM" if (flagged or kind == "ambiguous") else "HIGH"
            rcv = flagged or any(f["sub_intent"] in TIME_SENSITIVE_SUBS for f in linked)
        year = lambda s: (str(inv.loc[s, "academic_year"]) if pd.notna(inv.loc[s, "academic_year"])
                          else str(inv.loc[s, "publication_date"])[:4])
        rows.append({
            "query": q, "intent": intent, "sub_intent": sub, "department": dept,
            "school": inv.loc[src_ids[0], "school"] if src_ids else "", "campus": "Mumbai (Vile Parle)",
            "program": program_of(q), "semester": semester_of(q), "user_type": role, "entities": entities_json(q),
            "required_documents": json.dumps(docs), "next_action": action, "question_type": question_type(q),
            "language_style": language_style(q), "source_id": "|".join(src_ids),
            "source_url": "|".join(inv.loc[s, "url"] for s in src_ids),
            "source_title": "|".join(inv.loc[s, "title"] for s in src_ids),
            "source_year": "|".join(year(s) for s in src_ids),
            "last_verified": "|".join(dict.fromkeys(inv.loc[s, "last_verified"] for s in src_ids)),
            "fact_ids": "|".join(fids), "answer_facts": " || ".join(f["fact"] for f in linked),
            "ambiguity": amb, "alternative_sub_intents": "|".join(alts),
            "multi_intent": kind == "multi_intent" or (kind == "multi_fact" and "|" in intent),
            "query_kind": kind, "source_status": status, "confidence": conf, "requires_current_verification": rcv,
        })
    df = pd.DataFrame(rows)
    df.insert(0, "group_id", [f"G{i:05d}" for i in range(1, len(df) + 1)])
    return df


def ner_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    rows = [{id_col: i, "query": q, **s} for i, q in zip(df[id_col], df["query"]) for s in annotate(q)]
    return pd.DataFrame(rows, columns=[id_col, "query", "entity_text", "entity_type", "start_position", "end_position"])


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    df = build_queries()
    df.to_csv(CORE_QUERIES, index=False)
    ner = ner_table(df, "group_id")
    ner.to_csv(CORE_NER, index=False)
    print(f"{len(df)} core queries, {len(ner)} entity spans")
    print(df["query_kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
