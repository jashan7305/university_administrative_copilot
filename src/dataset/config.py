"""Shared configuration for the NMIMS dataset pipeline: paths, taxonomy, departments, sources.

Everything the four phase modules need to agree on lives here, so each phase file contains only its own steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES_DIR = DATA / "sources"          # raw HTML/PDF (gitignored, re-downloadable)
INTERIM = DATA / "interim"              # extracted text + logs
EXTRACTED = INTERIM / "extracted"
FETCH_LOG = INTERIM / "fetch_log.csv"
EXTRACTION_LOG = INTERIM / "extraction_log.csv"
ANNOTATIONS = DATA / "annotations"      # curated inputs from official NMIMS sources: facts.yaml, queries.yaml
FINAL = DATA / "final"                  # released dataset files
REPORTS = ROOT / "reports" / "dataset"

SEED = 42

# ----------------------------------------------------------------------------- taxonomy and departments

TAXONOMY: dict[str, list[str]] = {
    "CERTIFICATES": ["BONAFIDE_CERTIFICATE", "PROVISIONAL_DEGREE", "DEGREE_CERTIFICATE", "DUPLICATE_DEGREE",
                     "CERTIFICATE_CORRECTION", "CLEARANCE_CERTIFICATE", "MIGRATION_CERTIFICATE",
                     "EDUCATION_VERIFICATION", "LETTER_OF_RECOMMENDATION", "ACADEMIC_DOCUMENTS", "DIGILOCKER"],
    "TRANSCRIPTS": ["TRANSCRIPT", "STATEMENT_OF_MARKS", "PERCENTAGE_LETTER", "GPA_CGPA_LETTER",
                    "DUPLICATE_GRADE_SHEET", "WES"],
    "EXAMINATIONS": ["EXAM_TIMETABLE", "EXAM_ELIGIBILITY", "EXAM_RESULT", "RE_EXAMINATION", "REVALUATION",
                     "MARK_VERIFICATION", "ANSWER_BOOK_COPY", "EXAM_GRIEVANCE", "SCRIBE_WRITER", "EXAM_MISCONDUCT",
                     "EXAM_ABSENCE", "BACKLOG_EXAM"],
    "FEES": ["FEE_PAYMENT", "FEE_RECEIPT", "LATE_FEE", "FEE_REFUND", "EXCESS_FEE_REFUND", "ADMISSION_CANCELLATION",
             "FAILED_PAYMENT", "EDUCATION_LOAN"],
    "SCHOLARSHIPS": ["SCHOLARSHIP_ELIGIBILITY", "SCHOLARSHIP_APPLICATION", "SCHOLARSHIP_STATUS",
                     "SCHOLARSHIP_RENEWAL"],
    "ATTENDANCE": ["ATTENDANCE_REQUIREMENT", "LOW_ATTENDANCE", "MEDICAL_ATTENDANCE", "ATTENDANCE_CORRECTION",
                   "ATTENDANCE_EXEMPTION", "ATTENDANCE_APPEAL", "ATTENDANCE_RECORD"],
    "ACADEMIC_REGISTRATION": ["COURSE_REGISTRATION", "ELECTIVE_SELECTION", "COURSE_CHANGE", "ACADEMIC_CALENDAR",
                              "ACADEMIC_GUIDELINES", "STUDENT_STATUS", "ACADEMIC_BANK_OF_CREDITS", "COURSE_ADD_DROP"],
    "ID_CARDS": ["ID_CARD_APPLICATION", "DUPLICATE_ID_CARD", "ID_CARD_ACCESS", "ID_CARD_NOT_WORKING",
                 "ID_CARD_CORRECTION"],
    "HOSTEL": ["HOSTEL_APPLICATION", "HOSTEL_ELIGIBILITY", "HOSTEL_DOCUMENTS", "HOSTEL_ALLOCATION", "HOSTEL_FEES",
               "HOSTEL_RULES", "HOSTEL_VACATING", "HOSTEL_REFUND", "HOSTEL_PROBLEMS"],
    "LIBRARY": ["LIBRARY_TIMINGS", "BOOK_BORROWING", "BOOK_RENEWAL", "LIBRARY_FINE", "LOST_BOOK", "DIGITAL_RESOURCES",
                "LIBRARY_ACCESS", "LIBRARY_RULES"],
    "STUDENT_SERVICES": ["STUDENT_PORTAL", "FACILITIES", "MEDICAL_FITNESS", "COUNSELLING", "STUDENT_ACTIVITIES",
                         "STUDENT_COUNCIL", "PLACEMENT", "INTERNATIONAL_EXCHANGE", "INTERNATIONAL_DOCUMENTS"],
    "STUDENT_WELFARE": ["STUDENT_GRIEVANCE", "ANTI_RAGGING", "INTERNAL_COMPLAINTS", "WOMEN_GRIEVANCE",
                        "EQUAL_OPPORTUNITY", "SC_ST_SUPPORT", "STUDENT_SAFETY", "MENTAL_HEALTH_SUPPORT"],
}
SUB_TO_INTENT = {s: i for i, subs in TAXONOMY.items() for s in subs}

# Sub-intents named in the brief for which no official NMIMS source was found during research.
UNSOURCED_SUB_INTENTS = {"FAILED_PAYMENT", "EDUCATION_LOAN", "SCHOLARSHIP_STATUS", "SCHOLARSHIP_RENEWAL",
                         "COURSE_ADD_DROP", "ID_CARD_CORRECTION"}

DEPARTMENTS = {
    "EXAMINATION_DEPARTMENT": "Examination Department",
    "CONTROLLER_OF_EXAMINATIONS": "Controller of Examinations",
    "ADMISSION_DEPARTMENT": "Admission Department",
    "ACCOUNTS_DEPARTMENT": "Accounts Department",
    "SCHOOL_ACADEMIC_OFFICE": "School Academic Office (Course Coordinator / AR / DR)",
    "DEAN_DIRECTOR": "Dean / Director of School",
    "FACULTY": "Faculty / Faculty Mentor",
    "HOSTEL_OFFICE": "Hostel Office (Mumbai)",
    "LIBRARY": "Library",
    "IT_HELPDESK": "IT Helpdesk / Computer Centre",
    "STUDENT_PORTAL_TEAM": "Student Portal Team",
    "SAP_BASIS_TEAM": "SAP Basis Team",
    "SECURITY_ID": "Security / Biometric Registration",
    "GRIEVANCE_COMMITTEE": "Student Grievance Redressal Committee / Ombudsman",
    "ICC_WOMEN_CELL": "Internal Complaints Committee / Women Grievance Redressal Cell",
    "ANTI_RAGGING_COMMITTEE": "Anti-Ragging Committee",
    "EQUAL_OPPORTUNITY_CELL": "SC/ST/OBC & Equal Opportunity Cell",
    "COUNSELLING": "Counselling (Psychologist)",
    "INTERNATIONAL_LINKAGES": "Department of International Linkages",
    "PLACEMENT_OFFICE": "Placement Office",
    "STUDENT_COUNCIL": "Student Council",
    "REGISTRAR_OFFICE": "Office of the Registrar",
    "EXTERNAL_HELPLINE": "External / National Helpline",
}

RAW_DEPARTMENT = {
    "AR / DR; Accounts": "SCHOOL_ACADEMIC_OFFICE", "Academic grievance cell": "GRIEVANCE_COMMITTEE",
    "Academic office": "SCHOOL_ACADEMIC_OFFICE", "Accounts": "ACCOUNTS_DEPARTMENT",
    "Accounts department": "ACCOUNTS_DEPARTMENT", "Admission department": "ADMISSION_DEPARTMENT",
    "Admission department; School examination department": "ADMISSION_DEPARTMENT",
    "All school departments": "SCHOOL_ACADEMIC_OFFICE", "Anti-Ragging Committee": "ANTI_RAGGING_COMMITTEE",
    "Biometric registration (SVKM)": "SECURITY_ID", "Class Representative": "SCHOOL_ACADEMIC_OFFICE",
    "Computer Centre": "IT_HELPDESK", "Concerned faculty": "FACULTY",
    "Controller of Examinations": "CONTROLLER_OF_EXAMINATIONS", "Counselling team": "COUNSELLING",
    "Course coordinator": "SCHOOL_ACADEMIC_OFFICE", "Course coordinator / AR / DR": "SCHOOL_ACADEMIC_OFFICE",
    "Course coordinator / Program chair": "SCHOOL_ACADEMIC_OFFICE",
    "Course coordinator; AR/DR; HOD/Dean": "SCHOOL_ACADEMIC_OFFICE", "Dean": "DEAN_DIRECTOR",
    "Dean / Director": "DEAN_DIRECTOR", "Dean / Director / HOD": "DEAN_DIRECTOR",
    "Dean / Director; Examination section": "EXAMINATION_DEPARTMENT", "Dean MPSTME": "DEAN_DIRECTOR",
    "Department of International Linkages": "INTERNATIONAL_LINKAGES",
    "Director – International Linkages": "INTERNATIONAL_LINKAGES", "Equal Opportunity Cell": "EQUAL_OPPORTUNITY_CELL",
    "Examination department": "EXAMINATION_DEPARTMENT", "Examination office": "EXAMINATION_DEPARTMENT",
    "External helpline": "EXTERNAL_HELPLINE", "Faculty / LMS": "FACULTY",
    "Faculty in charge of Student Activity / HOD": "FACULTY", "Faculty mentor": "FACULTY",
    "Head of Department": "SCHOOL_ACADEMIC_OFFICE", "Home school": "SCHOOL_ACADEMIC_OFFICE",
    "Hostel in-charge": "HOSTEL_OFFICE", "Hostel office": "HOSTEL_OFFICE", "Hostel office (Mumbai)": "HOSTEL_OFFICE",
    "Hostel office (Registrar)": "HOSTEL_OFFICE", "IT": "IT_HELPDESK", "IT Helpdesk": "IT_HELPDESK",
    "IT security": "IT_HELPDESK", "Internal Complaints Committee": "ICC_WOMEN_CELL",
    "International Office": "INTERNATIONAL_LINKAGES", "Library": "LIBRARY", "MPSTME": "SCHOOL_ACADEMIC_OFFICE",
    "MPSTME Counsellor": "COUNSELLING", "MPSTME Library": "LIBRARY", "Management": "DEAN_DIRECTOR",
    "Multiple departments": "SCHOOL_ACADEMIC_OFFICE", "NUSC": "STUDENT_COUNCIL",
    "National Anti-Ragging Helpline": "EXTERNAL_HELPLINE", "Office of the Registrar": "REGISTRAR_OFFICE",
    "Ombudsman": "GRIEVANCE_COMMITTEE", "Placement Office": "PLACEMENT_OFFICE", "Portal team": "STUDENT_PORTAL_TEAM",
    "Psychologist / Counsellor": "COUNSELLING", "Registrar": "REGISTRAR_OFFICE",
    "Registrar / Deputy Registrar": "HOSTEL_OFFICE", "Registrar / Rector": "HOSTEL_OFFICE",
    "SAP Basis team": "SAP_BASIS_TEAM", "SAP Basis team; School examination office": "SAP_BASIS_TEAM",
    "SC/ST/OBC & Equal Opportunity Cell": "EQUAL_OPPORTUNITY_CELL", "School": "SCHOOL_ACADEMIC_OFFICE",
    "School Examination Office": "EXAMINATION_DEPARTMENT", "School administration": "SCHOOL_ACADEMIC_OFFICE",
    "School examination department": "EXAMINATION_DEPARTMENT", "Security": "SECURITY_ID",
    "Student Grievance Redressal Committee": "GRIEVANCE_COMMITTEE",
    "Student Grievances Redressal Cell": "GRIEVANCE_COMMITTEE", "Student support services": "COUNSELLING",
    "Student welfare": "GRIEVANCE_COMMITTEE", "Unfair Means Inquiry Committee": "EXAMINATION_DEPARTMENT",
    "University": "SCHOOL_ACADEMIC_OFFICE", "University Admission Department": "ADMISSION_DEPARTMENT",
    "University Student Grievance Redressal Committee": "GRIEVANCE_COMMITTEE",
    "Vice-Chancellor's office": "DEAN_DIRECTOR", "Women Grievance Redressal Cell": "ICC_WOMEN_CELL",
}


# ----------------------------------------------------------------------------- sources

@dataclass(frozen=True)
class Source:
    url: str
    category: str          # sources/<category>/ folder
    school: str
    campus: str
    title: str
    source_type: str       # WEBPAGE | PDF types from the brief (FAQ, FORM, POLICY, ...)
    academic_year: str     # as stated by the document itself, else ""
    currency: str          # CURRENT | HISTORICAL
    relevance: str         # HIGH | MEDIUM | LOW
    notes: str


# Official NMIMS sources: the 20 seed URLs from the brief plus relevant NMIMS-domain pages linked from them.
# Metadata was written after reading each source.
SOURCES: dict[str, Source] = {
    'SRC001': Source('https://www.nmims.edu/students', 'other', 'NMIMS (all schools)', 'All',
                    'NMIMS Students hub page', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Link hub; little body text'),
    'SRC002': Source('https://www.nmims.edu/examination', 'examinations', 'NMIMS (all schools)', 'All',
                    'NMIMS Examinations page (procedures, FAQs, document timelines)', 'FAQ', '', 'CURRENT', 'HIGH', 'Core exam/document procedures; some FAQ content conflicts with SRC003 and SRB 2026 (see contradictions)'),
    'SRC003': Source('https://www.nmims.edu/faq', 'other', 'NMIMS (all schools)', 'All',
                    'NMIMS FAQ page', 'FAQ', '', 'CURRENT', 'HIGH', 'Overlaps SRC002 with different fee figures for name correction'),
    'SRC004': Source('https://nmims.edu/docs/faqs-and-examination-related-documents-procedures-and-timelines.pdf', 'examinations', 'NMIMS (all schools)', 'All',
                    'FAQs and examination related documents, procedures and timelines (PDF)', 'FAQ', '', 'HISTORICAL', 'MEDIUM', 'PDF created 2018; superseded by SRC002/SRC003 web content'),
    'SRC005': Source('https://www.nmims.edu/duplicate-grade-sheets', 'certificates', 'NMIMS (all schools)', 'All',
                    'Duplicate or Correction in Degree / Grade Sheets', 'PROCEDURE', '', 'CURRENT', 'HIGH', ''),
    'SRC006': Source('https://nmims.edu/education-verification', 'certificates', 'NMIMS (all schools)', 'All',
                    'Procedure for Education Verification', 'PROCEDURE', '', 'CURRENT', 'HIGH', 'Charges revised 13 Oct 2025'),
    'SRC007': Source('https://www.nmims.edu/hostel-mumbai-form.php', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'Revised Hostel Application Process (Mumbai)', 'PROCEDURE', '', 'CURRENT', 'HIGH', 'Hostel portal temporarily closed; email-based process'),
    'SRC008': Source('https://upload.nmims.edu/admission/cancellation/', 'fees', 'NMIMS (all schools)', 'All',
                    'Online Admission Cancellation', 'PROCEDURE', '2026-27', 'CURRENT', 'HIGH', 'Full refund for technical programmes until 20 July 2026'),
    'SRC009': Source('https://www.nmims.edu/student-welfare.php', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Student Welfare page', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Link hub to welfare policies'),
    'SRC010': Source('https://nmims.edu/library', 'library', 'NMIMS (all schools)', 'Mumbai',
                    'NMIMS Library (Prof. Y. K. Bhushan IKRC)', 'WEBPAGE', '', 'CURRENT', 'HIGH', 'Main NMIMS library, Mumbai'),
    'SRC011': Source('https://www.nmims.edu/apply-verification-revaluation', 'examinations', 'NMIMS (all schools)', 'All',
                    'Photocopy / Verification / Revaluation under Grievance Redressal', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Link page to SRC028/SRC029'),
    'SRC012': Source('https://www.nmims.edu/application-for-e-transcript-delivery', 'certificates', 'NMIMS (all schools)', 'All',
                    'Application for E-Transcript Delivery', 'PROCEDURE', '', 'CURRENT', 'HIGH', ''),
    'SRC013': Source('https://www.nmims.edu/admission-cancellation', 'fees', 'NMIMS (all schools)', 'All',
                    'Admission Cancellation (nmims.edu)', 'PROCEDURE', '2026-27', 'CURRENT', 'HIGH', 'Duplicate of SRC008 content'),
    'SRC014': Source('https://www.nmims.edu/students-academic', 'academic', 'NMIMS (all schools)', 'All',
                    'Students - Academic', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Mainly SBM-oriented descriptive content'),
    'SRC015': Source('https://www.nmims.edu/students-international', 'other', 'NMIMS (all schools)', 'All',
                    'Students - International', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Partner list differs from SRB 2026 MPSTME list'),
    'SRC016': Source('https://www.nmims.edu/department-of-welfare', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Department of Welfare', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Duplicate of SRC009'),
    'SRC017': Source('https://www.nmims.edu/national-ragging-prevention-program.php', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'National Ragging Prevention Programme', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', ''),
    'SRC018': Source('https://www.nmims.edu/examination-management-reforms', 'examinations', 'NMIMS (all schools)', 'All',
                    'Examination Management Reforms @ NMIMS', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'DigiLocker / NAD, onscreen evaluation'),
    'SRC019': Source('https://nmims.edu/gr-jani-boys-hostel-non-ac.php', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'G.R. Jani Boys Hostel page', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Images and form link only'),
    'SRC020': Source('https://nmims.edu/mkm-sanghvi-girls-hostel-ac.php', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'MKM Sanghvi Girls Hostel page', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Images and form link only'),
    'SRC021': Source('https://nmims.edu/docs/2025/Mumbai-Campus-Updated.pdf', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'Hostel Application Form - Mumbai Campus (rules, fees 2025-26)', 'FORM', '2025-26', 'CURRENT', 'HIGH', 'Fee table is for AY 2025-26; 2026-27 fees not published'),
    'SRC022': Source('https://www.nmims.edu/docs/scribe-writer.pdf', 'examinations', 'NMIMS (all schools)', 'All',
                    'Scribe/Writer application form', 'FORM', '', 'CURRENT', 'HIGH', 'PDF dated 2014; same content as SRB 2026 Annexure 6'),
    'SRC023': Source('https://www.nmims.edu/docs/SVKM-NMIMS-Re-exam-Rules-and-online-application-Process-document.pdf', 'examinations', 'NMIMS (all schools)', 'All',
                    'Re-Exam Rules and Online Application Process', 'PROCEDURE', '2026-27', 'CURRENT', 'HIGH', 'Internal inconsistency: portal closes 4.00 pm (p1) vs 11:59 PM (p2)'),
    'SRC024': Source('https://www.nmims.edu/docs/Duplicate-Correction-in-GS-Certificate-Format.pdf', 'certificates', 'NMIMS (all schools)', 'All',
                    'Application form for Duplicate / Correction in Degree Certificate / Grade Sheets', 'FORM', '', 'CURRENT', 'HIGH', ''),
    'SRC025': Source('https://www.nmims.edu/docs/proforma-indemnity-bond new.pdf', 'certificates', 'NMIMS (all schools)', 'All',
                    'Proforma of Indemnity Bond', 'FORM', '', 'CURRENT', 'HIGH', ''),
    'SRC026': Source('https://www.nmims.edu/docs/Refund Form-RTGS-REVISED.pdf', 'fees', 'NMIMS (all schools)', 'All',
                    'Application for Refund (RTGS)', 'FORM', '', 'CURRENT', 'HIGH', 'Hostel deposit signatory differs from SRB 2026 Annexure 8'),
    'SRC027': Source('https://www.nmims.edu/docs/2025/Medical-Certificate-Format.pdf', 'other', 'NMIMS (all schools)', 'All',
                    'Medical Fitness Certificate format', 'FORM', '', 'CURRENT', 'MEDIUM', 'PDF dated 2012'),
    'SRC028': Source('https://www.nmims.edu/docs/2026/Revised_Revaluation_Process_Guide 2025-26.pdf.pdf', 'examinations', 'NMIMS (all schools)', 'All',
                    'Your Guide to the Exam Revaluation Process (2025-26)', 'PROCEDURE', '2025-26', 'CURRENT', 'HIGH', 'Scanned; OCR text. Contains default portal password - deliberately excluded from facts'),
    'SRC029': Source('https://www.nmims.edu/docs/2025/FAQ-for-NEW-Revaluation-process.pdf', 'examinations', 'NMIMS (all schools)', 'All',
                    'FAQ for NEW Revaluation process', 'FAQ', '2025-26', 'CURRENT', 'HIGH', ''),
    'SRC030': Source('https://www.nmims.edu/docs/guidelines-on-safety-of-students--nmims.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'UGC guidelines on safety of students', 'POLICY', '', 'CURRENT', 'LOW', 'UGC document, general'),
    'SRC031': Source('https://www.nmims.edu/docs/2026/Circular-Composition-of-Students-Grievance-Redresal-Committee.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Circular: Composition of Student Grievance Redressal Committee (30 Sep 2026)', 'NOTICE', '2026-27', 'CURRENT', 'MEDIUM', 'Supersedes committee listed in SRB 2026 Part I'),
    'SRC032': Source('https://www.nmims.edu/docs/student-grievances-redressal-policy-2015.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Student Grievances Redressal Policy, 2015', 'POLICY', '', 'CURRENT', 'HIGH', '2015 policy; no newer version found'),
    'SRC033': Source('https://www.nmims.edu/docs/2026/Internal-Complaints-Committee.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Circular: Internal Complaints Committee (27 Jul 2026)', 'NOTICE', '2026-27', 'CURRENT', 'MEDIUM', ''),
    'SRC034': Source('https://www.nmims.edu/docs/women-grievance-redressal-cell-policy.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Women Grievance Redressal Cell Policy', 'POLICY', '', 'CURRENT', 'MEDIUM', 'PDF dated 2014'),
    'SRC035': Source('https://www.nmims.edu/docs/prevention-of-sexual-harrasment.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Manual on Prevention of Sexual Harassment', 'POLICY', '', 'CURRENT', 'LOW', 'PDF dated 2014'),
    'SRC036': Source('https://www.nmims.edu/docs/7.1.1 FACILITIES for WOMEN at NMIMS pdf.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Facilities for Women at NMIMS', 'OTHER', '', 'CURRENT', 'LOW', 'Scanned, mostly images'),
    'SRC037': Source('https://www.nmims.edu/docs/2026/SCST-Committee.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Circular: SC/ST/OBC & Equal Opportunity Cell Committee (13 Dec 2025)', 'NOTICE', '2025-26', 'CURRENT', 'MEDIUM', ''),
    'SRC038': Source('https://www.nmims.edu/docs/undertaking-by-students.pdf', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'MHRD circular on online anti-ragging undertaking (2013)', 'NOTICE', '', 'HISTORICAL', 'LOW', '2013 circular; scanned'),
    'SRC039': Source('https://www.nmims.edu/docs/2026/NUSC-2025-26.pdf', 'other', 'NMIMS (all schools)', 'All',
                    'NMIMS University Student Council 2025-26', 'OTHER', '2025-26', 'CURRENT', 'LOW', ''),
    'SRC040': Source('https://www.nmims.edu/Gen-AI-Policy.php', 'other', 'NMIMS (all schools)', 'All',
                    'NMIMS Gen AI Policy', 'POLICY', '', 'CURRENT', 'LOW', 'Page content loaded by script; no text extracted'),
    'SRC041': Source('https://engineering.nmims.edu/students/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Students page', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Link hub'),
    'SRC042': Source('https://engineering.nmims.edu/students/examination/', 'examinations', 'MPSTME', 'Mumbai',
                    'MPSTME Examination page', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Link hub'),
    'SRC043': Source('https://engineering.nmims.edu/students/student-resource-book/', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Student Resource Book page', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', 'Links SRB 2024/2025/2026'),
    'SRC044': Source('https://engineering.nmims.edu/academics/academic-calendar/', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Academic Calendar page', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', ''),
    'SRC045': Source('https://engineering.nmims.edu/academics/infrastructure/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Infrastructure', 'WEBPAGE', '', 'CURRENT', 'LOW', ''),
    'SRC046': Source('https://engineering.nmims.edu/about-us/administration/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Administration', 'WEBPAGE', '', 'CURRENT', 'LOW', 'HOD names differ from SRB 2026 for IT and Mechatronics'),
    'SRC047': Source('https://engineering.nmims.edu/mandatory-disclosure/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Mandatory Disclosure', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Links audited statements/faculty handbook'),
    'SRC048': Source('https://engineering.nmims.edu/students/counselling/', 'student_welfare', 'MPSTME', 'Mumbai',
                    'MPSTME Counselling', 'WEBPAGE', '', 'CURRENT', 'MEDIUM', ''),
    'SRC049': Source('https://engineering.nmims.edu/students/facilities/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Facilities', 'WEBPAGE', '', 'CURRENT', 'LOW', ''),
    'SRC050': Source('https://engineering.nmims.edu/library/', 'library', 'MPSTME', 'Mumbai',
                    'MPSTME Library', 'WEBPAGE', '', 'CURRENT', 'HIGH', ''),
    'SRC051': Source('https://engineering.nmims.edu/students/student-council/', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Student Council', 'WEBPAGE', '', 'CURRENT', 'LOW', ''),
    'SRC052': Source('https://engineering.nmims.edu/wp-content/uploads/2026/08/SRB_2026_compressed.pdf', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Student Resource Book 2026 (Part I + Part II)', 'STUDENT_RESOURCE_BOOK', '2026-27', 'CURRENT', 'HIGH', 'Primary source; effective July 2026'),
    'SRC053': Source('https://engineering.nmims.edu/wp-content/uploads/2025/09/SRB-2025-1.pdf', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Student Resource Book 2025', 'STUDENT_RESOURCE_BOOK', '2025-26', 'HISTORICAL', 'MEDIUM', 'Superseded by SRB 2026'),
    'SRC054': Source('https://engineering.nmims.edu/wp-content/uploads/2024/08/SRB-2024.pdf', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Student Resource Book 2024', 'STUDENT_RESOURCE_BOOK', '2024-25', 'HISTORICAL', 'MEDIUM', 'Superseded; older attendance rule (70-80% exemption)'),
    'SRC055': Source('https://engineering.nmims.edu/wp-content/uploads/2026/05/Academic-Calendar-All-Programs_Firstyear_-AY-2026-27.pdf', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Academic Calendar AY 2026-27 (First year)', 'ACADEMIC_CALENDAR', '2026-27', 'CURRENT', 'HIGH', 'Scanned; OCR; dates subject to change'),
    'SRC056': Source('https://engineering.nmims.edu/wp-content/uploads/2026/05/Academic-Calendar-All-Programs_SecondYear-Six-Year_AY-2026-27.pdf', 'academic', 'MPSTME', 'Mumbai',
                    'MPSTME Academic Calendar AY 2026-27 (Second year onwards)', 'ACADEMIC_CALENDAR', '2026-27', 'CURRENT', 'HIGH', 'Scanned; OCR'),
    'SRC057': Source('https://engineering.nmims.edu/wp-content/uploads/2026/04/Faculty-Handbook-2021.pdf', 'other', 'MPSTME', 'Mumbai',
                    'MPSTME Faculty Handbook 2021', 'HANDBOOK', '2021-22', 'HISTORICAL', 'LOW', 'Faculty-facing; not used for student facts'),
    'SRC058': Source('https://navimumbai.nmims.edu/docs/SRB-Part-I-University-2025-26.pdf', 'academic', 'NMIMS (all schools)', 'All',
                    'SRB Part I - University 2025-26', 'STUDENT_RESOURCE_BOOK', '2025-26', 'HISTORICAL', 'MEDIUM', 'University-wide Part I; superseded by 2026 Part I'),
    'SRC059': Source('https://navimumbai.nmims.edu/docs/SRB-Part-I-University-2024-25.pdf', 'academic', 'NMIMS (all schools)', 'All',
                    'SRB Part I - University 2024-25', 'STUDENT_RESOURCE_BOOK', '2024-25', 'HISTORICAL', 'MEDIUM', 'Lists bonafide/railway concession in portal services'),
    'SRC060': Source('https://navimumbai.nmims.edu/students/students-resource-book/', 'other', 'NMIMS (all schools)', 'All',
                    'Student Resource Books index (Navi Mumbai site)', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Other-campus school SRBs not collected (not Mumbai)'),
    'SRC061': Source('https://www.nmims.edu/statement-of-marks-percentage-letter', 'certificates', 'NMIMS (all schools)', 'All',
                    'Statement of Marks (Percentage Letter)', 'PROCEDURE', '', 'CURRENT', 'HIGH', ''),
    'SRC062': Source('https://www.nmims.edu/wes-process', 'certificates', 'NMIMS (all schools)', 'All',
                    'WES Process', 'PROCEDURE', '', 'CURRENT', 'HIGH', ''),
    'SRC063': Source('https://www.nmims.edu/time-tables-engineering', 'examinations', 'MPSTME', 'Mumbai',
                    'Exam Time Tables - Engineering', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Timetables are posted as linked notices; no body text'),
    'SRC064': Source('https://www.nmims.edu/mental-health-support.php', 'student_welfare', 'NMIMS (all schools)', 'All',
                    'Mental Health Support', 'WEBPAGE', '', 'CURRENT', 'HIGH', ''),
    'SRC065': Source('https://nmims.edu/anand-premises-boys.php', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'Anand Premises Boys hostel page', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Broken link on nmims.edu (HTTP 404)'),
    'SRC066': Source('https://nmims.edu/bansi-villa-girls-residential-flats.php', 'hostel', 'NMIMS (all schools)', 'Mumbai',
                    'Bansi Villa Girls Residential Flats page', 'WEBPAGE', '', 'CURRENT', 'LOW', 'Images and form link only'),
}

# Contradictions and duplicates found while reading: (topic, sources, detail, resolution used in the dataset)
CONTRADICTIONS = [
    ('Name-correction fee', 'SRC002 vs SRC003', 'SRC002 states Rs. 1000 per grade sheet and Rs. 2000; SRC003 states Rs. 200 per grade sheet and Rs. 500.', 'Neither used as a fact; the duplicate/correction fee in SRC005 and SRC024 (Rs. 2000 degree, Rs. 1000 per grade sheet) is used'),
    ('Re-exam portal closing time', 'SRC023 p1 vs SRC023 p2 vs SRC052', 'SRC023 page 1 says 4.00 pm on day 3; page 2 says 11:59 PM (late window 4th day 11.59 pm); SRB 2026 says 4:00 PM.', 'SRB 2026 (4:00 PM) used; flagged requires_current_verification'),
    ('Late re-exam applications', 'SRC002/SRC003 vs SRC052/SRC023', 'Exam page/FAQ say no application after the window; SRB 2026 and the 2026 re-exam document allow a 24-hour late window with Rs. 5,000.', '2026 documents used as current'),
    ('Re-exam portal help email', 'SRC002 vs SRC003', 'SRC002 lists sapbasis@svkm.ac.in; SRC003 lists examnmims@nmims.edu.', 'Both kept as separate facts (F0211, F0220)'),
    ('GPA scale', 'SRC002/SRC003 vs SRC052', "Exam FAQ mentions a 'four-point' GPA; SRB 2026 grading is a ten-point scale.", 'Ten-point scale from SRB 2026 used'),
    ('D grade', 'SRC002/SRC023 vs SRC052', "Exam pages mention 'D grade' re-exam eligibility; MPSTME SRB 2026 grading table has no D grade.", 'Kept as stated (school-dependent); not applied to MPSTME grade table'),
    ('Attendance rule', 'SRC054 vs SRC053/SRC052', 'SRB 2024: 70-80% eligible with Dean exemption; SRB 2025/2026: below 80% requires re-admission.', '2026 rule current; 2024 rule kept as HISTORICAL fact F0420'),
    ('Student Grievance Redressal Committee', 'SRC052 vs SRC031', 'SRB 2026 lists the Registrar as chair; the 30 Sep 2026 circular names the Dean (SPPSPTM) as chair.', 'Circular treated as latest'),
    ('Hostel deposit refund signatory', 'SRC052 Annexure 8 vs SRC026', 'SRB names hostel-in-charge and a named staff member; the refund form says hostel-in-charge and DR Administration.', 'Both facts kept; no individual named in queries'),
    ('Refund annexure number', 'SRC052 p39 vs p61-62', "Section 23.4 says Application for Refund 'as per Annexure 9', but Annexure 9 is the migration form and the refund form is Annexure 8.", 'Refund form referenced without annexure number'),
    ('HOD names', 'SRC052 vs SRC046', 'IT and Mechatronics HODs differ between SRB 2026 and the MPSTME administration page.', 'Individual names not used in the dataset'),
    ('Bonafide certificate', 'SRC059/SRC054 vs SRC052', 'Portal listed bonafide certificates and railway concession up to 2024-25; omitted in 2026.', 'HISTORICAL; bonafide queries flagged requires_current_verification'),
    ('Exchange partner lists', 'SRC015 vs SRC052', 'nmims.edu lists older partners (RMIT, Grenoble...); SRB 2026 lists different MPSTME partners.', 'SRB 2026 used for MPSTME'),
    ('Duplicate pages', 'SRC008/SRC013, SRC009/SRC016, SRC002/SRC003', 'Same content served at multiple URLs.', 'Kept in inventory; facts cite one URL'),
]

# Facts whose answers change often or come from scans / conflicting sources (used for confidence flags).
TIME_SENSITIVE_SUBS = ['ACADEMIC_CALENDAR', 'ADMISSION_CANCELLATION', 'EDUCATION_VERIFICATION', 'EXAM_TIMETABLE', 'HOSTEL_APPLICATION', 'HOSTEL_FEES', 'INTERNAL_COMPLAINTS', 'LATE_FEE', 'RE_EXAMINATION', 'SC_ST_SUPPORT', 'STUDENT_GRIEVANCE']
OCR_SOURCES = ['SRC028', 'SRC031', 'SRC033', 'SRC037', 'SRC055', 'SRC056']
CONTRADICTED_FACTS = ['F0101', 'F0107', 'F0140', 'F0141', 'F0211', 'F0220', 'F0240', 'F0405']


# ----------------------------------------------------------------------------- annotations

@lru_cache(maxsize=1)
def annotations() -> tuple[dict, dict]:
    facts_doc = yaml.safe_load((ANNOTATIONS / "facts.yaml").read_text())
    queries_doc = yaml.safe_load((ANNOTATIONS / "queries.yaml").read_text())
    return facts_doc, queries_doc


def load_facts() -> dict[str, dict]:
    """Facts keyed by id, with intent and canonical department resolved."""
    facts_doc, _ = annotations()
    out = {}
    for f in facts_doc["facts"]:
        out[f["id"]] = {**f, "fact_id": f["id"], "source_id": f["source"], "intent": SUB_TO_INTENT[f["sub_intent"]],
                        "department_raw": f["department"],
                        "department": DEPARTMENTS[RAW_DEPARTMENT[f["department"]]],
                        "documents": f.get("documents", []),
                        "historical": f["fact"].startswith("[Historical")}
    return out


def next_actions() -> dict[str, str]:
    return annotations()[0]["next_action_by_sub_intent"]
