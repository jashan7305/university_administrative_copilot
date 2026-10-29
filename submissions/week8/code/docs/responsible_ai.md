# Responsible-AI analysis: elective advisor and administrative copilot

This analysis covers the Week 7 elective advisor and the copilot NLP component it is served with. Each risk below
comes with what we did about it and, where possible, a measurement. The numbers are from
`python -m nlp.pipeline.evaluate_week7` (`reports/week7/results.json`).

## 1. What the system decides, and what it does not

The advisor tells a student which of the given electives they can enrol in and which fits best. It does **not**
enrol anyone. NMIMS routes elective choices through the course coordinator or programme chair (fact F0009), and an
elective runs only if at least 15 students or 40% of the class register (SRB 2026 p. 13). Every output says this,
and lists what the coordinator should confirm (`human_review`).

**Who is affected:**
- **Students:** a wrong "not eligible" could stop a student from taking a course they want. A wrong "eligible" could
  lead to a rejected registration.
- **Coordinators:** they see the advice the student brings.
- **Faculty:** they see which courses are recommended.

## 2. Risks, mitigations and evidence

| Principle | Risk | What we did | Evidence |
|---|---|---|---|
| **Fairness** | Advice depends on who the student is rather than what they studied | Only programme, year, semester, completed courses and optional interests are used; nothing else is extracted from the text | Counterfactual test: 23 cases x 10 cues (names suggesting different religions and genders, "I am a woman", "I belong to the SC category", "I come from a small village in Bihar", "I have a disability") = 230 variants, **0 changed** the advice |
| **Transparency** | A student cannot see why a course was excluded | Every check has a plain-language detail, a source, and a rule type: NMIMS rule, syllabus, or advisor policy | **100%** of 488 checks cite a source; **100%** of exclusions give a reason; every recommendation lists why |
| **Accountability** | Our own judgement presented as a university rule | The one policy that is ours (do not recommend a course that repeats a compulsory course) is labelled `advisor_policy` | 16 of 488 checks are advisor policy; all labelled |
| **Human oversight** | Advice taken as an enrolment | Disclaimer; coordinator step; new courses flagged as needing approval | **100%** of outputs include a human-review step |
| **Reliability** | Different wording of the same thing gives different advice | Programme and course names recognised against the official catalogue; abbreviations; prose fallback | Identity rewordings ("3rd yr CS student", "comps engineering sem 6", "Computer Engg., Sem VI"): **20 / 20** read correctly. Reformatted syllabi (lowercased; labels removed and moved): **46 / 46** give the same advice |
| **Knowing its limits** | Confident advice on input it does not understand | An unknown programme is not guessed; missing year or semester leads to `needs_review`, not to a confident "no" | All 3 probes (unknown programme, no year, empty) give **no recommendation**; courses that depend on the missing information are `needs_review` |
| **Accuracy** | Wrong eligibility | Rules taken from the official structures; evaluated on held-out cases labelled from the rules | Held out: 98.5% of 65 course decisions, 12 / 12 recommendations (section 4) |
| **Privacy** | Student data stored, logged or over-collected | The API is stateless, so nothing is stored or logged by the application. The partner student-record contract requests only programme, year, semester and completed courses | By design; contract in [integration.md](integration.md) |
| **Data coverage** | Students outside covered programmes get wrong advice | Catalogue limited to MPSTME (Mumbai) 2026; unknown programmes are reported, not guessed | 24 programmes, 166 programme-semesters; 7 have rows the PDF extraction missed (listed in the results) |
| **Honesty about data** | Illustrative syllabi taken as official NMIMS documents | Labelled illustrative in the UI, API (`/v1/electives/samples`), docs and data README | |

## 3. Other risks considered

- **Steering.** Ranking by interests could quietly narrow students' choices. The advisor shows every eligible course,
  not only the top one. Interests only order eligible courses and never change eligibility.
- **Automation bias.** A confident recommendation might not be questioned. The confidence drops when the runner-up
  is close. The "did you mean" programme alternatives are shown. The coordinator step is always present.
- **Out-of-date rules.** Programme structures change every year. The catalogue is rebuilt from the current Student
  Resource Book with one command (`python -m nlp.electives.catalog`), and every rule shows its page.
- **Backlogs (ATKT).** The advisor assumes compulsory courses of earlier semesters were passed. A student with a
  backlog could be told they meet a prerequisite they do not. The output lists the courses it assumed completed, and
  students can enter what they actually completed.
- **Language.** Identity text is English with common Indian-English abbreviations. Other languages are not
  supported, and that input leads to `needs_review`.
- **Copilot (Week 6 component).** Its confidence scores are not calibrated (Week 5 ECE 0.10 for intent). Low
  confidence leads to "confirm with student" or "refer to staff" instead of an automatic answer. Answers are
  grounded in, and cite, official NMIMS sources.

## 4. How we measured

- **Cases.** There are 23 labelled cases: identity, five syllabi, the expected status of each, and the expected
  recommendation. Ten (dev) were used while building. Thirteen (held out) were labelled from the published rules
  before the advisor was run on them.
- **Before and after fixes.** The first run on the held-out cases found four faults, all fixed: ordinal semesters,
  the upper-case "IT", "Prerequisites: None" in prose, and missing information treated as failure. The before-fixes
  numbers are kept in `reports/week7/results_before_fixes.json`: held-out status accuracy 95.4%, recommendation 11/12,
  identity rewordings 70%. The final held-out numbers are therefore no longer fully unseen.
- **Remaining error.** "Entrepreneurship and New Venture Creation" is not recognised as overlapping the compulsory
  "Managing Innovation and Entrepreneurship" (B Tech CSBS). Duplicate detection compares course titles only.
- **Limits of the measurement.** The syllabi and cases were written by the team, so the evaluation measures whether
  the advisor applies the rules correctly. It does not measure how real students fare. A pilot with a course
  coordinator, comparing the advisor's answer with the coordinator's on real requests, is the next step.

## 5. Commitments for deployment

1. Show the disclaimer and the coordinator step with every result. This is already done in the API and the UI.
2. Rebuild the catalogue each academic year and check the extraction gaps.
3. Use only official syllabi in production. The demo syllabi are illustrative.
4. Keep the API stateless. If logging is ever added, log only aggregate counts, never identity text.
5. Re-run `evaluate_week7` (including the counterfactual test) on every change to the rules or parsers.
