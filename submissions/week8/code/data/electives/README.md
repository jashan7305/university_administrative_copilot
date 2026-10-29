# Elective advisor data

- `syllabi/<topic>/` - **illustrative course outlines written by Group 10** for the Week 7 demo and evaluation,
  in the MPSTME course-outline format. They are *not* official NMIMS syllabi. Files named `NEW_*` are new-course
  proposals in a free prose format, to show that an unseen syllabus is handled.
- `eval_cases.yaml` - hand-labelled evaluation cases (student identity + syllabus set -> expected eligibility and
  recommendation), labelled from the published rules before the advisor was run on them.

The official data the advisor relies on is `data/final/program_structures.csv` (programme structures and elective
slots, parsed from the MPSTME Student Resource Book 2026, SRC052) and the verified facts in
`data/annotations/facts.yaml`.
