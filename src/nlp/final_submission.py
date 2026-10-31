"""Final submission package (handout section 15) built from the saved results, so every number comes from a run.

  python -m nlp.final_submission
Writes submissions/final/:
  B  Group10_Final_Project_Report.docx        the 16 sections of handout section 12 (+ LLM usage, section 20)
  E  Group10_API_Documentation.docx           endpoints, parameters, example invocation and response
  F  Group10_Dataset_Documentation.docx       the 9 items of handout section 8
  G  Group10_Evaluation_Results.docx          metrics, baseline vs improved, significance, error analysis
  H  Group10_Architecture_Diagram.png         input -> preprocessing -> NLP model -> post-processing -> API -> application
  I  Group10_Responsible_AI_Statement.docx    one page
  A  Group10_Source_Code.zip                  git archive of the repository (other groups' files excluded)
The presentation (C) and the demo guide (D) are written separately; README.md indexes everything.
"""
from __future__ import annotations

import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nlp import PROJECT_ROOT, REPORTS
from nlp.reports import REPO, SUBMIT, TEAM, Doc

OUT = SUBMIT / "final"
f3 = lambda x: f"{float(x):.3f}"
pc = lambda x: f"{float(x) * 100:.1f}%"


def load() -> dict:
    j = lambda p: json.loads((REPORTS / p).read_text(encoding="utf-8"))
    r = {w: j(f"{w}/results.json") for w in ("week2", "week3", "week4", "week5", "week6", "week7")}
    r["ex_analyze"], r["ex_advise"] = j("week6/example_analyze.json"), j("week7/example_advise.json")
    r["splits"] = {s: len(pd.read_csv(PROJECT_ROOT / "data" / "final" / f"{s}.csv")) for s in ("train", "validation", "test", "test_core")}
    return r


# --------------------------------------------------------------------------- H. architecture diagram

def architecture(path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.0))
    ax.set_xlim(0, 16), ax.set_ylim(0, 7.0), ax.axis("off")
    stages = [
        ("Input", "#e8eef7", ["Student / staff question", "(free text, Hinglish, typos)", "", "Student identity +", "several syllabi (PDF, DOCX,",
                                "text; incl. a new course)"]),
        ("Preprocessing", "#e8f3ec", ["Text cleaning, tokenisation", "PDF / DOCX text extraction", "Syllabus field parsing", "(labelled + prose)",
                                      "Identity parsing against", "24 official programmes"]),
        ("NLP models", "#fbeee6", ["Intent + sub-intent classifier", "(fine-tuned BGE-small +", "TF-IDF/embedding ensemble)",
                                   "Entity extraction (rules +", "transformer NER)", "Hybrid retrieval (BM25 +", "fine-tuned vectors, RRF)"]),
        ("Post-processing", "#f3eaf7", ["Confidence bands and", "decision rules", "Extractive answer (MMR)", "with citations",
                                        "Structured service request", "Eligibility checks, ranking"]),
        ("API / asset", "#fff6d9", ["FastAPI  /v1/analyze,", "/v1/intent, /v1/entities,", "/v1/retrieve,", "/v1/electives/advise",
                                    "Python module AdminNLP", "JSON schemas v1.0"]),
        ("Applications", "#eceff1", ["Demo web UI (/ui)", "Student portal / helpdesk", "Other AI University groups",
                                     "(Group 9 sends syllabi;", "Groups 4, 6, 12 call", "the API)"]),
    ]
    w, h, y, gap = 2.3, 3.75, 2.35, 0.38
    for i, (title, colour, lines) in enumerate(stages):
        x = 0.25 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=colour, ec="#37474f", lw=1.4))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=13.5, weight="bold", color="#1f2a44")
        for k, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.9 - k * 0.37, line, ha="center", va="center", fontsize=9.6, color="#263238")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + w + gap - 0.02, y + h / 2), arrowstyle="-|>", mutation_scale=18,
                                         color="#37474f", lw=1.6))
    ax.add_patch(FancyBboxPatch((1.2, 0.3), 13.6, 1.35, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f5f5f5", ec="#90a4ae", lw=1.2))
    ax.text(8.0, 1.27, "Official NMIMS knowledge (shared by every stage)", ha="center", fontsize=12, weight="bold", color="#1f2a44")
    ax.text(8.0, 0.72, "66 official sources  |  269 verified facts with quotes  |  727 cited passages  |  "
                       "MPSTME programme structures (24 programmes, 191 elective slots)", ha="center", fontsize=10, color="#263238")
    for xs in (4.95, 8.0, 11.05):
        ax.add_patch(FancyArrowPatch((xs, 1.67), (xs, y - 0.04), arrowstyle="-|>", mutation_scale=14, color="#90a4ae", lw=1.2, linestyle="--"))
    ax.text(8.0, 6.6, "Group 10 - University Administrative Copilot: system architecture", ha="center", fontsize=16, weight="bold", color="#1f2a44")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# --------------------------------------------------------------------------- shared result tables

def tables(r: dict) -> dict:
    w4, w5, w7 = r["week4"], r["week5"], r["week7"]
    c = {x["model"]: x for x in w4["classification"]["comparison"]}
    cls = pd.DataFrame([(name, f3(c[k]["intent_f1_test_core"]), f3(c[k]["sub_acc_test_core"]), f3(c[k]["sub_f1_test_core"]), f3(c[k]["oos_recall"]))
                        for name, k in (("Baseline: TF-IDF + logistic regression (Week 2)", "TF-IDF + LogReg (Week 2)"),
                                        ("TF-IDF + sentence-embedding ensemble (Week 3)", "TF-IDF + frozen embeddings ensemble (Week 3)"),
                                        ("SetFit (contrastive fine-tuning)", "SetFit: fine-tuned sentence transformer + LogReg"),
                                        ("Fine-tuned BGE-small transformer", "Fine-tuned transformer (BGE-small, joint sub-intent + OOS)"),
                                        ("Final: fine-tuned transformer + ensemble blend", "Fine-tuned transformer + Week 3 ensemble (blend)"))],
                       columns=["Model", "Intent macro-F1", "Routing accuracy", "Routing macro-F1", "Out-of-scope recall"])
    q = {x["system"]: x for x in w5["quantitative"]["comparison"]}
    ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
    cis = pd.DataFrame([(k, f"{x['intent_acc']:.3f} {ci(x['intent_acc_ci'])}", f"{x['sub_acc']:.3f} {ci(x['sub_acc_ci'])}", f3(x["oos_recall_clinc"]))
                        for k, x in q.items()], columns=["System", "Intent accuracy [95% CI]", "Routing accuracy [95% CI]", "OOS recall"])
    rt = {x["retriever"]: x for x in w4["retrieval"]["comparison"]}
    ret = pd.DataFrame([(name, f3(rt[k]["recall@1"]), f3(rt[k]["recall@3"]), f3(rt[k]["recall@5"]), f3(rt[k]["mrr@20"]))
                        for name, k in (("Baseline: BM25 keyword search", "BM25 keyword search"),
                                        ("Vector search, pre-trained BGE-small", "Vector, frozen BGE-small"),
                                        ("Vector search, fine-tuned BGE-small", "Vector, fine-tuned BGE-small"),
                                        ("Week 3: hybrid + intent boost", "Week 3 v2: hybrid + intent/sub-intent boost"),
                                        ("Final: hybrid (fine-tuned vectors) + boost", "Hybrid (fine-tuned vector) + boost"),
                                        ("+ cross-encoder re-ranking (rejected)", "+ cross-encoder re-rank (fine-tuned)"))],
                       columns=["Retriever", "Recall@1", "Recall@3", "Recall@5", "MRR@20"])
    nr = {x["model"]: x for x in w4["ner"]["comparison"]}
    n5 = w5["ner"]
    ner = pd.DataFrame([(name, f3(nr[k]["gold_all_strict_f1"]), f3(nr[k]["gold_all_relaxed_f1"]))
                        for name, k in (("Baseline: rule-based (dictionary + regex)", "Rule-based labeller (dictionary + regex)"),
                                        ("CRF with gazetteer (Week 3)", "CRF (Week 3)"),
                                        ("Fine-tuned transformer NER (Week 4)", "Fine-tuned transformer NER (Week 4)"))],
                       columns=["Entity extractor (hand-annotated gold set)", "Strict span F1", "Relaxed span F1"])
    ner_dep = (n5["Week 3 pipeline (v2)"]["core_relaxed_f1"], n5["Week 4 final (v3)"]["core_relaxed_f1"])
    sm = {x["summarizer"]: x for x in w4["summarization"]["comparison"]}
    summ = pd.DataFrame([(name, f3(sm[k]["rougeL"]), f3(sm[k]["embedding_similarity"]), f"{sm[k]['avg_words']:.0f}", pc(sm[k]["unsupported_number_rate"]))
                         for name, k in (("Baseline: lead sentences", "lead"), ("Final: query-focused MMR (extractive)", "mmr"),
                                         ("LLM-assisted (Qwen2.5-1.5B, local)", "llm_local"))],
                        columns=["Answer generator", "ROUGE-L", "Similarity to gold fact", "Words", "Numbers not in evidence"])
    e = w4["end_to_end"]
    e2e = pd.DataFrame([(lab, f3(e["v2"][k]), f3(e["v3"][k]), f"{e['v3'][k] - e['v2'][k]:+.3f}")
                        for lab, k in (("Intent accuracy", "intent_acc"), ("Sub-intent (routing) accuracy", "sub_intent_acc"),
                                       ("Department accuracy", "department_acc"), ("Relevant passage in top 3", "relevant_passage_top3"),
                                       ("Out-of-scope queries declined (200)", "oos_declined_200"))],
                       columns=["End to end, 291 held-out questions", "Week 3 pipeline", "Final pipeline", "Change"])
    el = w7["eligibility"]
    adv = pd.DataFrame([
        ("Eligibility decisions correct (held out)", f"{pc(el['heldout']['status_accuracy'])} of {el['heldout']['course_decisions']}"),
        ("Recommendations correct (held out)", f"{round(el['heldout']['recommendation_accuracy'] * el['heldout']['recommendation_cases'])} / {el['heldout']['recommendation_cases']}"),
        ("Eligibility decisions correct (development)", f"{pc(el['dev']['status_accuracy'])} of {el['dev']['course_decisions']}"),
        ("Syllabus fields parsed (15 syllabi, 3 formats)", pc(w7["syllabus_parsing"]["labels_removed_reordered"]["all_fields"])),
        ("Identity rewordings read correctly", f"{round(w7['paraphrase']['accuracy'] * w7['paraphrase']['variants'])} / {w7['paraphrase']['variants']}"),
        ("Advice changed by name / gender / religion / caste / region / disability", f"{w7['counterfactual']['changed']} of {w7['counterfactual']['variants']}"),
        ("Checks that cite their source", f"{pc(w7['transparency']['checks_with_citation'])} of {w7['transparency']['checks']}"),
    ], columns=["Elective advisor (Week 7)", "Result"])
    rob = pd.DataFrame(w5["robustness"]["perturbations"])
    rob = rob.pivot(index="perturbation", columns="system", values="sub_acc")
    order = ["clean", "typo x1", "typo x3", "lowercase, no punctuation", "all abbreviations", "Hinglish suffix", "drop 25% of words",
             "shuffle word order", "first half only"]
    rob = rob.loc[[o for o in order if o in rob.index]].reset_index()
    rob.columns = ["Perturbation (routing accuracy)"] + [c.split(" (")[0] for c in rob.columns[1:]]
    rob = rob[["Perturbation (routing accuracy)", "Week 2 TF-IDF", "Week 3 pipeline", "Week 4 final"]]
    rob[rob.columns[1:]] = rob[rob.columns[1:]].map(f3)
    return {"cls": cls, "cis": cis, "ret": ret, "ner": ner, "ner_dep": ner_dep, "summ": summ, "e2e": e2e, "adv": adv, "rob": rob}


ERRORS = [
    ("what is the cutoff for IIT Bombay computer science", "out of scope", "ACADEMIC_REGISTRATION",
     "Out-of-scope training data (CLINC150) is general-domain, so academic words look in-scope (9 of 12 near-domain questions accepted)."),
    ("pyq for re exam available?", "LIBRARY / previous papers", "EXAMINATIONS / RE_EXAMINATION (0.96)",
     "Student slang 'pyq' is rare in training; the keyword 're exam' dominates with high confidence."),
    ("I am not asking about revaluation, I want a photocopy of my answer sheet", "ANSWER_BOOK_COPY", "REVALUATION (0.99)",
     "The classifier relies on keywords and ignores the negation; shuffling word order barely changes predictions."),
    ("how to issue books in mpstme library", "BOOK_BORROWING", "DIGITAL_RESOURCES",
     "At NMIMS 'issue a book' means borrow it; in training 'issue' mostly means a problem."),
    ("can my parents do this for me", "TRANSCRIPTS (follow-up)", "out of scope",
     "A follow-up question with no content of its own; the component is single-turn."),
    ("result kab aayega sem 5 ka", "EXAM_RESULT", "ACADEMIC_CALENDAR",
     "Hindi words are unknown to the model; only 'result' and 'sem' carry signal."),
    ("my SAP ID is 70022300145 and I need a bonafide certificate", "70022300145 = student number", "tagged PERCENTAGE",
     "The transformer NER learned from rule labels in which 11-digit SAP IDs are rare."),
    ("problem with a faculty, how do i complain", "grievance-redressal passage", "ranked 9th",
     "Vocabulary mismatch: students say 'complain', NMIMS documents say 'grievance redressal'."),
    ("attendance published on monday, can i report a mistake on friday?", "No (3-day limit)", "LLM answer: 'Yes'",
     "The 1.5B LLM does not do the date arithmetic; the extractive answer quotes the rule correctly."),
    ("Entrepreneurship elective for a B Tech CSBS student", "not recommended (duplicate)", "eligible",
     "Duplicate courses are detected from titles only, so a differently named compulsory course is missed."),
]


# --------------------------------------------------------------------------- B. final report

def report(r: dict, T: dict, arch) -> None:
    w2, w4, w5, w6, w7 = r["week2"], r["week4"], r["week5"], r["week6"], r["week7"]
    sp = r["splits"]
    doc = Doc()
    doc.title([("University Administrative Copilot", 24, True), ("Final Project Report", 16, False),
               ("A reusable NLP asset for the AI University platform", 12, False),
               ("NLP Course Project 2026 - SVKM's NMIMS, MPSTME", 11, False), (TEAM, 11, False), (f"Repository: {REPO}", 10, False)])

    doc.h("Summary")
    doc.p("Students and staff ask administrative questions in natural language, while the answers are spread across "
          "NMIMS handbooks, circulars, web pages and offices. We built a reusable NLP asset that understands such a "
          "request, routes it to the right service and department, extracts its details, answers it from cited official "
          "sources and produces a structured service request. On top of it, an elective advisor reads several syllabi "
          "(including a new course) and a student's academic identity and recommends the elective the student can enrol in. "
          "Both are exposed as a typed HTTP API and a Python module, evaluated against baselines on held-out data, and "
          "demonstrated in a web interface.")
    q = {x["system"]: x for x in w5["quantitative"]["comparison"]}
    fin, base = q["Week 4 final (v3)"], q["Week 2 TF-IDF"]
    rt = {x["retriever"]: x for x in w4["retrieval"]["comparison"]}
    for lead, t in (("Routing: ", f"{f3(base['sub_acc'])} -> {f3(fin['sub_acc'])} accuracy over 95 services (baseline TF-IDF -> final), "
                                  f"intent accuracy {f3(fin['intent_acc'])}, out-of-scope recall {f3(fin['oos_recall_clinc'])}."),
                    ("Retrieval: ", f"Recall@3 {f3(rt['BM25 keyword search']['recall@3'])} -> {f3(rt['Hybrid (fine-tuned vector) + boost']['recall@3'])} "
                                    "(BM25 -> fine-tuned hybrid)."),
                    ("Elective advisor: ", f"{pc(w7['eligibility']['heldout']['status_accuracy'])} of held-out eligibility decisions and "
                                           f"{w7['eligibility']['heldout']['recommendation_cases']} / {w7['eligibility']['heldout']['recommendation_cases']} "
                                           "recommendations correct; advice unchanged in all 230 counterfactual tests."),
                    ("Asset: ", f"versioned JSON API, {w6['contract']['outputs_validated']} / {w6['contract']['outputs_validated']} outputs schema-valid, "
                                f"median {w6['latency'][0]['p50_ms']:.0f} ms per full analysis.")):
        doc.b(t, lead)

    doc.h("1. Problem statement")
    doc.p("NMIMS administrative procedures (certificates, transcripts, ID cards, examinations, fees, scholarships, "
          "attendance, hostel, registration, library, student services and welfare) are documented across the Student "
          "Resource Book, circulars, forms, FAQs and web pages. Students do not know which office handles a request, which "
          "documents are needed or what the rule is, so the same questions reach the help desk repeatedly. Choosing an "
          "elective is a second, harder case: eligibility depends on the programme, semester, elective slots, prerequisites "
          "and courses already taken, and a new course may not be in any catalogue yet.")

    doc.h("2. Institutional context")
    doc.b("ask a question in their own words (including typos, abbreviations and Hinglish) and get the office, "
          "required documents, next step and the official source; get elective advice with reasons.", "Students: ")
    doc.b("receive structured, pre-routed requests and see the confidence of "
          "each; low-confidence cases are referred to them instead of being answered automatically.", "Administrative staff and course coordinators: ")
    doc.b("call the API (e.g. Group 4's knowledge assistant, Group 6's early-warning engine, "
          "Group 9's curriculum engine as a syllabus source, Group 12's security layer in front of it).", "Other AI University components: ")

    doc.h("3. Existing approaches")
    doc.p("Universities typically use static FAQ pages and portals (users must know where to look), e-mail or ticket help "
          "desks (slow, manual routing), keyword search (fails on paraphrases and typos), and rule-based chatbots (brittle, "
          "small coverage). General-purpose LLM chatbots answer fluently but have no institutional knowledge and can invent "
          "rules, deadlines and fees without a source. In research, intent detection with out-of-scope rejection (e.g. the "
          "CLINC150 benchmark) and retrieval-augmented generation are the standard building blocks; our system combines them "
          "with institution-specific data, citations and human hand-off.")

    doc.h("4. NLP formulation")
    for lead, t in (("Text classification: ", "hierarchical intent detection - 12 intents and 95 sub-intents (services) plus an out-of-scope class."),
                    ("Information extraction / NER: ", "28 entity types (documents, dates, fees, student numbers, programmes, ...) filled into a service request."),
                    ("Semantic search / RAG: ", "retrieve evidence passages from 727 cited official passages; answer by query-focused extractive summarisation "
                                                 "(an LLM is supported but not the default)."),
                    ("Structured generation: ", "a service request (department, documents, next action, missing information, priority, status) and a decision "
                                                 "(auto-route, ask for information, confirm, refer to staff, out of scope)."),
                    ("Elective advice: ", "information extraction from syllabi and identity text, entity linking to the official programme catalogue, "
                                          "embedding similarity for prerequisites and duplicates, rule checks and ranking.")):
        doc.b(t, lead)

    doc.h("5. Dataset")
    doc.p(f"All data was built from official, publicly available NMIMS sources: 66 sources (website pages, the MPSTME "
          f"Student Resource Book 2026, circulars, calendars, forms; 28 PDFs, scanned ones OCR'd). From them we wrote down 269 "
          f"verified facts, each with its page and a verbatim quote that a script re-checks against the source text. "
          f"Questions were built from these facts (1,973 core questions linked to facts, including ambiguous and multi-intent "
          f"ones), and rule-based rewordings (casual, abbreviations, typos, formal, polite, Hinglish) add 3,890 variants: "
          f"5,863 labelled queries in total, not collected from students. Out-of-scope examples are real user queries from "
          f"the public CLINC150 dataset. Split by group (a question and its rewordings stay together): train {sp['train']:,}, "
          f"validation {sp['validation']:,}, test {sp['test']:,} (of which {sp['test_core']} core). Further test sets: a "
          f"hand-annotated gold NER set (120 questions, 245 spans), a 96-query challenge set of hard cases, and 23 labelled "
          f"elective cases over 15 illustrative syllabi. Full documentation: Group10_Dataset_Documentation.docx.")

    doc.h("6. Preprocessing")
    for t in ("Extraction: HTML main text (trafilatura), PDF text per page (pypdf), OCR (RapidOCR) for pages without a text layer.",
              "Knowledge base: page-level chunks for PDFs and ~220-word windows for HTML, each carrying its source URL, page and fact ids.",
              "Cleaning: whitespace and Unicode normalisation, exact and near-duplicate removal, 19 automatic quality checks (all pass).",
              f"Classifier input: lowercasing and tokenisation; the Week 2 ablation selected stop-word removal for the TF-IDF features "
              f"(word 1-2-grams + character 2-5-grams).",
              "Labels: rule-based entity spans (silver) for training; a separately hand-annotated gold set for testing.",
              "Syllabi: labelled fields are read directly; free-text proposals by prose patterns; programme names are linked to the catalogue."):
        doc.b(t)

    doc.h("7. Baseline")
    doc.p(f"Classification: TF-IDF (word + character n-grams) with logistic regression (Week 2; test macro-F1 "
          f"{w2['final']['test']['macro_f1']:.3f} on the full test set). Entities: dictionary and regular-expression rules. "
          f"Retrieval: BM25 keyword search. Answers: the first sentences of the best passage (lead).")

    doc.h("8. Proposed method")
    for lead, t in (("Router: ", "BAAI/bge-small-en-v1.5 fine-tuned end to end on 95 sub-intents + out-of-scope (square-root class weights, separate "
                                 "learning rate for the new head), blended with the Week 3 TF-IDF + sentence-embedding ensemble; blend weights and "
                                 "the out-of-scope threshold chosen on validation."),
                    ("Entities: ", "a transformer token classifier (BIO tags aligned to word pieces) merged with high-precision rules."),
                    ("Retrieval: ", "hybrid search - BM25 and a bi-encoder fine-tuned on (training question, relevant passage) pairs, fused by "
                                    "reciprocal rank fusion and boosted by the predicted intent and sub-intent. A cross-encoder re-ranker was tried "
                                    "and rejected because it lowered recall."),
                    ("Answers: ", "query-focused extractive summarisation (MMR) over the retrieved passages and verified facts, with citations; an "
                                  "LLM summariser (Qwen2.5-1.5B-Instruct, local) was compared and kept optional."),
                    ("Decisions: ", "confidence bands and rules turn the scores into an action, so uncertain requests go to staff."),
                    ("Elective advisor: ", "parses syllabi and the identity, applies seven checks (level, programme, semester, elective slot, first-year "
                                           "rule, prerequisites, duplicate of a compulsory course), each with its source, and ranks eligible courses.")):
        doc.b(t, lead)

    doc.h("9. Architecture")
    doc.fig(arch, "Figure 1. Input -> preprocessing -> NLP models -> post-processing -> API / asset -> applications", 16.5)
    doc.p("The component is a stateless FastAPI service; the web interface and other groups' applications use the same "
          "versioned /v1 endpoints, so the demo is also an integration test. Partner services (a syllabus or curriculum feed "
          "and a student-record service) plug in through documented contracts with one setting each.")

    doc.h("10. Experiments")
    doc.p("Models are trained on the training split and selected on validation; test sets are scored once. We report "
          "accuracy and macro-F1 (classification), strict and relaxed span F1 on hand-annotated gold spans (NER), Recall@k "
          "and MRR (retrieval), ROUGE, embedding similarity and the share of numbers not supported by the evidence "
          "(answers), and end-to-end routing accuracy and latency. Week 5 added 95% bootstrap confidence intervals, McNemar "
          "significance tests, calibration and selective prediction, ten controlled perturbations (typos, abbreviations, "
          "Hinglish, dropped or shuffled words, truncation) and a 96-query challenge set written after the models were frozen. "
          "The elective advisor was evaluated on 23 labelled cases (10 development, 13 held out), syllabus reformatting, "
          "identity rewordings and 230 counterfactual variants.")

    doc.h("11. Results")
    doc.table(T["cls"], "Table 1. Classification on held-out core questions (baseline vs improved)", [6.2, 2.4, 2.4, 2.4, 2.4], size=8)
    doc.table(T["cis"], "Table 2. Accuracy with 95% bootstrap confidence intervals (test_core)", [4.6, 4.2, 4.2, 2.4], size=8)
    sg = w5["quantitative"]["significance_v3_vs_v2"]
    doc.p(f"Routing beats the TF-IDF baseline significantly (McNemar p = {sg['sub_intent_vs_week2']['p_value']}); intent accuracy beats "
          f"the Week 3 pipeline significantly (p = {sg['intent']['p_value']}); the routing gain over Week 3 is small and not significant "
          f"(p = {sg['sub_intent']['p_value']}).")
    doc.table(T["ret"], f"Table 3. Retrieval of the passage that cites the answer's fact ({w4['retrieval']['n_queries']} test questions, {w4['retrieval']['n_passages']} passages)", [6.4, 2, 2, 2, 2], size=8)
    doc.table(T["ner"], "Table 4. Entity extraction on the hand-annotated gold set (120 questions, 245 spans)", [8, 3, 3], size=8)
    doc.p(f"As deployed (rules + transformer), relaxed F1 on core questions is {T['ner_dep'][1]:.3f} vs {T['ner_dep'][0]:.3f} for the "
          "Week 3 rules + CRF extractor.")
    doc.table(T["summ"], "Table 5. Answer generation (120 core test questions)", [5.6, 2, 2.6, 1.4, 3], size=8)
    doc.table(T["e2e"], "Table 6. Whole copilot, Week 3 vs final", [6.5, 3, 3, 2.2], size=8)
    doc.table(T["adv"], "Table 7. Elective advisor", [10.5, 4.5], size=8)
    ab = w5["fixes"]["abstention"]
    doc.p(f"Selective prediction: with the routing-confidence threshold of {ab['threshold']} chosen on validation, the final system answers "
          f"{pc(ab['test_coverage'])} of in-scope questions automatically at {pc(ab['test_accuracy_answered'])} routing accuracy and refers "
          f"the rest to staff, catching {pc(ab['errors_caught'])} of its routing errors.")

    doc.h("12. Error analysis")
    doc.p(f"Of {w5['errors']['n_test']} test queries the final system makes {w5['errors']['n_errors']} errors: "
          + ", ".join(f"{v} {k}" for k, v in w5["errors"]["taxonomy"].items()) +
          ". Most are near-misses between sibling services, and 22 predict a sub-intent the dataset itself lists as a plausible "
          "alternative. Robustness: formatting, abbreviations and greetings do not matter; typos, dropped words and truncation do "
          "(Table 8). Representative failures:")
    doc.table(T["rob"], "Table 8. Routing accuracy under controlled perturbations (test_core)", [5.6, 3, 3, 3], size=8)
    doc.table(pd.DataFrame(ERRORS, columns=["Input", "Expected", "Got", "Why it failed"]), "Table 9. Ten representative errors",
              [4.4, 2.8, 2.8, 6.5], size=7.5)

    doc.h("13. Reusability")
    doc.p("The asset is used without reading the source code: a versioned HTTP API (OpenAPI page at /docs, JSON Schemas in "
          "docs/schemas), a Python module and a command line. Inputs are validated (unknown fields rejected), outputs carry "
          "confidence, sources and schema_version, and errors are always {error, detail}. GET /v1/manifest describes what the "
          "component offers and consumes for a service registry. Example:")
    ex = r["ex_analyze"]
    doc.code("POST /v1/analyze   {\"text\": \"I lost my ID card, my roll no is J054. What should I do? It's urgent\"}\n" + json.dumps(
        {"intent": {"intent": ex["intent"]["intent"], "confidence": ex["intent"]["confidence"]["score"]},
         "routing": {"sub_intent": ex["routing"]["sub_intent"], "department": ex["routing"]["department"]},
         "entities": [f"{e['label']}: {e['text']}" for e in ex["entities"]],
         "decision": ex["decision"]["action"], "sources": [s["doc_id"] for s in ex["sources"]][:2]}, indent=1, ensure_ascii=False)
        .replace('[\n  "', '["').replace('",\n  "', '", "').replace('"\n ]', '"]'))
    doc.p("Integration: the advisor takes syllabi from a partner syllabus/curriculum service (e.g. Group 9) and the academic "
          "identity from a student-record service through documented contracts; Group 4 can call /v1/retrieve or /v1/analyze for "
          "administrative questions, and Group 12's security module can screen input before it reaches the component. Full "
          "details: Group10_API_Documentation.docx.")

    doc.h("14. Responsible AI")
    for lead, t in (("Privacy: ", "stateless API, nothing stored or logged; partner contracts request only programme, year, semester and courses; no student data was "
                                   "used to build the dataset."),
                    ("Bias: ", f"advice unchanged in {w7['counterfactual']['variants']} counterfactual tests (name, gender, religion, caste, region, disability); "
                               "only programme, year, semester and courses are used."),
                    ("Hallucination: ", "answers are extractive and cited by default; the LLM option wrote unsupported numbers (1.4% of numbers in Week 4), so it "
                                        "is not the default."),
                    ("Explainability: ", "every decision has a reason, a confidence and a source; every elective check names its rule (NMIMS, syllabus or our own policy)."),
                    ("Security: ", "typed input limits, escaped output in the UI; deployment behind the university gateway; Group 12's module can screen inputs."),
                    ("Human oversight: ", "low-confidence requests are referred to staff; elective advice must be confirmed by the course coordinator.")):
        doc.b(t, lead)
    doc.p("One-page statement: Group10_Responsible_AI_Statement.docx.")

    doc.h("15. Limitations")
    for t in ("Questions were built from official sources by the team, not collected from students; real traffic will be noisier.",
              "Near-domain out-of-scope questions (other universities, coursework) are often accepted; negation and follow-up questions are not handled.",
              "Heavy typos, truncated questions and Hinglish lower accuracy; rare services with few examples are routed poorly.",
              "Entity extraction misses open-ended phrases and some identifiers (e.g. SAP IDs).",
              "The elective advisor covers MPSTME programme structures for 2026; the demo syllabi are illustrative, and real NMIMS syllabus "
              "formats (e.g. Group 9's output) are only partly parsed.",
              "Small evaluation sets for some tasks (120 NER questions, 96 challenge queries, 23 elective cases) give wide confidence intervals."):
        doc.b(t)

    doc.h("16. Future work")
    for t in ("Collect and anonymise real (consented) student queries and re-evaluate; active learning on low-confidence cases.",
              "Near-domain out-of-scope data, a slang / Hinglish normaliser, and multi-turn clarification.",
              "Production: authentication behind the gateway, monitoring of confidence and drift, scheduled re-ingestion of circulars "
              "and the Student Resource Book, a faithfulness check before any LLM answer is shown.",
              "Live connections to Group 9 (syllabi), the student-record system and Group 4 (knowledge assistant)."):
        doc.b(t)

    doc.h("Appendix A. Use of LLMs (handout section 20)")
    rows = [("Qwen/Qwen2.5-1.5B-Instruct (local)", "LLM-assisted answer summarisation, compared with extractive baselines (Week 4)",
             "Evidence-only prompt: at most 2 sentences, copy fees and dates exactly", "None (runs on the laptop GPU); optional, not the default"),
            ("Claude (Anthropic API)", "Optional answer generator, switched on with COPILOT_GENERATOR=claude", "Same evidence-only prompt",
             "Needs an API key; off by default; no student data should be sent"),
            ("BGE-small, MiniLM, ms-marco cross-encoder", "Fine-tuned classifier, NER, embeddings, re-ranking experiments", "-", "Local, open-source")]
    doc.table(pd.DataFrame(rows, columns=["Model", "Purpose", "Prompt / strategy", "Dependency / cost"]), "Table 10. Models used",
              [3.8, 5, 4.2, 3.5], size=7.5)
    doc.p("Generative AI tools were also used as development assistants (coding, debugging and documentation), as the "
          "handout allows; the design, data, evaluation and results are the group's work.")
    doc.save(OUT / "Group10_Final_Project_Report.docx")


# --------------------------------------------------------------------------- G. evaluation results

def evaluation(r: dict, T: dict) -> None:
    w5 = r["week5"]
    doc = Doc()
    doc.h("Group 10 - Evaluation Results")
    doc.p(f"{TEAM}. All numbers are produced by the evaluation scripts in the repository (reports/week2-7/results.json).")
    for key, cap in (("cls", "Classification (baseline vs improved), held-out core questions"), ("cis", "Accuracy with 95% confidence intervals"),
                     ("ret", "Retrieval"), ("ner", "Entity extraction, hand-annotated gold set"), ("summ", "Answer generation"),
                     ("e2e", "Whole copilot, Week 3 vs final"), ("adv", "Elective advisor"), ("rob", "Robustness to perturbations")):
        doc.table(T[key], cap, size=8)
    sg = w5["quantitative"]["significance_v3_vs_v2"]
    doc.p(f"Significance (McNemar): routing final vs TF-IDF p = {sg['sub_intent_vs_week2']['p_value']}; intent final vs Week 3 "
          f"p = {sg['intent']['p_value']}; routing final vs Week 3 p = {sg['sub_intent']['p_value']} (not significant).")
    doc.table(pd.DataFrame(list(w5["errors"]["taxonomy"].items()), columns=["Error type (final system, full test set)", "Count"]), "Error taxonomy")
    doc.table(pd.DataFrame(ERRORS, columns=["Input", "Expected", "Got", "Why it failed"]), "Representative errors", [4.4, 2.8, 2.8, 6.5], size=7.5)
    doc.p("Detailed reports: Week 4 (baseline vs improved), Week 5 (evaluation and error analysis, 13 analysed failure cases), "
          "Week 7 (elective advisor and responsible-AI measurements) in the repository's submissions/ folder.")
    doc.save(OUT / "Group10_Evaluation_Results.docx")


# --------------------------------------------------------------------------- F. dataset documentation

def dataset(r: dict) -> None:
    sp = r["splits"]
    doc = Doc()
    doc.h("Group 10 - Dataset Documentation")
    doc.p(TEAM)
    rows = [
        ("1. Source", "66 official, publicly available NMIMS sources (nmims.edu / engineering.nmims.edu pages, MPSTME Student Resource Book 2026, "
                      "circulars, academic calendars, forms, FAQs; 28 PDFs). Out-of-scope class: CLINC150 (public research dataset of real user "
                      "queries). Elective advisor: MPSTME programme structures (Student Resource Book 2026) and 15 illustrative syllabi written by "
                      "the group in the MPSTME course-outline format."),
        ("2. Size", f"269 verified facts; 5,863 labelled queries (1,973 core + 3,890 rule-generated rewordings); 727 knowledge-base passages; "
                    f"6,800 entity spans (silver) and a gold set of 120 questions / 245 spans; 96-query challenge set; programme structures: 24 "
                    f"programmes, 1,474 course rows, 191 elective slots; 23 labelled elective cases."),
        ("3. Format", "CSV in data/final/ (queries with intent, sub-intent, department, documents, next action, entities as JSON, source URL, "
                      "fact ids, group id); YAML for facts, queries, gold NER and challenge sets (data/annotations/); knowledge base CSV with "
                      "source URL, page and fact ids."),
        ("4. Labels", "12 intents + out_of_scope, 95 sub-intents, department, required documents, next action, ambiguity and multi-intent flags, "
                      "source status (verified / unverified); 28 entity types; elective cases: expected status per syllabus and expected course."),
        ("5. Split", f"Group-aware 70/15/15 split, stratified by sub-intent; a question and its rewordings never cross splits. Train {sp['train']:,}, "
                     f"validation {sp['validation']:,}, test {sp['test']:,} (core {sp['test_core']}). Elective cases: 10 development, 13 held out."),
        ("6. Preprocessing", "HTML main-text extraction, PDF text per page, OCR for scanned pages; chunking; normalisation; de-duplication; "
                             "rule-generated rewordings (casual, abbreviation, typo, formal, polite, Hinglish); rule-based entity labels."),
        ("7. Quality issues", "Questions are written from official facts, not collected from students; entity training labels are silver (rules); "
                              "some PDFs needed OCR; 7 of 166 programme-semesters have rows missing from PDF extraction; sources can change each year "
                              "(14 contradictions between sources are documented). 19 automatic checks pass."),
        ("8. Privacy", "No student data was collected or used. Facts and passages come from public institutional documents. Example roll "
                       "numbers / SAP IDs in queries are made up. The API is stateless and stores nothing."),
        ("9. Licensing / usage", "NMIMS documents are publicly available; copyright remains with SVKM's NMIMS. They are used for non-commercial "
                                 "coursework; the raw downloaded files are not redistributed (data/sources is git-ignored), only short cited "
                                 "excerpts with source URLs. CLINC150 is published under CC BY 3.0. Models: BGE-small (MIT), MiniLM and the "
                                 "ms-marco cross-encoder (Apache 2.0), Qwen2.5-1.5B-Instruct (Apache 2.0). The illustrative syllabi are the group's own."),
    ]
    doc.table(pd.DataFrame(rows, columns=["Item (handout section 8)", "Description"]), "Dataset documentation", [3.4, 13.2], size=8)
    doc.p("Rebuild: python -m dataset.phase1_sources; python -m dataset.phase2_annotate; python -m dataset.phase3_expand; "
          "python -m dataset.phase4_validate; python -m nlp.electives.catalog. Quality report: reports/dataset/dataset_quality_report.md.")
    doc.save(OUT / "Group10_Dataset_Documentation.docx")


# --------------------------------------------------------------------------- E. API documentation

def api(r: dict) -> None:
    doc = Doc()
    doc.h("Group 10 - API / Module Documentation")
    doc.p(f"{TEAM}. Component version 1.0, schema version 1.0. Interactive documentation of every field: http://127.0.0.1:8000/docs.")
    doc.h("Run it", 2)
    doc.code("uv sync            # or: pip install -r requirements.txt && pip install -e .\nuvicorn backend.main:app --port 8000")
    ep = [("GET", "/v1/health", "-", "status, versions, models_loaded"), ("GET", "/v1/info", "-", "labels, thresholds, confidence bands, limits"),
          ("GET", "/v1/manifest", "-", "component card: what it offers and consumes"),
          ("POST", "/v1/analyze", "AnalyzeInput", "intent, routing, entities, answer, sources, service request, decision"),
          ("POST", "/v1/analyze/batch", "BatchInput (1-64)", "BatchOutput"), ("POST", "/v1/intent", "IntentInput", "intent + routing with alternatives"),
          ("POST", "/v1/entities", "TextInput", "entities with confidence"), ("POST", "/v1/retrieve", "RetrieveInput", "evidence passages with sources"),
          ("POST", "/v1/electives/advise", "AdviseInput", "eligibility per course, recommendation, citations, review step"),
          ("POST", "/v1/electives/parse", "one syllabus (text or base64 PDF/DOCX/TXT)", "ParsedSyllabus"),
          ("POST", "/v1/electives/advise-from-partners", "topic, student_id", "partner calls + advice"),
          ("GET", "/v1/electives/programs, /samples", "-", "recognised programmes; demo syllabus sets")]
    doc.table(pd.DataFrame(ep, columns=["Method", "Endpoint", "Input", "Output"]), "Endpoints", [1.4, 5, 4.2, 6], size=8)
    params = [("text", "string, 1-1000 chars", "required", "The request"), ("request_id", "string <= 128", "null", "Echoed back"),
              ("options.top_k", "int 1-13", "3", "Ranked alternatives"), ("options.k_sources", "int 1-10", "3", "Evidence passages"),
              ("options.routing_review_below", "float 0-1", "0.5", "Refer to staff below this routing confidence"),
              ("options.include_timings", "bool", "true", "Per-stage latency")]
    doc.table(pd.DataFrame(params, columns=["AnalyzeInput field", "Type", "Default", "Meaning"]), "Parameters of /v1/analyze", [4.6, 3.4, 2, 6.6])
    adv = [("student.identity", "string <= 500", "e.g. \"third year student, computer science program\""),
           ("student.program / year / semester", "optional", "override what is read from identity"),
           ("student.completed_courses", "list", "courses taken besides the compulsory ones"), ("student.interests", "optional string", "used for ranking only"),
           ("syllabi[]", "1-20 items", "{name, text} or {name, file_base64} (.pdf, .docx, .txt, .md)")]
    doc.table(pd.DataFrame(adv, columns=["AdviseInput field", "Type", "Meaning"]), "Parameters of /v1/electives/advise", [5, 3, 8.6])
    doc.h("Example invocation and response", 2)
    doc.code("curl -s localhost:8000/v1/analyze -H 'content-type: application/json' \\\n"
             "     -d '{\"text\": \"I lost my ID card, my roll no is J054. What should I do? It is urgent\"}'")
    ex = r["ex_analyze"]
    doc.code(json.dumps({"intent": {"intent": ex["intent"]["intent"], "in_scope": ex["intent"]["in_scope"], "confidence": ex["intent"]["confidence"]},
                         "routing": {k: ex["routing"][k] for k in ("sub_intent", "department")},
                         "entities": [{k: e[k] for k in ("label", "text")} for e in ex["entities"]],
                         "decision": ex["decision"], "sources": [{k: s[k] for k in ("doc_id", "title", "url")} for s in ex["sources"][:1]],
                         "model": ex["model"]}, indent=1, ensure_ascii=False))
    a = r["ex_advise"]["recommendation"]
    doc.code("curl -s localhost:8000/v1/electives/advise -H 'content-type: application/json' -d '{\n"
             "  \"student\": {\"identity\": \"third year student, computer science program\"},\n"
             "  \"syllabi\": [{\"name\": \"deep_learning.md\", \"text\": \"...\"}, {\"name\": \"NEW_genai.txt\", \"text\": \"...\"}]}'")
    doc.code(json.dumps({"recommendation": {k: a[k] for k in ("title", "semester", "slot", "confidence")}}, indent=1, ensure_ascii=False))
    doc.h("Python module", 2)
    doc.code("from nlp.component import AdminNLP\nnlp = AdminNLP()\nout = nlp.analyze(\"I lost my ID card\")\n"
             "out.intent.intent, out.routing.department, out.decision.action\n"
             "from nlp.electives.advisor import get_advisor            # elective advisor")
    doc.h("Decisions and errors", 2)
    doc.table(pd.DataFrame([("out_of_scope", "intent rejected"), ("confirm_with_student", "intent confidence medium"),
                            ("refer_to_staff", "routing confidence below routing_review_below"),
                            ("ask_for_information", "routed, but required details are missing"), ("auto_route", "otherwise")],
                           columns=["decision.action", "When"]), "Decision rules (first match)", [5, 11.6])
    doc.p("Errors: HTTP 422 validation_error (blank or too long text, unknown field, option out of range); HTTP 503 "
          "models_unavailable (models not trained). Body: {\"error\": ..., \"detail\": ...}. Schemas: docs/schemas/*.json. Full "
          "guides: docs/nlp_component.md, docs/elective_advisor.md, docs/integration.md.")
    doc.save(OUT / "Group10_API_Documentation.docx")


# --------------------------------------------------------------------------- I. responsible-AI statement (one page)

def responsible(r: dict) -> None:
    w4, w7 = r["week4"], r["week7"]
    llm = {x["summarizer"]: x for x in w4["summarization"]["comparison"]}["llm_local"]
    doc = Doc()
    doc.h("Group 10 - Responsible AI Statement")
    doc.p(f"University Administrative Copilot. {TEAM}")
    for lead, t in (
            ("Privacy. ", "No student data was used to build the system: the data comes from public NMIMS documents and questions built from them. "
                          "The API is stateless and stores or logs nothing. Partner contracts ask only for programme, year, semester and completed "
                          "courses - no names, contact details or grades. The optional external LLM is off by default, and no student data should be "
                          "sent to public AI services."),
            ("Bias and fairness. ", f"Only programme, year, semester, courses and stated interests affect advice. Adding a name, gender, religion, caste, "
                                    f"region or disability to the student's text changed the advice in {w7['counterfactual']['changed']} of "
                                    f"{w7['counterfactual']['variants']} counterfactual tests. Remaining risks: lower accuracy for Hinglish and heavily "
                                    "misspelt questions, and coverage limited to MPSTME programmes."),
            ("Hallucination. ", "Answers are extractive by default: every sentence is copied from an official passage or verified fact, and the "
                                f"sources are returned. The LLM option was measured and wrote numbers not present in its evidence "
                                f"({llm['unsupported_number_rate'] * 100:.1f}% of numbers), so it is not the default. Elective rules come from the "
                                "official programme structures, each with its page."),
            ("Explainability. ", f"Every output has a confidence and a reason; {w7['transparency']['checks_with_citation'] * 100:.0f}% of "
                                 f"{w7['transparency']['checks']} elective checks cite their source, and each check says whether it is an NMIMS rule, "
                                 "a syllabus statement or our own advisor policy."),
            ("Security. ", "Inputs are typed and size-limited, unknown fields are rejected, and all text is escaped before display. There is no "
                           "authentication, so the API must run behind the university gateway; a prompt-injection filter such as Group 12's module "
                           "should screen inputs before any LLM is enabled."),
            ("Human oversight. ", "The system advises; it does not decide. Low-confidence requests are referred to staff instead of being answered, "
                                  "and every elective recommendation tells the student to confirm with the course coordinator, who also checks seats "
                                  "and timetable. Unknown programmes or missing information give 'needs review', never a confident 'no'."),
            ("Limitations. ", "Questions were built from official sources rather than collected from students; near-domain off-topic questions, "
                              "negation and follow-up questions are handled poorly; NMIMS rules change every year, so the knowledge base and "
                              "programme structures must be re-ingested each year.")):
        doc.p(t, lead)
    doc.save(OUT / "Group10_Responsible_AI_Statement.docx")


# --------------------------------------------------------------------------- A. source code

def source_code() -> None:
    excluded = [":(exclude)Group9_*", ":(exclude)submissions", ":(exclude)NLP_Project_2026_Assignment.pdf"]
    subprocess.run(["git", "archive", "--format=zip", "--prefix=university_administrative_copilot/", "-o",
                    str(OUT / "Group10_Source_Code.zip"), "HEAD", "--", ".", *excluded], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = load()
    T = tables(r)
    arch = OUT / "Group10_Architecture_Diagram.png"
    architecture(arch)
    report(r, T, arch)
    evaluation(r, T)
    dataset(r)
    api(r)
    responsible(r)
    source_code()
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size / 1024:8.0f} KB  {p.name}")


if __name__ == "__main__":
    main()
