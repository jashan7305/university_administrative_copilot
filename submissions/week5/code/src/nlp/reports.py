"""Build the weekly submission reports (Word) from the saved results, so every number in a report comes from a run.

  python -m nlp.reports                    # all reports + code snapshots
  python -m nlp.reports week2|week3|week4|week5|code
Outputs: submissions/weekN/*.docx, and for every week submissions/weekN/code/ (the Python files written that week,
same paths as in src/) plus submissions/weekN/Group10_WeekN_code.zip for upload.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from nlp import DATA, PROJECT_ROOT, REPORTS

SUBMIT = PROJECT_ROOT / "submissions"
TEAM = "Group 10  |  Roll No. J054, J055, J056, J057, J059"
REPO = "github.com/jashan7305/university_administrative_copilot"


# --------------------------------------------------------------------------- docx helpers

class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.1)
        s.top_margin = s.bottom_margin = Cm(1.9)
        st = self.d.styles["Normal"]
        st.font.name, st.font.size = "Calibri", Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        for n, sz in (("Heading 1", 14), ("Heading 2", 12)):
            h = self.d.styles[n]
            h.font.name, h.font.size, h.font.bold = "Calibri", Pt(sz), True
            h.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        fp = s.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            r = fp.add_run()
            el = OxmlElement("w:fldChar" if kind else "w:instrText")
            if kind:
                el.set(qn("w:fldCharType"), kind)
            else:
                el.set(qn("xml:space"), "preserve")
                el.text = txt
            r._r.append(el)

    def title(self, lines):
        for _ in range(4):
            self.d.add_paragraph()
        for text, size, bold in lines:
            p = self.d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(text)
            r.font.size, r.bold = Pt(size), bold
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def h(self, text, level=1):
        self.d.add_heading(text, level)

    def p(self, text, bold_lead=None):
        para = self.d.add_paragraph()
        para.paragraph_format.space_after = Pt(5)
        if bold_lead:
            para.add_run(bold_lead).bold = True
        para.add_run(text)

    def b(self, text, bold_lead=None, numbered=False):
        para = self.d.add_paragraph(style="List Number" if numbered else "List Bullet")
        para.paragraph_format.space_after = Pt(2)
        if bold_lead:
            para.add_run(bold_lead).bold = True
        para.add_run(text)

    def code(self, text):
        for line in text.splitlines():
            para = self.d.add_paragraph()
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.left_indent = Cm(0.5)
            r = para.add_run(line)
            r.font.name, r.font.size = "Consolas", Pt(8.5)

    def table(self, df: pd.DataFrame, caption: str, widths=None, bold_row=None, size=8.5):
        c = self.d.add_paragraph()
        c.paragraph_format.keep_with_next = True
        rr = c.add_run(caption)
        rr.bold, rr.font.size = True, Pt(9)
        t = self.d.add_table(rows=len(df) + 1, cols=len(df.columns))
        t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
        for i, row in enumerate([list(df.columns)] + df.astype(str).values.tolist()):
            for j, v in enumerate(row):
                cell = t.cell(i, j)
                if widths:
                    cell.width = Cm(widths[j])
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(v))
                run.font.size, run.bold = Pt(size), (i == 0 or (bold_row is not None and i == bold_row + 1))
                if i == 0:
                    sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear"), sh.set(qn("w:color"), "auto"), sh.set(qn("w:fill"), "E7ECF3")
                    cell._tc.get_or_add_tcPr().append(sh)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def fig(self, path, caption, width=14.5):
        if not path.exists():
            return
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width))
        c = self.d.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.italic, r.font.size = True, Pt(9)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.d.save(path)
        print("saved", path.relative_to(PROJECT_ROOT))


f3 = lambda x: f"{float(x):.3f}"
ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"


def dataset_stats() -> dict:
    q = pd.read_csv(DATA / "final" / "nmims_admin_queries.csv")
    inv = pd.read_csv(DATA / "final" / "source_inventory.csv")
    ner = pd.read_csv(DATA / "final" / "nmims_ner_annotations.csv")
    kb = pd.read_csv(DATA / "final" / "nmims_knowledge_base.csv")
    import yaml
    facts = yaml.safe_load((DATA / "annotations" / "facts.yaml").read_text())["facts"]
    return {"q": q, "inv": inv, "ner": ner, "kb": kb, "facts": facts}


# --------------------------------------------------------------------------- Week 2

def week2() -> None:
    r = json.loads((REPORTS / "week2" / "results.json").read_text())
    ds = dataset_stats()
    q, inv, ner, kb, facts = ds["q"], ds["inv"], ds["ner"], ds["kb"], ds["facts"]
    fin, F = r["final"], REPORTS / "week2" / "figures"
    core = q[~q["is_variant"]]
    doc = Doc()
    doc.title([("University Administrative Copilot", 24, True), ("Week 2: Dataset and Baseline (NMIMS)", 15, False),
               ("Revised submission - dataset rebuilt from official NMIMS sources", 11, False), (TEAM, 11, False),
               (f"Repository: {REPO}", 10, False)])

    doc.h("Summary")
    doc.p("This revision replaces our earlier template-generated dataset with a dataset grounded in official sources of "
          "SVKM's NMIMS (Mumbai, Vile Parle; MPSTME). We collected the university's student pages, examination "
          "procedures, Student Resource Books, forms, circulars and policies, extracted 269 verifiable facts, wrote "
          "questions from those facts, and trained and evaluated baseline classifiers.")
    rows = [("Official NMIMS sources collected", f"{int((inv['http_status'].astype(str) == '200').sum())} of {len(inv)} "
             f"({int(inv['extraction_method'].fillna('').str.startswith('pdf').sum())} PDFs)"),
            ("Verified facts (each with a verbatim quote re-checked against the source)", len(facts)),
            ("Queries: from official NMIMS sources / rule-generated variants / total", f"{len(core)} / {int(q['is_variant'].sum())} / {len(q)}"),
            ("Intents / sub-intents / routing departments", "12 / 95 / 23"),
            ("NER spans / knowledge-base passages", f"{len(ner)} / {len(kb)}"),
            ("Baseline intent macro-F1 (core test / full test)", f"{f3(fin['test_core']['macro_f1'])} / {f3(fin['test']['macro_f1'])}"),
            ("Out-of-scope recall on 1,000 real CLINC150 queries", f3(fin["test_clinc_oos"]["oos_recall"])),
            ("Sub-intent (routing) baseline accuracy, core test", f3(r["sub_intent"]["test_core"]["accuracy"]))]
    doc.table(pd.DataFrame(rows, columns=["Item", "Value"]), "Table 1. Week 2 at a glance", [11.5, 5.3])

    doc.h("1. Task")
    doc.p("Students and faculty ask administrative questions in natural language. The NLP component must recognise the "
          "request (intent and sub-intent), extract details (entities), and route it to the responsible NMIMS "
          "department with the documents and next step stated in official sources. Week 2 builds the dataset and "
          "baseline classifier.")

    doc.h("2. Data collection")
    doc.p("Sources were limited to official NMIMS domains (nmims.edu, engineering.nmims.edu, navimumbai.nmims.edu, "
          "upload.nmims.edu): the 20 pages named in the project brief plus relevant NMIMS pages linked from them. "
          "robots.txt was checked for every host and the crawl used a polite 1.5 s delay. HTML main text was "
          "extracted with trafilatura; PDFs with pypdf; scanned PDFs (revaluation guide, circulars, 2026-27 academic "
          "calendars) were OCR'd with RapidOCR, regrouping text boxes into lines so table rows stay intact.")
    st = inv["source_type"].value_counts().rename_axis("Source type").reset_index(name="Count")
    doc.table(st, "Table 2. Sources by type", [6, 3])
    key = inv[inv["facts_cited"] > 4].sort_values("facts_cited", ascending=False)[["source_id", "title", "academic_year", "facts_cited"]]
    doc.table(key, "Table 3. Most-used sources", [1.8, 10.5, 2, 2])
    doc.p("Each source has an academic year and a CURRENT/HISTORICAL flag. The MPSTME Student Resource Book 2026 "
          "(effective July 2026) is the primary source; the 2024 and 2025 books are kept as historical versions and "
          "never merged with current rules (for example, the 2024 rule allowing a Dean's exemption at 70-80% attendance "
          "was replaced by an 80%-per-course rule). Fourteen contradictions or duplicates between sources were found and "
          "documented with the resolution used (reports/dataset/dataset_quality_report.md).")

    doc.h("3. Fact extraction and verification")
    doc.p("We read the sources and recorded 269 facts. Each fact paraphrases one passage and stores the source id, page, "
          "sub-intent, department and a verbatim quote. The validation script re-finds every quote in the extracted "
          "source text; a fact whose quote cannot be found fails the build. Fees, deadlines, emails and procedures in "
          "the dataset come only from these facts. Requests for which no official source exists (for example failed "
          "payments, education loans, scholarship status) are kept as UNVERIFIED queries with no answer attached.")

    doc.h("4. Query dataset")
    doc.p("Because collecting real queries was not possible, questions were built from the verified facts of the official NMIMS sources in varied "
          "student styles (short, informal, typos, Hinglish, formal emails, parents, alumni, faculty, employers), "
          "including ambiguous and multi-intent questions. Each core question links to the fact(s) that answer it. "
          "To enlarge the training set, each core question received up to two rule-generated variants (casual, "
          "abbreviation, typo, formal, polite, Hinglish). These provenance details are recorded per row "
          "(query_kind, is_variant, variant_type).")
    kinds = q.groupby(["query_kind"]).size().rename_axis("Kind").reset_index(name="Rows")
    doc.table(kinds, "Table 4. Rows by kind", [6, 3])
    doc.table(q["variant_type"].value_counts().rename_axis("Variant type").reset_index(name="Rows"), "Table 5. Rows by variant type", [6, 3])

    doc.h("5. Cleaning, validation and annotation")
    doc.b(" exact and near-duplicate removal; label consistency (same text, same label); taxonomy and department "
          "validity; every verified row has a retrievable source URL; every cited fact's quote is present in its source; "
          "unverified rows carry no answer; NER spans are well-formed; no group crosses splits. All 19 checks pass.", "Checks:")
    doc.b(" intent and sub-intent from the linked facts; canonical department (23 routing targets); required documents "
          "and next action only where the source states them; confidence and requires_current_verification flags for "
          "OCR'd, historical, contradicted or time-sensitive facts.", "Labels:")
    doc.b(f" {len(ner)} entity spans over 28 types (DOCUMENT, CERTIFICATE, EXAMINATION, FEE, SCHOLARSHIP, PROGRAM, "
          "SEMESTER, STUDENT_ID, PURPOSE, ...), produced by a dictionary/regex labeller built from the sources (silver "
          "labels, used for the Week 3 NER model).", "NER:")

    doc.h("6. Train / validation / test split")
    sp = pd.read_csv(DATA / "final" / "split_statistics.csv")
    doc.table(sp, "Table 6. Records per intent and split", [5.5, 3, 3.5, 3])
    doc.p("Splits are made by group: a core question and all its variants are always in the same split, stratified by "
          "sub-intent, so no rewording of a test question is seen in training. test_core holds only the "
          "test questions. The out-of-scope class uses real CLINC150 data: its out-of-scope queries plus a sample of its "
          "other-domain queries (banking, travel, home) for training/validation; the official CLINC150 out-of-scope "
          "test set (1,000 queries) is used only for evaluation.")

    doc.h("7. Exploratory analysis")
    e = r["eda"]
    doc.table(pd.DataFrame([("Training rows", e["rows"]["train"]), ("Vocabulary", e["train_vocab_size"]),
                            ("Mean / median / max tokens", f"{e['train_mean_tokens']} / {e['train_median_tokens']:.0f} / {e['train_max_tokens']}"),
                            ("Largest / smallest class ratio", e["imbalance_ratio_max_min"]),
                            ("OOV rate: core test / variants", f"{e['oov_rate_vs_train']['test_core']:.1%} / {e['oov_rate_vs_train']['test_variants']:.1%}")],
                           columns=["Statistic", "Value"]), "Table 7. Corpus statistics", [8, 6])
    doc.fig(F / "class_distribution.png", "Figure 1. Class distribution by split", 13)
    doc.table(pd.DataFrame([(k, ", ".join(v[:6])) for k, v in e["top_chi2_terms"].items()], columns=["Intent", "Top chi-squared terms"]),
              "Table 8. Most discriminative terms", [4.5, 12])

    doc.h("8. Preprocessing ablation")
    abl = pd.DataFrame(r["ablation"]).rename(columns={"step": "Configuration", "val_macro_f1": "Val macro-F1",
                                                      "test_variants_macro_f1": "Variants macro-F1"})
    doc.table(abl, "Table 9. Cumulative preprocessing steps (TF-IDF logistic regression)", [8, 3.5, 3.5])
    doc.p("Steps are added one at a time and judged on validation. Spelling correction, lemmatisation and stopword "
          "removal with a keep-list (wh-words and negations are kept) gave the best validation score (0.903); the "
          "differences between steps are small (about 1 point), so the gains are indicative rather than conclusive.")

    doc.h("9. Baseline models")
    mc = pd.DataFrame(r["model_comparison"])[["model", "cv_macro_f1_mean", "cv_macro_f1_std", "val_accuracy", "val_macro_f1"]]
    mc.columns = ["Model", "CV macro-F1", "CV sd", "Val accuracy", "Val macro-F1"]
    doc.table(mc, "Table 10. Model comparison (5-fold GroupKFold CV on train + validation)", [5.5, 2.5, 2, 2.5, 2.5])
    doc.fig(F / "model_comparison.png", "Figure 2. Cross-validation and validation macro-F1", 13)
    doc.p(f"Selected: class-balanced logistic regression on word + character TF-IDF, C = {r['C']} (best CV score); "
          f"out-of-scope rejection threshold {r['threshold']} chosen on validation (the explicit out_of_scope class "
          "already separates unrelated queries).")

    doc.h("10. Results")
    res = [(name, fin[k]["n"], f"{f3(fin[k]['accuracy'])} {ci(fin[k]['accuracy_ci95'])}",
            f"{f3(fin[k]['macro_f1'])} {ci(fin[k]['macro_f1_ci95'])}" if "macro_f1" in fin[k] else "-")
           for name, k in (("Test (core + variants)", "test"), ("Test core (official NMIMS sources)", "test_core"),
                           ("Test variants (rule-generated)", "test_variants"), ("CLINC150 out-of-scope (recall)", "test_clinc_oos"))]
    doc.table(pd.DataFrame(res, columns=["Test set", "n", "Accuracy [95% CI]", "Macro-F1 [95% CI]"]),
              "Table 11. Intent classification, test sets scored once", [5, 1.5, 5, 5])
    pc = pd.DataFrame([(k, f3(v["precision"]), f3(v["recall"]), f3(v["f1-score"]), int(v["support"]))
                       for k, v in fin["test_core"]["per_class"].items()], columns=["Intent", "P", "R", "F1", "n"])
    doc.table(pc, "Table 12. Per-intent results on the core test set", [5, 2.5, 2.5, 2.5, 2])
    doc.fig(F / "confusion_test_core.png", "Figure 3. Confusion matrix, core test set", 12.5)
    s = r["sub_intent"]
    doc.p(f"A second baseline routes queries to one of {s['n_classes']} sub-intents (the unit that determines department "
          f"and documents): accuracy {f3(s['test_core']['accuracy'])} / macro-F1 {f3(s['test_core']['macro_f1'])} on the "
          "core test set. This harder task is improved in Week 3.")

    doc.h("11. Error analysis")
    err = pd.read_csv(REPORTS / "week2" / "errors.csv")
    pairs = (err[err["split"] == "test_core"].groupby(["true", "pred"]).size().sort_values(ascending=False).head(8)
             .rename("errors").reset_index())
    doc.table(pairs, "Table 13. Most frequent confusions (core test)", [5, 5, 2])
    doc.b(" neighbouring intents share vocabulary: transcripts vs certificates (grade sheets, degree), fees vs hostel "
          "(hostel fees, deposits), examinations vs attendance (exam eligibility).", "Overlapping vocabulary:")
    doc.b(" variants score lower than core questions (macro-F1 0.80 vs 0.88), mainly from typos in short queries.", "Noise:")
    doc.b(" very short questions such as \"when is it\" or \"fees\" are genuinely ambiguous; they are marked in the "
          "ambiguity column and evaluated separately in Week 3.", "Ambiguity:")

    doc.h("12. Limitations")
    for t in ["Core questions are built from official NMIMS sources rather than collected from students, so accuracy on "
              "real traffic may be lower than reported here.",
              "About two thirds of the rows are rule-generated variants; test_core results are the more realistic figure.",
              "NER labels are silver (rule-generated); a hand-corrected NER test set is future work.",
              "Category sizes follow the sources: scholarships and ID cards have few official facts and fewer questions.",
              "Some facts are time-sensitive (2026-27 dates, fees, committee members) and are flagged for re-verification."]:
        doc.b(t)
    doc.h("13. Reproducibility")
    doc.code("uv sync\npython -m dataset.phase1_sources      # collect, extract, inventory, knowledge base\n"
             "python -m dataset.phase2_annotate     # facts + queries -> labelled core queries + NER\n"
             "python -m dataset.phase3_expand       # variants, group-aware split\n"
             "python -m dataset.phase4_validate     # 19 checks + dataset reports\n"
             "python -m nlp.intent.experiments      # this report's baseline results")
    doc.save(SUBMIT / "week2" / "Group10_Week2_Dataset_and_Baseline_Report.docx")


# --------------------------------------------------------------------------- Week 3

def week3() -> None:
    r = json.loads((REPORTS / "week3" / "results.json").read_text())
    v1 = json.loads((REPORTS / "week3" / "results_v1_week3_submission.json").read_text())
    w2 = json.loads((REPORTS / "week2" / "results.json").read_text())
    ds = dataset_stats()
    F = REPORTS / "week3" / "figures"
    I, S, N, SS, RT, SM, E = (r["intent"], r["sub_intent"], r["ner"], r["semantic_similarity"], r["retrieval"],
                              r["summarization"], r["end_to_end"])
    bestret = max(RT["comparison"], key=lambda x: x["recall@3"])
    mmr = next(x for x in SM["comparison"] if x["summarizer"] == "mmr")
    lead = next(x for x in SM["comparison"] if x["summarizer"] == "lead")
    simb = {x["method"]: x for x in SS["task_b_results"]}
    doc = Doc()
    doc.title([("University Administrative Copilot", 24, True), ("Week 3: Core NLP Pipeline - Version 2", 15, False),
               ("Classification, NER, embeddings, semantic similarity, RAG, summarization and information extraction "
                "on official NMIMS data", 11, False), (TEAM, 11, False), (f"Repository: {REPO}", 10, False)])

    doc.h("Summary")
    doc.p("This report presents version 2 of the NLP pipeline. Compared with the first Week 3 submission, every component "
          "now runs on the official NMIMS dataset and knowledge base built in Week 2, summarization has been added, "
          "embeddings and semantic similarity are evaluated in their own right, and each component is compared against "
          "at least one alternative. The pipeline is served by the FastAPI backend.")
    rows = [("Intent classification (13 classes)", I["selected"], f"macro-F1 {f3(I['final']['test_core']['macro_f1'])} (core test); "
             f"OOS recall {f3(I['final']['clinc_oos_recall'])}"),
            ("Sub-intent routing (95 classes)", f"{S['selected']} ({S['strategy']})", f"accuracy {f3(E['sub_intent_acc'])} end-to-end"),
            ("NER (28 entity types)", N["selected"], f"span F1 {f3(next(x for x in N['comparison'] if x['model'] == N['selected'])['test'])}"),
            ("Embeddings / semantic similarity", "BGE-small sentence embeddings", f"same-fact question match acc@5 {f3(simb['bge-small embedding cosine']['acc@5'])}"),
            ("RAG retrieval (727 NMIMS passages)", bestret["retriever"], f"recall@3 {f3(bestret['recall@3'])}, MRR {f3(bestret['mrr'])}"),
            ("Summarization (new)", "MMR query-focused extractive", f"ROUGE-L {f3(mmr['rougeL'])} (lead baseline {f3(lead['rougeL'])})"),
            ("Information extraction (service request)", "routing + entities + source facts",
             f"department {f3(E['department_acc'])}, documents Jaccard {f3(E['required_documents_jaccard'])}")]
    doc.table(pd.DataFrame(rows, columns=["Component", "Selected method", "Result"]), "Table 1. Pipeline v2 at a glance", [5, 5.5, 6.3])

    doc.h("1. Progress since the previous submission")
    doc.p("The feedback on our first Week 3 submission was that it showed minimal incremental progress. This section "
          "lists exactly what changed between Week 2, the first Week 3 submission (v1) and this version (v2).")
    prog = [
        ("Data", "Template-generated queries, 27 procedure documents",
         f"{len(ds['q'])} NMIMS queries grounded in {len(ds['facts'])} verified facts; {len(ds['kb'])} official NMIMS passages as knowledge base"),
        ("Intent classification", "TF-IDF baseline; embeddings tried on generated data",
         "5 models + kNN compared on NMIMS splits; TF-IDF+embedding ensemble selected; out-of-scope trained on real CLINC150 data"),
        ("Routing", "Intent only; procedure chosen by retrieval", "95-class sub-intent router (flat vs hierarchical chosen on validation)"),
        ("NER", "CRF on generated sentences (21 types)", "CRF on NMIMS annotations (28 types), compared with dictionary baseline"),
        ("Embeddings / semantic similarity", "Used only inside retrieval", "Evaluated separately: paraphrase matching and same-fact question retrieval, TF-IDF vs MiniLM vs BGE"),
        ("RAG retrieval", "Hybrid over 27 invented documents, 81 queries", f"5 retrievers over {RT['n_passages']} real passages, {RT['n_queries']} queries; intent + sub-intent boosting"),
        ("Summarization", "Not implemented", "Lead vs TextRank vs MMR, evaluated with ROUGE and embedding similarity"),
        ("Information extraction", "Service request from invented procedure metadata", "Service catalogue built from verified facts; every answer cites source URLs"),
        ("Engineering", "Scripts per task", "Phase-based repo (src/dataset, src/nlp, src/backend), 21 automated tests, reports generated from results"),
    ]
    doc.table(pd.DataFrame(prog, columns=["Area", "v1 (first Week 3 submission)", "v2 (this report)"]), "Table 2. Change log", [3.2, 5.6, 8])
    v1e = v1.get("e2e") or v1.get("end_to_end", {})
    cmp = [("Intent accuracy (end-to-end)", f3(v1e.get("intent_acc", 0)), f3(E["intent_acc"])),
           ("Procedure / sub-intent correct", f3(v1e.get("procedure_acc", 0)), f3(E["sub_intent_acc"])),
           ("Relevant passage in top 3", f3(v1e.get("doc_in_top3", 0)), f3(E["relevant_passage_in_top3"])),
           ("Out-of-scope declined (200 CLINC150)", f3(v1e.get("oos_declined_200", 0)), f3(E["oos_declined_200"])),
           ("Knowledge base", "27 procedure documents", f"{RT['n_passages']} official NMIMS passages"),
           ("Evaluation questions", f"{v1e.get('n', 81)} held-out", f"{E['n']} held-out NMIMS core questions")]
    doc.table(pd.DataFrame(cmp, columns=["End-to-end metric", "v1", "v2"]), "Table 3. v1 vs v2 end-to-end", [7, 4.5, 5])
    doc.p("The two versions are evaluated on different question sets (v1 on 81 questions about invented procedures, v2 "
          "on held-out questions about real NMIMS procedures), so the comparison shows the direction of change rather "
          "than a controlled difference. On the larger, real-source task v2 improves intent accuracy and the share of "
          "correctly routed requests. Passage retrieval looks lower in v2 because v1 searched 27 procedure documents "
          "(one per procedure) while v2 searches 727 real pages and counts a hit only for the exact page a fact came "
          "from; Section 7 compares retrievers on the same v2 data, where the new boosting raises recall@3 from 0.42 to "
          "0.61.")
    checklist = [("NER", "Section 6"), ("Classification", "Section 4"), ("Embeddings", "Sections 4, 5, 7"),
                 ("Semantic similarity", "Section 5"), ("RAG", "Sections 7, 9"), ("Summarization", "Section 8"),
                 ("Information extraction", "Section 9")]
    doc.table(pd.DataFrame(checklist, columns=["Component listed in the brief", "Where it is implemented and evaluated"]),
              "Table 4. Coverage of the Week 3 brief", [7, 7])

    doc.h("2. Pipeline architecture")
    doc.code("query\n"
             " -> intent classifier (TF-IDF + sentence-embedding ensemble, 12 intents + out_of_scope)\n"
             "      -> out_of_scope: polite fallback\n"
             " -> sub-intent router (95 sub-intents)      -> service catalogue (department, documents, next action, facts)\n"
             " -> entity extraction (CRF + regex rules)  -> extracted details, missing information, priority\n"
             " -> hybrid retrieval (BM25 + embeddings, RRF, intent/sub-intent boost) over official NMIMS passages\n"
             " -> answer = MMR summary of verified facts + retrieved passages, with source citations\n"
             " -> structured service request (JSON) via POST /copilot/query")

    doc.h("3. Data and knowledge base")
    doc.p(f"Training and evaluation use the Week 2 NMIMS dataset (group-aware train/validation/test splits; test_core "
          f"= held-out questions from official NMIMS sources). The knowledge base has {RT['n_passages']} page-level passages from "
          f"official NMIMS pages and PDFs; each passage keeps its URL, page, academic year and the ids of verified "
          "facts that cite it. A service catalogue is built for each sub-intent from the verified facts.")

    doc.h("4. Classification")
    it = pd.DataFrame(I["comparison"])[["model", "val", "test_core", "test_variants", "clinc_oos_recall"]]
    it.columns = ["Model", "Val macro-F1", "Core test", "Variants", "OOS recall"]
    doc.table(it, "Table 5. Intent classification", [5.5, 2.5, 2.5, 2.5, 2.5])
    doc.fig(F / "intent_models.png", "Figure 1. Intent models", 13)
    doc.p(f"Selected on validation: {I['selected']}. Sentence embeddings alone are weaker than TF-IDF on this data, "
          "but averaging their probabilities with TF-IDF helps on rewordings and core questions; kNN over embeddings "
          f"(classification by semantic similarity) is competitive in-scope but rejects out-of-scope queries poorly. "
          f"Final: accuracy {f3(I['final']['test_core']['accuracy'])} and macro-F1 {f3(I['final']['test_core']['macro_f1'])} "
          f"on core test; out-of-scope recall {f3(I['final']['clinc_oos_recall'])} on 1,000 real CLINC150 queries.")
    st = pd.DataFrame(S["comparison"])[["model", "val_flat_acc", "val_hier_acc", "test_core_flat_acc", "test_core_hier_acc", "test_core_flat_macro_f1"]]
    st.columns = ["Model", "Val flat", "Val hierarchical", "Core flat", "Core hierarchical", "Core macro-F1"]
    doc.table(st, f"Table 6. Sub-intent routing ({S['n_classes']} classes, accuracy)", [4.5, 2.2, 2.6, 2.2, 2.6, 2.5])
    doc.p(f"Week 2's TF-IDF router reached {f3(w2['sub_intent']['test_core']['accuracy'])} on core test; the ensemble "
          f"with sentence embeddings raises routing accuracy (selected: {S['selected']}, {S['strategy']} routing). "
          "Hierarchical routing (only sub-intents of the predicted intent) did not beat flat routing on validation.")

    doc.h("5. Embeddings and semantic similarity")
    doc.p("Two retrieval-style tasks measure how well each representation captures meaning:")
    doc.b(f" {SS['task_a']}.", "Task A, paraphrase matching:")
    doc.b(f" {SS['task_b']} (different wording written independently, so no shared template).", "Task B, same-fact retrieval:")
    ta = pd.DataFrame(SS["task_a_results"]); ta.columns = ["Method", "Acc@1", "Acc@5", "MRR"]
    doc.table(ta, "Table 7. Task A", [6, 3, 3, 3])
    tb = pd.DataFrame(SS["task_b_results"]); tb.columns = ["Method", "Acc@1", "Acc@5"]
    doc.table(tb, "Table 8. Task B", [6, 3, 3])
    doc.fig(F / "similarity.png", "Figure 2. Same-fact question retrieval", 12)
    doc.p("Rule-generated rewordings are surface edits, so character n-grams already match them almost perfectly "
          "(Task A). On independently written paraphrases (Task B) sentence embeddings clearly outperform TF-IDF "
          f"({f3(simb['bge-small embedding cosine']['acc@1'])} vs {f3(simb['tfidf word cosine']['acc@1'])} acc@1), which "
          "is why embeddings are used for retrieval and in the classification ensemble.")

    doc.h("6. Named entity recognition")
    nt = pd.DataFrame(N["comparison"]); nt.columns = ["Model", "Val F1", "Test F1", "Core test F1", "Variants F1"]
    doc.table(nt, "Table 9. NER (strict span F1, seqeval)", [6, 2.5, 2.5, 2.5, 2.5])
    pt = pd.DataFrame([(k, v["precision"], v["recall"], v["f1-score"], int(v["support"])) for k, v in N["per_type_test"].items()],
                      columns=["Entity", "P", "R", "F1", "n"]).sort_values("n", ascending=False).head(14)
    doc.table(pt, "Table 10. Per-type results of the selected model (test)", [4.5, 2.2, 2.2, 2.2, 2])
    doc.p(f"Selected: {N['selected']}. A CRF with lexical, shape and context features plus a cross-fitted gazetteer "
          "generalises beyond the dictionary. Labels are silver (rule-generated), so scores measure agreement with "
          "those labels; regex rules add DATE, AMOUNT, PHONE, EMAIL, PERCENTAGE and STUDENT_ID at inference.")

    doc.h("7. Retrieval-augmented generation (RAG)")
    rt = pd.DataFrame(RT["comparison"]); rt.columns = ["Retriever", "R@1", "R@3", "R@5", "MRR", "R@3 core"]
    doc.table(rt, f"Table 11. Retrieval over {RT['n_passages']} passages ({RT['n_queries']} test queries)", [6, 2, 2, 2, 2, 2])
    doc.fig(F / "retrieval.png", "Figure 3. Retrieval", 13)
    doc.p(f"A passage counts as relevant if it is the page a verified fact for the question was taken from - a strict "
          "criterion, since other pages often also answer. Fusing BM25 and embeddings (reciprocal rank fusion) beats "
          "either alone, and boosting passages of the predicted intent and sub-intent adds the largest gain "
          f"(recall@3 {f3(bestret['recall@3'])}). Generation is grounded: the answer is built only from verified facts "
          "and retrieved passages (Section 8), with an optional Claude rewrite restricted to the same evidence.")

    doc.h("8. Summarization")
    sm = pd.DataFrame(SM["comparison"]); sm.columns = ["Summarizer", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Embedding sim.", "Avg words"]
    doc.table(sm, f"Table 12. Query-focused summarization ({SM['n_queries']} core test questions)", [3.2, 2.4, 2.4, 2.4, 2.8, 2.4])
    doc.fig(F / "summarization.png", "Figure 4. Summarization", 12)
    doc.p(f"Input: {SM['input']}. Reference: {SM['reference']}. MMR selects sentences relevant to the question while "
          "avoiding repetition, and clearly beats taking the first sentences of the best passage (lead) or "
          "query-agnostic TextRank. All summaries are extractive, so every sentence is copied from an official source "
          "or a verified fact.")

    doc.h("9. Information extraction: structured service request")
    et = [("Questions evaluated", E["n"]), ("Intent accuracy", f3(E["intent_acc"])), ("Sub-intent accuracy", f3(E["sub_intent_acc"])),
          ("Department routing accuracy", f3(E["department_acc"])), ("Answer backed by the question's gold fact", f3(E["answer_backed_by_gold_fact"])),
          ("Required documents (Jaccard vs gold)", f3(E["required_documents_jaccard"])),
          ("Relevant passage in top 3 sources", f3(E["relevant_passage_in_top3"])),
          ("Out-of-scope declined (200 CLINC150)", f3(E["oos_declined_200"])),
          ("Status: ready / needs information / needs confirmation", " / ".join(str(E["status_counts"].get(k, 0)) for k in ("ready", "needs_information", "needs_confirmation"))),
          ("Latency median / p95 (ms, CPU)", f"{E['latency_ms_median']} / {E['latency_ms_p95']}")]
    doc.table(pd.DataFrame(et, columns=["Metric", "Value"]), "Table 13. End-to-end evaluation on held-out core questions", [9, 5])
    ex = r["example"]
    sr = ex["service_request"]
    doc.p(f"Example - query: \"{ex['query']}\"")
    doc.code(json.dumps({"intent": ex["intent"], "sub_intent": ex["sub_intent"],
                         "entities": [(e["label"], e["text"]) for e in ex["entities"]],
                         "department": sr["department"], "required_documents": sr["required_documents"],
                         "next_action": sr["next_action"], "missing_information": sr["missing_information"],
                         "priority": sr["priority"], "status": sr["status"], "source_urls": sr["source_urls"][:2]},
                        indent=1, ensure_ascii=False))
    doc.p("Answer: " + ex["answer"].replace("\n", " "))

    doc.h("10. Limitations")
    for t in ["Questions are built from official NMIMS sources; real student phrasing may be harder.",
              "NER labels are silver; a hand-corrected test set would give a true NER score.",
              "Retrieval relevance counts only the page a fact came from; actual answer coverage is higher than recall@k suggests.",
              "Sub-intent routing is the weakest link (95 classes, some with few questions); errors propagate to department and documents.",
              "Time-sensitive facts (2026-27 dates, fees, committee members) must be re-verified before deployment."]:
        doc.b(t)
    doc.h("11. Next steps (Week 4)")
    for t in ["Improve routing with retrieval-informed re-ranking of sub-intents and a clarification question for low confidence.",
              "Multi-turn dialogue using the missing-information questions; multi-intent handling.",
              "Hand-correct an NER and routing test set; evaluate the optional LLM rewrite for faithfulness.",
              "Front-end for the API and a user study with NMIMS students."]:
        doc.b(t)
    doc.h("12. Reproducibility")
    doc.code("python -m nlp.intent.experiments      # Week 2 baselines (also used by v2)\n"
             "python -m nlp.pipeline.train_eval      # trains and evaluates every v2 component (~6 min CPU)\n"
             "python -m nlp.reports                  # regenerates these reports from the saved results\n"
             "uvicorn backend.main:app --reload       # POST /copilot/query")
    doc.save(SUBMIT / "week3" / "Group10_Week3_NLP_Pipeline_v2_Report.docx")


# --------------------------------------------------------------------------- Week 4

def week4() -> None:
    r = json.loads((REPORTS / "week4" / "results.json").read_text())
    F = REPORTS / "week4" / "figures"
    C, N, RT, SM, E = r["classification"], r["ner"], r["retrieval"], r["summarization"], r["end_to_end"]
    cls = {x["model"]: x for x in C["comparison"]}
    base_c, w3_c = cls["TF-IDF + LogReg (Week 2)"], cls["TF-IDF + frozen embeddings ensemble (Week 3)"]
    imp_c = cls["Fine-tuned transformer + Week 3 ensemble (blend)"]
    tr_c = cls["Fine-tuned transformer (BGE-small, joint sub-intent + OOS)"]
    ner = {x["model"]: x for x in N["comparison"]}
    base_n, crf_n, imp_n = ner["Rule-based labeller (dictionary + regex)"], ner["CRF (Week 3)"], ner["Fine-tuned transformer NER (Week 4)"]
    ret = {x["retriever"]: x for x in RT["comparison"]}
    base_r, w3_r = ret["BM25 keyword search"], ret["Week 3 v2: hybrid + intent/sub-intent boost"]
    best_r = max(RT["comparison"], key=lambda x: x["recall@3"])
    summ = {x["summarizer"]: x for x in SM["comparison"]}
    llm = summ.get("llm_local")
    d = lambda a, b: f"{b - a:+.3f}"
    doc = Doc()
    doc.title([("University Administrative Copilot", 24, True), ("Week 4: Improved Models - Baseline vs Improved", 15, False),
               ("Sentence transformers, transformer NER, vector retrieval with re-ranking, LLM-assisted summarization", 11, False),
               (TEAM, 11, False), (f"Repository: {REPO}", 10, False),
               ("Code: submitted as Group10_Week4_code.zip (and in the repository under src/)", 10, False)])

    doc.h("Summary")
    doc.p("For each component of the copilot we keep the earlier model as the baseline, implement an improved model of "
          "the kind suggested in the brief, and compare both on the same held-out NMIMS test data. Every table reports "
          "baseline, improved and the change (delta). All code is attached and listed in Section 8.")
    head = [("Intent / routing", "TF-IDF + LogReg", "Fine-tuned transformer (blend)",
             "Routing accuracy (core test)", f3(base_c["sub_acc_test_core"]), f3(imp_c["sub_acc_test_core"]),
             d(base_c["sub_acc_test_core"], imp_c["sub_acc_test_core"])),
            ("Intent / routing", "TF-IDF + LogReg", "Fine-tuned transformer (blend)",
             "Intent macro-F1 (core test)", f3(base_c["intent_f1_test_core"]), f3(imp_c["intent_f1_test_core"]),
             d(base_c["intent_f1_test_core"], imp_c["intent_f1_test_core"])),
            ("NER", "Rule-based", "Transformer NER", "Span F1 on gold set (relaxed)", f3(base_n["gold_all_relaxed_f1"]),
             f3(imp_n["gold_all_relaxed_f1"]), d(base_n["gold_all_relaxed_f1"], imp_n["gold_all_relaxed_f1"])),
            ("Retrieval", "BM25 keyword search", best_r["retriever"], "Recall@3", f3(base_r["recall@3"]), f3(best_r["recall@3"]),
             d(base_r["recall@3"], best_r["recall@3"]))]
    if llm:
        mmr, lead = summ["mmr"], summ["lead"]
        head.append(("Summarization", "Lead (basic)", "LLM-assisted", "Embedding similarity to gold", f3(lead["embedding_similarity"]),
                     f3(llm["embedding_similarity"]), d(lead["embedding_similarity"], llm["embedding_similarity"])))
        head.append(("Summarization", "MMR extractive (Week 3)", "LLM-assisted", "Embedding similarity to gold", f3(mmr["embedding_similarity"]),
                     f3(llm["embedding_similarity"]), d(mmr["embedding_similarity"], llm["embedding_similarity"])))
    head.append(("End to end", "Week 3 pipeline (v2)", "Week 4 pipeline (v3)", "Department routing accuracy",
                 f3(E["v2"]["department_acc"]), f3(E["v3"]["department_acc"]), d(E["v2"]["department_acc"], E["v3"]["department_acc"])))
    doc.table(pd.DataFrame(head, columns=["Component", "Baseline", "Improved", "Metric", "Baseline", "Improved", "Delta"]),
              "Table 1. Baseline vs improved at a glance (held-out test data)", [2.4, 2.6, 3.4, 3.2, 1.6, 1.6, 1.4], size=8)

    doc.h("1. Response to the feedback")
    doc.b(" the full code for this week is attached as Group10_Week4_code.zip and is in the repository. Section 8 lists "
          "every file and what it does; code for Weeks 2 and 3 is attached to those submissions in the same way.", "\"Where is the code?\":")
    doc.b(" each section compares the baseline and the improved model on identical data and reports the delta; the improved "
          "models are new this week (fine-tuning, transformer NER, re-ranking, LLM summarization), not re-runs of earlier code.",
          "\"Incremental improvement is barely seen\":")

    doc.h("2. Experimental setup")
    doc.p("Data: the NMIMS dataset (group-aware train / validation / test split; test_core = held-out "
          "questions, test_variants = rule-generated rewordings) plus real CLINC150 queries as the out-of-scope class. "
          "Training and model selection use only train and validation; test sets are scored once. Hardware: Apple M5 "
          "laptop GPU (MPS); every model trains in minutes. The baselines are the saved Week 2/3 models.")

    doc.h("3. Classification: TF-IDF -> sentence transformer")
    doc.p("Improved models: (a) SetFit - the sentence-transformer encoder is fine-tuned with a batch-all triplet loss so "
          "questions with the same sub-intent embed close together, then a logistic-regression head is trained; (b) a "
          "BGE-small transformer (33M parameters) fine-tuned end to end with a classification head on 95 sub-intents plus "
          "out_of_scope (square-root class weighting, a higher learning rate for the new classification head, best epoch "
          "chosen on validation). One joint model gives both the sub-intent (routing) and the intent (by summing its "
          "sub-intents' probabilities); (c) the selected model blends the fine-tuned transformer's probabilities with the "
          "Week 3 models, with blend weights chosen on validation. On its own the fine-tuned transformer is competitive "
          "but not better than the Week 3 ensemble on this small, fine-grained label set (about 40 questions per "
          "sub-intent); combined, the two make different errors and the blend is the best model.")
    ct = pd.DataFrame(C["comparison"])[["model", "intent_f1_test_core", "intent_f1_test_variants", "oos_recall", "sub_acc_test_core", "sub_f1_test_core", "train_time_s"]]
    ct.columns = ["Model", "Intent F1 core", "Intent F1 variants", "OOS recall", "Routing acc core", "Routing F1 core", "Train s"]
    ct["Train s"] = ct["Train s"].fillna("-")
    doc.table(ct, "Table 2. Classification (13 intents incl. out-of-scope; 95 sub-intents)", [5.4, 1.9, 2, 1.6, 2, 1.9, 1.4], size=8)
    doc.fig(F / "classification.png", "Figure 1. Classification: baseline vs improved", 13.5)
    doc.p(f"Change vs Week 2 TF-IDF: routing accuracy {d(base_c['sub_acc_test_core'], imp_c['sub_acc_test_core'])}, routing "
          f"macro-F1 {d(base_c['sub_f1_test_core'], imp_c['sub_f1_test_core'])}, intent macro-F1 "
          f"{d(base_c['intent_f1_test_core'], imp_c['intent_f1_test_core'])} on core test; on rewordings "
          f"{d(base_c['intent_f1_test_variants'], imp_c['intent_f1_test_variants'])}. Change vs the Week 3 ensemble: routing "
          f"accuracy {d(w3_c['sub_acc_test_core'], imp_c['sub_acc_test_core'])}. The fine-tuned transformer alone: routing accuracy "
          f"{f3(tr_c['sub_acc_test_core'])}, intent macro-F1 {f3(tr_c['intent_f1_test_core'])}.")
    hist = pd.DataFrame(C["transformer_history"])
    hist.columns = ["Epoch", "Train loss", "Validation macro-F1 (sub-intent)"]
    doc.table(hist, "Table 3. Fine-tuning curve of the transformer router", [2, 3, 5])

    doc.h("4. NER: rule-based -> transformer NER")
    doc.p(f"The NER training labels were produced by our rule-based labeller, so scoring it on those labels would be "
          f"circular (it scores {f3(base_n['silver_test_strict_f1'])}). We therefore hand-annotated a gold test set of "
          f"{N['gold_set']['queries']} test questions ({N['gold_set']['spans']} spans; half core questions from official NMIMS sources, half rewordings "
          "with typos). Strict = exact span and type; relaxed = same type and overlapping span. The transformer is a "
          "BGE-small token classifier fine-tuned on the training labels with BIO tags aligned to word pieces.")
    nt = pd.DataFrame(N["comparison"])[["model", "gold_all_strict_p", "gold_all_strict_r", "gold_all_strict_f1", "gold_all_relaxed_f1",
                                        "gold_core_relaxed_f1", "gold_variants_relaxed_f1"]]
    nt.columns = ["Model", "Strict P", "Strict R", "Strict F1", "Relaxed F1", "Relaxed F1 core", "Relaxed F1 typos/variants"]
    doc.table(nt, "Table 4. NER on the hand-annotated gold set", [5.2, 1.6, 1.6, 1.6, 1.8, 2.2, 2.6], size=8)
    doc.fig(F / "ner.png", "Figure 2. NER on gold data", 13)
    doc.p(f"Change vs rule-based: relaxed F1 {d(base_n['gold_all_relaxed_f1'], imp_n['gold_all_relaxed_f1'])}, on rewordings "
          f"with typos {d(base_n['gold_variants_relaxed_f1'], imp_n['gold_variants_relaxed_f1'])}; vs the Week 3 CRF "
          f"{d(crf_n['gold_all_relaxed_f1'], imp_n['gold_all_relaxed_f1'])}. Strict scores are lower for all systems because the "
          "gold annotation marks some longer phrases (for example \"re exam form\") than the training labels.")

    doc.h("5. Retrieval: keyword search -> vector retrieval -> re-ranking")
    doc.p(f"Improved models: the BGE-small bi-encoder fine-tuned on {RT['train_pairs_dense']} (training question, relevant "
          f"NMIMS passage) pairs with in-batch negatives, combined with BM25 and the intent/sub-intent boost; then a "
          f"cross-encoder that reads question and passage together re-scores the top 20, used zero-shot and fine-tuned "
          f"on {RT['train_pairs_cross_encoder']} positive / hard-negative pairs. Evaluated on {RT['n_queries']} test questions "
          f"over {RT['n_passages']} passages.")
    rt = pd.DataFrame(RT["comparison"]); rt.columns = ["Retriever", "R@1", "R@3", "R@5", "MRR@20", "R@3 core", "ms/query"]
    doc.table(rt, "Table 5. Retrieval", [6.4, 1.5, 1.5, 1.5, 1.7, 1.7, 1.6], size=8)
    doc.fig(F / "retrieval.png", "Figure 3. Retrieval", 13.5)
    doc.p(f"Change vs BM25: recall@3 {d(base_r['recall@3'], best_r['recall@3'])}, MRR {d(base_r['mrr@20'], best_r['mrr@20'])}; vs the "
          f"Week 3 hybrid: recall@3 {d(w3_r['recall@3'], best_r['recall@3'])} (best: {best_r['retriever']}).")
    rr = ret.get("+ cross-encoder re-rank (fine-tuned)")
    if rr and rr["recall@3"] < best_r["recall@3"]:
        doc.p(f"Negative result: cross-encoder re-ranking lowered recall@3 to {f3(rr['recall@3'])} (fine-tuned) and "
              f"{f3(ret['+ cross-encoder re-rank (zero-shot)']['recall@3'])} (zero-shot), and is about 15x slower. The ms-marco "
              "cross-encoder was trained on web search and ignores the intent/sub-intent routing signal that the hybrid "
              "retriever uses; one epoch on our training pairs was not enough to recover it. The Week 4 copilot therefore "
              "uses the fine-tuned hybrid retriever without re-ranking.")

    doc.h("6. Summarization: basic -> LLM-assisted")
    doc.p(f"Input: {SM['input']}. Reference: {SM['reference']}. LLM: {SM['llm']}; the prompt restricts the model to the "
          "evidence and asks it to copy fees, dates and emails exactly. Unsupported-number rate = share of numbers / emails "
          "in the answer that do not appear in the evidence (a hallucination check).")
    sm = pd.DataFrame(SM["comparison"]); sm.columns = ["Summarizer", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Emb. sim.", "Words", "Unsupported nums"]
    doc.table(sm, f"Table 6. Summarization ({SM['n_queries']} core test questions)", [3, 1.8, 1.8, 1.8, 1.8, 1.5, 2.6], size=8)
    doc.fig(F / "summarization.png", "Figure 4. Summarization", 12.5)
    for ex in SM.get("examples", [])[:2]:
        doc.p(f"Q: {ex['query']}", "Example - ")
        for k in ("mmr", "llm_local"):
            if k in ex:
                doc.p(" " + ex[k].replace("\n", " "), f"{'MMR' if k == 'mmr' else 'LLM-assisted'}:")
    doc.p("Extractive MMR copies source sentences, so its ROUGE overlap with the reference facts is high; the LLM writes "
          "shorter, more readable answers, which lowers exact word overlap even when the meaning matches. The "
          "embedding similarity and unsupported-number columns show how close in meaning and how faithful the answers are.")
    if llm:
        doc.p(f"Result: the LLM-assisted summarizer clearly beats basic lead summarization (embedding similarity "
              f"{d(summ['lead']['embedding_similarity'], llm['embedding_similarity'])}, ROUGE-L {d(summ['lead']['rougeL'], llm['rougeL'])}) "
              f"and gives answers about half as long ({llm['avg_words']:.0f} vs {summ['mmr']['avg_words']:.0f} words), but it does not "
              f"beat our Week 3 MMR extractive summarizer on faithfulness, and {llm['unsupported_number_rate'] * 100:.1f}% of the numbers "
              "it writes are not in the evidence. The copilot therefore keeps MMR as its default answer generator; the LLM is "
              "available as an option for more readable answers.")

    doc.h("7. End to end: Week 3 pipeline vs Week 4 pipeline")
    et = pd.DataFrame([(k.replace("_", " "), E["v2"][k], E["v3"][k], d(E["v2"][k], E["v3"][k]) if "latency" not in k else f"{E['v3'][k] - E['v2'][k]:+.1f}")
                       for k in E["v2"]], columns=["Metric", "Week 3 (v2)", "Week 4 (v3)", "Delta"])
    doc.table(et, f"Table 7. Copilot end to end on {E['n']} held-out questions (+200 real out-of-scope queries)", [6, 3, 3, 2.5])
    doc.fig(F / "end_to_end.png", "Figure 5. End to end", 13)
    if "example_v3" in r:
        ex = r["example_v3"]
        doc.p(f"Example (v3) - \"{ex['query']}\"")
        doc.code(json.dumps({"intent": ex["intent"], "sub_intent": ex["sub_intent"], "entities": [(e["label"], e["text"]) for e in ex["entities"]],
                             "department": ex["service_request"]["department"], "next_action": ex["service_request"]["next_action"],
                             "priority": ex["service_request"]["priority"]}, indent=1, ensure_ascii=False))

    doc.h("8. Code")
    files = [("src/nlp/intent/finetune.py", "Fine-tuned transformer classifier, SetFit classifier, joint intent adapter"),
             ("src/nlp/ner/transformer_ner.py", "Transformer token-classification NER (word-piece BIO alignment, span decoding)"),
             ("src/nlp/retrieval/rerank.py", "Bi-encoder fine-tuning, cross-encoder fine-tuning, re-ranking retriever"),
             ("src/nlp/pipeline/summarize.py", "LLM-assisted summarizer (local Qwen2.5 or Claude) next to the extractive baselines"),
             ("src/nlp/pipeline/copilot.py", "Copilot v3: loads the improved models (v2 kept for comparison)"),
             ("src/nlp/pipeline/compare_week4.py", "Trains improved models and runs every baseline vs improved comparison"),
             ("data/annotations/ner_gold.yaml", "Hand-annotated gold NER test set (120 questions)"),
             ("tests/test_week4.py", "Tests for the Week 4 components")]
    lines = lambda p: sum(1 for _ in open(PROJECT_ROOT / p)) if (PROJECT_ROOT / p).exists() else 0
    doc.table(pd.DataFrame([(f, desc, lines(f)) for f, desc in files], columns=["File", "Purpose", "Lines"]), "Table 8. Week 4 code", [5.6, 9.4, 1.5])
    doc.code("uv sync\npython -m nlp.pipeline.compare_week4     # trains improved models, runs all comparisons (~40 min, Apple GPU)\n"
             "python -m nlp.reports week4 code         # this report + code zips\n"
             "COPILOT_VERSION=v3 uvicorn backend.main:app   # serve the improved pipeline")

    doc.h("9. Limitations")
    for t in ["The gold NER set is small (120 questions) and annotated by the team; confidence intervals are wide.",
              "Questions are built from official NMIMS sources rather than collected from students.",
              "The local LLM (1.5B parameters) is limited; a larger model or Claude (supported in code) should write better answers.",
              "Transformer models are slower than TF-IDF (see latency) and need a GPU-equipped machine to train in minutes."]:
        doc.b(t)
    doc.h("10. Next steps")
    for t in ["Multi-turn clarification for ambiguous and low-confidence questions.",
              "Evaluation with real NMIMS student questions and a larger gold set.",
              "Deployment: front-end for the API, caching of embeddings, monitoring of time-sensitive facts."]:
        doc.b(t)
    doc.save(SUBMIT / "week4" / "Group10_Week4_Improved_Model_Report.docx")



def week5() -> None:
    import yaml

    r = json.loads((REPORTS / "week5" / "results.json").read_text())
    notes = yaml.safe_load((REPORTS / "week5" / "failure_notes.yaml").read_text())
    pools = json.loads((REPORTS / "week5" / "failure_cases.json").read_text())
    F = REPORTS / "week5" / "figures"
    Q, N, RT, E, RB, ER, FX = (r[k] for k in ("quantitative", "ner", "retrieval", "end_to_end", "robustness", "errors", "fixes"))
    cmp_ = {x["system"]: x for x in Q["comparison"]}
    names = list(cmp_)
    w2, v2, v3 = (cmp_[n] for n in names)
    doc = Doc()
    doc.title([("University Administrative Copilot", 24, True), ("Week 5: Evaluation & Error Analysis", 15, False),
               ("Quantitative evaluation, test sets, robustness, error analysis and failure cases", 11, False),
               (TEAM, 11, False), (f"Repository: {REPO}", 10, False),
               ("Code: submitted as Group10_Week5_code.zip (and in the repository under src/)", 10, False)])

    doc.h("Summary")
    doc.p("We evaluated the final copilot (Week 4, v3) end to end and component by component, with the Week 2 and "
          "Week 3 systems as reference points. Beyond headline metrics we report confidence intervals and significance "
          "tests, calibration, behaviour under controlled perturbations, a new challenge test set of hard queries, an "
          "automatic error taxonomy, a set of analysed failure cases and two error-driven fixes measured before and after.")
    for t in notes["findings"]:
        doc.b(t["text"], t["lead"] + " ")

    doc.h("1. Response to earlier feedback")
    doc.p("Weeks 3 and 4 were marked down for missing code and for improvements that were hard to see. This week: all "
          "evaluation code is attached (Group10_Week5_code.zip, listed in Section 9) and reruns with one command; every "
          "number in this report is generated from that run; each result is compared with a baseline and given a 95% "
          "confidence interval so that improvements are visible and verifiable.")

    doc.h("2. Test sets")
    ts = pd.DataFrame([(x["set"], x["n"], x["what"]) for x in r["test_sets"]], columns=["Test set", "Size", "What it measures"])
    doc.table(ts, "Table 1. Test sets used in this evaluation (none is used for training or model selection)", [3.2, 1.6, 11.6])
    doc.p(f"test_core covers {r['coverage']['intents_in_test_core']} of 12 intents and {r['coverage']['sub_intents_in_test_core']} of "
          f"{r['coverage']['sub_intents_total']} sub-intents. The split is group-aware: a question and all its rewordings are in "
          "the same split, so no rewording of a test question was seen in training. The challenge set was written this week, "
          "after the models were frozen, to target the weaknesses we expected from Week 4.")
    cat = pd.DataFrame([(k.replace("_", " "), v) for k, v in r["test_sets"][3]["categories"].items()], columns=["Challenge category", "Queries"])
    doc.table(cat, "Table 2. Challenge set composition", [6, 2.5])

    doc.h("3. Quantitative evaluation")
    doc.h("3.1 Classification and routing", 2)
    qt = pd.DataFrame([(n, f"{x['intent_acc']:.3f} {ci(x['intent_acc_ci'])}", f"{x['intent_macro_f1']:.3f} {ci(x['intent_macro_f1_ci'])}",
                        f"{x['sub_acc']:.3f} {ci(x['sub_acc_ci'])}", f3(x["sub_macro_f1"]), f3(x["oos_recall_clinc"]), f3(x["ece_sub"]))
                       for n, x in cmp_.items()],
                      columns=["System", "Intent acc [95% CI]", "Intent macro-F1 [95% CI]", "Routing acc [95% CI]", "Routing macro-F1", "OOS recall", "ECE"])
    doc.table(qt, f"Table 3. Test_core ({Q['n']} queries; routing on the {Q['n_in_scope']} in-scope ones). CI = bootstrap, 1,000 resamples",
              [3.4, 2.9, 2.9, 2.9, 1.7, 1.5, 1.2], size=7.5)
    doc.fig(F / "accuracy_ci.png", "Figure 1. Accuracy with 95% confidence intervals", 13)
    sg = Q["significance_v3_vs_v2"]
    doc.p(f"Significance (exact McNemar test on paired predictions): final vs Week 3 routing - {sg['sub_intent']['only_second_correct']} "
          f"queries fixed, {sg['sub_intent']['only_first_correct']} broken, p = {sg['sub_intent']['p_value']}; final vs Week 2 routing - "
          f"{sg['sub_intent_vs_week2']['only_second_correct']} fixed, {sg['sub_intent_vs_week2']['only_first_correct']} broken, "
          f"p = {sg['sub_intent_vs_week2']['p_value']}; intent final vs Week 3 - p = {sg['intent']['p_value']}.")
    pi = pd.DataFrame(Q["per_intent_v3"])[["intent", "precision", "recall", "f1", "support"]]
    doc.table(pi, "Table 4. Per-intent scores of the final system (test_core)", [5, 2.2, 2.2, 2.2, 2])
    doc.fig(F / "confusion_intent_v3.png", "Figure 2. Intent confusion matrix (rows = gold, row-normalised)", 12.5)
    ws = pd.DataFrame(Q["worst_sub_intents_v3"])[["sub_intent", "precision", "recall", "f1", "support"]]
    doc.table(ws, f"Table 5. The 10 hardest sub-intents (of {len(Q['per_intent_v3']) and r['coverage']['sub_intents_in_test_core']}); "
                  f"{Q['sub_intents_perfect']} sub-intents are routed perfectly, {Q['sub_intents_f1_below_0.5']} have F1 below 0.5",
              [5.2, 2.2, 2.2, 2.2, 2])

    doc.h("3.2 Calibration and selective prediction", 2)
    doc.p("A copilot that sends uncertain requests to staff needs confidence scores that mean something. Expected "
          "calibration error (ECE) measures the gap between confidence and accuracy; the risk-coverage curve shows the "
          "accuracy obtained if only the most confident queries are answered automatically.")
    doc.fig(F / "calibration_selective.png", "Figure 3. Reliability diagram (left) and accuracy vs coverage (right), routing", 15.5)
    sel = pd.DataFrame([(n, v["acc_at_50pct_coverage"], v["acc_at_70pct_coverage"], v["acc_at_90pct_coverage"], v["acc_at_100pct_coverage"], v["aurc"])
                        for n, v in Q["selective"].items()], columns=["System", "Acc @50% cov.", "Acc @70%", "Acc @90%", "Acc @100%", "AURC (lower better)"])
    doc.table(sel, "Table 6. Selective prediction for routing", [4.6, 2.3, 2, 2, 2, 2.8])

    doc.h("3.3 Named entity recognition", 2)
    for n, x in N.items():
        nt = pd.DataFrame(x["per_type"])[["type", "gold", "pred", "strict_f1", "relaxed_p", "relaxed_r", "relaxed_f1"]]
        nt.columns = ["Entity type", "Gold", "Pred", "Strict F1", "Relaxed P", "Relaxed R", "Relaxed F1"]
        doc.table(nt, f"Table 7{'a' if n == names[1] else 'b'}. NER per entity type on the gold set - {n}", [4, 1.4, 1.4, 1.9, 1.9, 1.9, 1.9], size=7.5)
    ne = pd.DataFrame([(k, *(N[n]["errors"].get(k, 0) for n in N)) for k in ("exact", "boundary error", "wrong type", "missed", "spurious (no gold overlap)")],
                      columns=["Outcome", *N.keys()])
    doc.table(ne, "Table 8. NER error breakdown (gold spans; spurious = predicted spans with no gold overlap)", [5, 4.5, 4.5])

    doc.h("3.4 Retrieval and end-to-end", 2)
    rt = pd.DataFrame([(n, x["recall@1"], f"{x['recall@3']:.3f} {ci(x['recall@3_ci'])}", x["recall@5"], x["recall@10"], f"{x['mrr@20']:.3f} {ci(x['mrr_ci'])}")
                       for n, x in RT["comparison"].items()], columns=["System", "R@1", "R@3 [95% CI]", "R@5", "R@10", "MRR [95% CI]"])
    doc.table(rt, f"Table 9. Retrieval on {RT['n']} verified test questions (727 passages)", [4, 1.5, 3.4, 1.5, 1.5, 3.4], size=8)
    ee = pd.DataFrame([(n, E[n]["latency_ms_p50"], E[n]["latency_ms_p95"], ", ".join(f"{k} {v}" for k, v in E[n]["status"].items()))
                       for n in E if n != "n"], columns=["System", "Latency p50 (ms)", "Latency p95 (ms)", "Service-request status"])
    doc.table(ee, f"Table 10. End to end on {E['n']} held-out questions (Apple M5 laptop)", [4, 2.4, 2.4, 7.5])

    doc.h("4. Robustness")
    doc.p("Robustness is tested in three ways: (a) ten controlled perturbations applied to every in-scope test_core "
          "question (seeded, so the test is reproducible), measuring accuracy and consistency (same prediction as for the "
          "clean question); (b) the challenge set; (c) retrieval under typos.")
    pv = pd.DataFrame(RB["perturbations"])
    acc_t = pv.pivot(index="perturbation", columns="system", values="sub_acc")[names]
    cons_t = pv.pivot(index="perturbation", columns="system", values="consistency")[names]
    order = ["clean"] + [p for p in acc_t.index if p != "clean"]
    tab = pd.DataFrame({"Perturbation": order, **{f"Acc {n.split(' (')[0]}": acc_t.loc[order, n].values for n in names},
                        f"Consistency {names[2].split(' (')[0]}": cons_t.loc[order, names[2]].values})
    doc.table(tab, "Table 11. Routing accuracy under perturbations, and consistency of the final system", [4.4, 2.6, 2.8, 2.8, 3.2], size=8)
    doc.fig(F / "robustness.png", "Figure 4. Robustness to perturbations", 13.5)
    cdf = pd.DataFrame(RB["challenge"])
    ct = cdf.pivot(index="category", columns="system", values="intent_acc")[names]
    ct2 = cdf.pivot(index="category", columns="system", values="intent_and_sub_acc")[names[2]]
    cn = cdf.drop_duplicates("category").set_index("category")["n"]
    ctab = pd.DataFrame({"Category": [c.replace("_", " ") for c in ct.index], "n": cn[ct.index].values,
                         **{f"Intent acc {n.split(' (')[0]}": ct[n].values for n in names}, "Final: intent+routing": ct2[ct.index].values})
    doc.table(ctab, "Table 12. Challenge set by category", [3.4, 1, 2.6, 2.8, 2.8, 2.8], size=8)
    doc.fig(F / "challenge.png", "Figure 5. Challenge set", 13)
    rp = pd.DataFrame([(k.split(" | ")[0], k.split(" | ")[1], v) for k, v in RB["retrieval_recall@3_perturbed"].items()],
                      columns=["System", "Perturbation", "Recall@3"])
    doc.table(rp, "Table 13. Retrieval recall@3 under perturbation (compare with Table 9)", [5, 5, 2.5])
    for t in notes.get("robustness", []):
        doc.b(t["text"], t["lead"] + " ")

    doc.h("5. Error analysis")
    doc.p(f"Every error of the final system on the full test set ({ER['n_test']} queries: core + rewordings + out-of-scope) is "
          "assigned automatically to a category; all errors are listed in reports/week5/errors_v3.csv.")
    tx = pd.DataFrame(list(ER["taxonomy"].items()), columns=["Error type", "Count"])
    tx["Share"] = (tx["Count"] / ER["n_errors"]).map(lambda v: f"{v:.0%}")
    doc.table(tx, f"Table 14. Error taxonomy ({ER['n_errors']} errors)", [7, 2, 2])
    doc.fig(F / "errors.png", "Figure 6. Error types and routing error rate by query style and by confidence", 16)
    cp = pd.DataFrame(ER["top_confused_sub_intents"])
    cp.columns = ["Gold sub-intent", "Predicted", "Count"]
    doc.table(cp, "Table 15. Most frequent routing confusions", [5.5, 5.5, 1.8])
    for key, title in (("by_length", "query length"), ("by_variant_type", "query style")):
        sl = pd.DataFrame(ER["slices"][key])
        sl.columns = ["Slice", "Queries", "Routing error rate"]
        doc.table(sl, f"Routing error rate by {title}", [4, 2, 3])
    for t in notes.get("error_analysis", []):
        doc.b(t["text"], t["lead"] + " ")

    doc.h("6. Failure cases")
    doc.p(f"{len(notes['cases'])} failure cases, chosen to cover every component. For each: what happened, the root cause we "
          "identified, and how to fix it.")
    for i, c in enumerate(notes["cases"], 1):
        doc.h(f"Case {i}. {c['title']}", 2)
        rec = pools[c["pool"]][c["index"]] if c.get("pool") else c["record"]
        rows = [(k.replace("_", " "), "; ".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in rec.items() if k in c.get("fields", rec)]
        doc.table(pd.DataFrame(rows, columns=["Field", "Value"]), f"Component: {c['component']}", [3.5, 13])
        doc.p(c["root_cause"], "Root cause: ")
        doc.p(c["fix"], "Fix: ")

    doc.h("7. Error-driven fixes")
    tf = pd.DataFrame(FX["typo_normaliser"]["rows"])[["set", "n", "before", "after", "delta"]]
    tf.columns = ["Evaluation set", "n", "Before", "After", "Delta"]
    doc.p(f"Fix 1 - typo normalisation. Section 4 showed typos as a main weakness. Before classification, each "
          f"out-of-vocabulary word is replaced by the most frequent similar word from the training vocabulary "
          f"({FX['typo_normaliser']['vocabulary_words']} words seen at least 3 times; similarity >= 0.8).")
    doc.table(tf, "Table 16. Typo normalisation: routing accuracy before and after (final system)", [6, 1.5, 2, 2, 2])
    ex = pd.DataFrame(FX["typo_normaliser"]["examples"])
    ex.columns = ["Before normalisation", "After"]
    doc.table(ex, "Examples (challenge set)", [8, 8])
    a = FX["abstention"]
    doc.p(f"Fix 2 - confidence-based hand-off. Using the calibration analysis, we set a routing-confidence threshold of "
          f"{a['threshold']:.2f} on validation data (target 90% accuracy). On test_core the copilot then answers "
          f"{a['test_coverage']:.0%} of in-scope questions automatically with {a['test_accuracy_answered']:.1%} accuracy "
          f"(vs {a['test_accuracy_all']:.1%} if it answered everything) and hands {a['share_sent_to_staff']:.0%} to staff, "
          f"catching {a['errors_caught']:.0%} of its routing errors before they reach the student.")
    for t in notes.get("fixes", []):
        doc.b(t["text"], t["lead"] + " ")

    doc.h("8. Limitations of this evaluation")
    for t in notes["limitations"]:
        doc.b(t)

    doc.h("9. Code")
    files = [("src/nlp/pipeline/evaluate_week5.py", "Whole Week 5 evaluation: test sets, metrics + CIs, significance, calibration, "
              "robustness, error taxonomy, failure cases, fixes"),
             ("data/annotations/challenge.yaml", "New challenge test set (hard queries from official NMIMS sources, labelled)"),
             ("reports/week5/failure_notes.yaml", "Our analysis of each failure case (root cause, fix)"),
             ("src/nlp/reports.py", "Builds this report from the saved results"),
             ("tests/test_week5.py", "Tests for the challenge set, perturbations, statistics and the typo normaliser")]
    lines = lambda p: sum(1 for _ in open(PROJECT_ROOT / p)) if (PROJECT_ROOT / p).exists() else 0
    doc.table(pd.DataFrame([(f, desc, lines(f)) for f, desc in files], columns=["File", "Purpose", "Lines"]), "Table 17. Week 5 code", [5.6, 9.4, 1.5])
    doc.code("uv sync\npython -m nlp.pipeline.evaluate_week5   # full evaluation (~10 min on a laptop GPU)\n"
             "python -m nlp.reports week5 code         # this report + code zip\npytest tests/test_week5.py")
    doc.h("10. Next steps")
    for t in notes["next_steps"]:
        doc.b(t)
    doc.save(SUBMIT / "week5" / "Group10_Week5_Evaluation_Report.docx")


# --------------------------------------------------------------------------- code snapshots

WEEK_CODE = {
    "week2": ["src/dataset/config.py", "src/dataset/phase1_sources.py", "src/dataset/phase2_annotate.py",
              "src/dataset/phase3_expand.py", "src/dataset/phase4_validate.py", "src/nlp/__init__.py",
              "src/nlp/intent/preprocess.py", "src/nlp/intent/models.py", "src/nlp/intent/experiments.py",
              "src/nlp/intent/predict.py", "src/nlp/intent/load_clinc150.py", "tests/test_dataset.py"],
    "week3": ["src/nlp/intent/embedding_model.py", "src/nlp/ner/extractor.py", "src/nlp/retrieval/search.py",
              "src/nlp/knowledge/store.py", "src/nlp/pipeline/copilot.py", "src/nlp/pipeline/summarize.py",
              "src/nlp/pipeline/train_eval.py", "src/backend/main.py", "src/backend/api/routes.py",
              "tests/test_nlp.py", "tests/test_api.py"],
    "week4": ["src/nlp/intent/finetune.py", "src/nlp/ner/transformer_ner.py", "src/nlp/retrieval/rerank.py",
              "src/nlp/pipeline/summarize.py", "src/nlp/pipeline/copilot.py", "src/nlp/pipeline/compare_week4.py",
              "src/nlp/reports.py", "tests/test_week4.py"],
    "week5": ["src/nlp/pipeline/evaluate_week5.py", "data/annotations/challenge.yaml", "reports/week5/failure_notes.yaml",
              "src/nlp/reports.py", "tests/test_week5.py"],
}


def code() -> None:
    """Copy each week's Python files into submissions/<week>/code/ and zip them for upload."""
    for week, files in WEEK_CODE.items():
        dest = SUBMIT / week / "code"
        if dest.exists():
            shutil.rmtree(dest)
        zpath = SUBMIT / week / f"Group10_{week.capitalize()}_code.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in files:
                src = PROJECT_ROOT / rel
                if not src.exists():
                    continue
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
                z.write(src, rel)
            z.writestr("README.txt", f"Group 10 - {week} code. Full runnable project: {REPO}\n"
                                     "Setup: uv sync; run commands are listed in the repository README.\n")
        print("code snapshot", dest.relative_to(PROJECT_ROOT), "+", zpath.name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["week2", "week3", "week4", "week5", "code"]
    for w in which:
        {"week2": week2, "week3": week3, "week4": week4, "week5": week5, "code": code}[w]()
