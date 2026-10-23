// Demo UI for the elective advisor and the NLP component. Plain JS; every value from the API or the user is
// escaped before it is put into the page.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const STATUS = { eligible: "Eligible", not_recommended: "Not recommended", needs_review: "Needs review", not_eligible: "Not eligible" };
const ICON = { pass: "✓", fail: "✗", warn: "!", unknown: "?" };
const RULE = { nmims: "NMIMS rule", syllabus: "Syllabus", advisor_policy: "Advisor policy" };
const TOPIC_LABEL = { artificial_intelligence: "Artificial intelligence", cybersecurity: "Cybersecurity", management: "Management (open electives)" };
let SAMPLES = {};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  return data;
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
}));

// ------------------------------------------------------------------ syllabus list
function addSyllabus(name, text, open = false) {
  const el = document.createElement("div");
  el.className = "syl" + (open ? " open" : "");
  el.innerHTML = `<div class="syl-head">
      ${/^NEW_/i.test(name) ? '<span class="badge new">NEW</span>' : ""}
      <input class="nm" aria-label="Syllabus name" value="${esc(name)}">
      <button class="toggle" type="button">${open ? "hide" : "show"}</button>
      <button class="x" type="button" aria-label="Remove">×</button></div>
    <textarea rows="9" aria-label="Syllabus text"></textarea>`;
  $("textarea", el).value = text;
  $(".toggle", el).onclick = () => { el.classList.toggle("open"); $(".toggle", el).textContent = el.classList.contains("open") ? "hide" : "show"; };
  $(".x", el).onclick = () => el.remove();
  $("#syllabusList").appendChild(el);
  return el;
}

function loadSample(topic) {
  $("#syllabusList").innerHTML = "";
  (SAMPLES[topic] || []).forEach((c) => addSyllabus(c.course_id, c.syllabus_text));
}

$("#addPaste").onclick = () => $("textarea", addSyllabus("NEW_my_course", "", true)).focus();
$("#files").onchange = async (e) => {
  for (const f of e.target.files) {
    const b64 = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1]); r.readAsDataURL(f); });
    const el = addSyllabus(f.name, "");
    el.dataset.file = b64;
    $("textarea", el).placeholder = `(file ${f.name}: read on the server)`;
  }
  e.target.value = "";
};

function collectSyllabi() {
  return [...document.querySelectorAll("#syllabusList .syl")].map((el) => {
    const name = $(".nm", el).value.trim() || "syllabus";
    const text = $("textarea", el).value;
    return el.dataset.file && !text.trim() ? { name, file_base64: el.dataset.file } : { name, text };
  }).filter((s) => s.file_base64 || (s.text && s.text.trim()));
}

// ------------------------------------------------------------------ rendering
const confBar = (c) => `<div class="conf"><div class="bar ${esc(c.level)}"><span style="width:${Math.round(c.score * 100)}%"></span></div>
  <span><b>${Math.round(c.score * 100)}%</b> confidence (${esc(c.level)})</span></div>`;
const cites = (cs) => cs.filter((c) => c.page || c.quote || c.source.startsWith("F")).map((c) =>
  `<div class="cite">${c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.source)}</a>` : esc(c.source)}${c.page ? `, p. ${esc(c.page)}` : ""}${c.quote ? ` - "${esc(c.quote)}"` : ""}</div>`).join("");

function renderAdvice(out) {
  const rec = out.recommendation, st = out.student;
  let h = "";
  if (rec) {
    const a = out.assessments.find((x) => x.name === rec.course);
    h += `<div class="card hero"><div class="kicker">Recommended elective</div><h2>${esc(rec.title)}</h2>
      <div class="meta">${rec.semester ? `<span class="chip">Semester ${ROMAN[rec.semester]}</span>` : ""}
        ${rec.slot ? `<span class="chip">${esc(rec.slot)}</span>` : ""}${a && a.is_new ? '<span class="badge new">NEW COURSE</span>' : ""}</div>
      ${confBar(rec.confidence)}<ul class="clean">${rec.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
  } else {
    h += `<div class="card hero none"><div class="kicker">No elective to recommend</div>
      <h2>None of these courses fits right now</h2><p class="muted">See each course below for the reason.</p></div>`;
  }
  if (out.per_semester.length > 1) {
    h += `<div class="card"><h3>By semester</h3><ul class="clean">${out.per_semester.map((r) =>
      `<li>Semester ${ROMAN[r.semester]}: <b>${esc(r.title)}</b> <span class="muted small">(${Math.round(r.confidence.score * 100)}%)</span></li>`).join("")}</ul></div>`;
  }
  h += `<div class="card"><h3>Read as</h3><div class="profile">
      <span><b>Programme:</b> ${esc(st.program || "not recognised")}</span>
      <span><b>Year:</b> ${esc(st.year || "?")}</span>
      <span><b>Semester:</b> ${st.semesters.map((s) => ROMAN[s]).join(", ") || "?"}</span>
      <span><b>Match:</b> ${esc(st.program_match)} (${Math.round(st.confidence.score * 100)}%)</span></div>
      ${st.notes.length ? `<ul class="clean small muted">${st.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}</div>`;
  h += `<div class="card"><h3>Every course checked</h3>${out.assessments.map((a) => `
    <div class="course"><div class="course-head" tabindex="0" role="button" aria-expanded="false">
      <span class="pill s-${a.status}">${STATUS[a.status]}</span>
      <span class="t">${esc(a.title)}</span>${a.is_new ? '<span class="badge new">NEW</span>' : ""}
      ${a.semester ? `<span class="chip">Sem ${ROMAN[a.semester]}</span>` : ""}
      ${a.fit_score != null ? `<span class="muted small">fit ${a.fit_score.toFixed(2)}</span>` : ""}
      <span class="why">${esc(a.reasons[0] || "")}</span></div>
      <div class="checks"><table><tbody>${a.checks.map((c) => `<tr><td class="ic ${c.status}" title="${c.status}">${ICON[c.status]}</td>
        <td><b>${esc(c.label)}</b> <span class="rule ${c.rule}">${RULE[c.rule]}</span><br>${esc(c.detail)}${cites(c.citations)}</td></tr>`).join("")}
      </tbody></table></div></div>`).join("")}<p class="muted small">Click a course to see its checks and sources.</p></div>`;
  if (out.overlaps.length) {
    h += `<div class="card"><h3>Overlapping courses</h3><ul class="clean">${out.overlaps.map((o) =>
      `<li><b>${esc(o.a)}</b> and <b>${esc(o.b)}</b> overlap (similarity ${o.similarity.toFixed(2)}): taking both repeats material.</li>`).join("")}</ul></div>`;
  }
  const how = out.how_to_enrol;
  h += `<div class="card"><h3>How to enrol</h3><p><b>${esc(how.department)}</b>: ${esc(how.next_action)}.</p>${cites(how.rules)}</div>`;
  h += `<div class="card"><h3>Before you register</h3><ul class="clean">${out.human_review.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <p class="muted small">${esc(out.disclaimer)}</p>
    <details><summary class="muted small">Structured output (JSON)</summary><pre class="json">${esc(JSON.stringify(out, null, 2))}</pre></details></div>`;
  return h;
}

function wireCourses(root) {
  root.querySelectorAll(".course-head").forEach((hd) => {
    const toggle = () => { const c = hd.parentElement; c.classList.toggle("open"); hd.setAttribute("aria-expanded", c.classList.contains("open")); };
    hd.onclick = toggle;
    hd.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } };
  });
}

// ------------------------------------------------------------------ advisor
$("#run").onclick = async () => {
  const btn = $("#run"); $("#advError").textContent = "";
  const syllabi = collectSyllabi();
  if (!syllabi.length) { $("#advError").textContent = "Add at least one syllabus (or load a sample set)."; return; }
  const num = (id) => ($(id).value ? Number($(id).value) : null);
  const student = { identity: $("#identity").value, program: $("#program").value || null, year: num("#year"),
    semester: num("#semester"), interests: $("#interests").value || null,
    completed_courses: $("#completed").value.split("\n").map((s) => s.trim()).filter(Boolean) };
  btn.disabled = true; btn.textContent = "Checking courses...";
  try {
    const out = await api("/v1/electives/advise", { student, syllabi });
    $("#results").innerHTML = renderAdvice(out); wireCourses($("#results"));
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { $("#advError").textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = "Find my elective"; }
};

// ------------------------------------------------------------------ copilot
const EXAMPLES = ["How do I choose my open elective? Roll no J054", "I lost my ID card, it's urgent",
  "my attendance is 68% in DBMS, what happens now", "can you recommend a good movie"];
$("#queryExamples").innerHTML = EXAMPLES.map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join("");
$("#queryExamples").querySelectorAll("button").forEach((b) => (b.onclick = () => { $("#query").value = b.textContent; $("#ask").click(); }));

$("#ask").onclick = async () => {
  const btn = $("#ask"); $("#askError").textContent = ""; btn.disabled = true; btn.textContent = "Thinking...";
  try {
    const o = await api("/v1/analyze", { text: $("#query").value });
    const ent = o.entities.map((e) => `<span class="chip">${esc(e.label)}: ${esc(e.text)}${e.confidence != null ? ` · ${Math.round(e.confidence * 100)}%` : ""}</span>`).join("");
    $("#answer").innerHTML = `<div class="card hero ${o.intent.in_scope ? "" : "none"}">
      <div class="kicker">Decision: ${esc(o.decision.action.replaceAll("_", " "))}</div>
      <h2>${esc(o.routing ? o.routing.department : "Outside administrative services")}</h2>
      <div class="meta"><span class="chip">Intent ${esc(o.intent.intent)}</span>${o.routing ? `<span class="chip">${esc(o.routing.sub_intent)}</span>` : ""}</div>
      ${confBar(o.intent.confidence)}<p class="muted small">${esc(o.decision.reason)}</p></div>
      <div class="card"><h3>Answer</h3><p class="answer-text">${esc(o.answer)}</p>${ent ? `<h3>Entities</h3><div class="chips">${ent}</div>` : ""}
      ${o.sources.length ? `<h3>Sources</h3><ul class="clean">${o.sources.map((s) => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a> <span class="muted small">${esc(s.page)}</span></li>`).join("")}</ul>` : ""}
      <details><summary class="muted small">Structured output (JSON)</summary><pre class="json">${esc(JSON.stringify(o, null, 2))}</pre></details></div>`;
  } catch (e) { $("#askError").textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = "Ask"; }
};

// ------------------------------------------------------------------ partner integration
$("#pRun").onclick = async () => {
  const btn = $("#pRun"); $("#pError").textContent = ""; btn.disabled = true; btn.textContent = "Calling partner services...";
  const extra = $("#pExtra").value.trim();
  try {
    const r = await api("/v1/electives/advise-from-partners", { topic: $("#pTopic").value, student_id: $("#pStudent").value,
      student: $("#pInterests").value ? { interests: $("#pInterests").value } : null,
      extra_syllabi: extra ? [{ name: "NEW_pasted_course", text: extra }] : [] });
    const log = `<div class="card"><h3>Calls to partner components</h3><table class="log"><tbody>${r.partner_calls.map((c) =>
      `<tr><td>${c.ok ? "✓" : "✗"}</td><td>${esc(c.service)}</td><td>${esc(c.url)}</td><td>${esc(c.ms)} ms</td><td>${c.mock ? "mock" : "live"}</td></tr>`).join("")}</tbody></table>
      ${r.student_record ? `<p class="small muted">Student record received: ${esc(r.student_record.program)}, year ${esc(r.student_record.year)}${r.student_record.semester ? `, semester ${ROMAN[r.student_record.semester]}` : ""}.</p>` : ""}</div>`;
    $("#pResults").innerHTML = log + renderAdvice(r.advice); wireCourses($("#pResults"));
  } catch (e) { $("#pError").textContent = e.message; }
  finally { btn.disabled = false; btn.textContent = "Fetch from partners and advise"; }
};

// ------------------------------------------------------------------ start
(async function init() {
  for (let i = 1; i <= 10; i++) $("#semester").insertAdjacentHTML("beforeend", `<option value="${i}">${ROMAN[i]}</option>`);
  try {
    const [p, s] = await Promise.all([api("/v1/electives/programs"), api("/v1/electives/samples")]);
    p.programs.forEach((x) => $("#program").insertAdjacentHTML("beforeend", `<option>${esc(x)}</option>`));
    SAMPLES = s.topics;
    $("#sampleNote").textContent = s.note;
    const topics = Object.keys(SAMPLES);
    $("#sampleButtons").innerHTML = topics.map((t) => `<button class="chip" type="button" data-t="${esc(t)}">${esc(TOPIC_LABEL[t] || t)}</button>`).join("");
    $("#sampleButtons").querySelectorAll("button").forEach((b) => (b.onclick = () => loadSample(b.dataset.t)));
    $("#pTopic").innerHTML = topics.map((t) => `<option value="${esc(t)}">${esc(TOPIC_LABEL[t] || t)}</option>`).join("");
    loadSample("artificial_intelligence");
    // ?tab=copilot|partner|about opens a tab; ?demo also runs that tab (default: the Group 10 elective demo)
    const q = new URLSearchParams(location.search);
    if (q.get("tab")) document.querySelector(`.tab[data-tab="${CSS.escape(q.get("tab"))}"]`)?.click();
    if (q.has("demo")) ({ copilot: $("#ask"), partner: $("#pRun") }[q.get("tab")] || $("#run")).click();
  } catch (e) { $("#advError").textContent = "Could not reach the API: " + e.message; }
})();
