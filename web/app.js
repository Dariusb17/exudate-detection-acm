// Exora - page logic. All analysis happens on the server (app.py -> exudates/detection.py).
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");

const els = {
  file: $("file"), drop: $("drop"), samples: $("samples"), samplesBlock: $("samples-block"),
  report: $("report"), reportFile: $("report-file"), verdict: $("verdict"),
  verdictTitle: $("verdict-title"), verdictSub: $("verdict-sub"), stats: $("stats"), expert: $("expert"),
  download: $("download"), again: $("again"), error: $("error"), errorText: $("error-text"),
  errorAgain: $("error-again"), tau: $("tau"), tauOut: $("tau-out"), minarea: $("minarea"),
  paperValues: $("paper-values"), defaults: $("defaults"),
  status: $("status"), placeholder: $("placeholder"), compare: $("compare"),
  before: $("img-before"), after: $("img-after"), single: $("img-single"),
  splitInput: $("split-input"), legend: $("legend"),
  tabResult: $("tab-result"), tabSteps: $("tab-steps"), tabExpert: $("tab-expert"),
  steps: $("steps"), strip: $("strip"), prev: $("prev"), next: $("next"),
  stepGroup: $("step-group"), stepNum: $("step-num"), stepTotal: $("step-total"),
  stepTitle: $("step-title"), stepText: $("step-text"),
};

let config = { tau_acm: 28, paper_tau_acm: 20, min_area: 2000 };
let current = null;      // last result from the server
let lastFile = null;     // last photo, so a settings change can re-run it
let stepIndex = 0;
let view = "result";
let previewUrl = null;

/* ---------- small helpers ---------- */
function setState(state) { document.body.dataset.state = state; }

function store(key, value) { try { localStorage.setItem("exora." + key, value); } catch (e) { /* storage off */ } }
function recall(key) { try { return localStorage.getItem("exora." + key); } catch (e) { return null; } }

function setStatus(text) { els.status.textContent = text; }

function legend(items) {
  els.legend.innerHTML = items.map(([color, text]) =>
    `<span><i style="background:${color}"></i>${text}</span>`).join("");
}

/* ---------- reticle ticks around the aperture ---------- */
function drawTicks() {
  const g = $("ticks");
  const parts = [];
  for (let i = 0; i < 72; i++) {
    const a = (i * 5 * Math.PI) / 180;
    const major = i % 6 === 0;
    const r1 = 48.6, r2 = major ? 46.6 : 47.6;
    const x1 = 50 + r1 * Math.cos(a), y1 = 50 + r1 * Math.sin(a);
    const x2 = 50 + r2 * Math.cos(a), y2 = 50 + r2 * Math.sin(a);
    parts.push(`<line class="tick${major ? " major" : ""}" x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`);
  }
  g.innerHTML = parts.join("");
}

/* ---------- settings ---------- */
function showSettings(tau, minArea) {
  els.tau.value = tau;
  els.tauOut.textContent = tau;
  els.minarea.value = minArea;
}

function saveSettings() {
  store("tau", els.tau.value);
  store("minarea", els.minarea.value);
}

function settingsChanged() {
  saveSettings();
  if (lastFile && document.body.dataset.state === "done") analyze(lastFile);
}

/* ---------- samples ---------- */
async function loadSamples() {
  try {
    const list = await (await fetch("/api/samples")).json();
    if (!list.length) return;
    els.samples.innerHTML = "";
    for (const s of list) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "sample";
      b.innerHTML = `<img src="${s.thumb}" alt="" loading="lazy"><span class="sample-name">${s.label}</span><span class="sample-kind">${s.kind}</span>`;
      b.setAttribute("aria-label", `Analyze sample ${s.label} (${s.kind.toLowerCase()})`);
      b.addEventListener("click", async () => {
        const blob = await (await fetch(s.url)).blob();
        analyze(new File([blob], s.file, { type: blob.type }));
      });
      els.samples.appendChild(b);
    }
    els.samplesBlock.hidden = false;
  } catch (e) { /* no samples: the upload still works */ }
}

/* ---------- analysis ---------- */
async function analyze(file) {
  lastFile = file;
  els.error.hidden = true;
  els.report.hidden = true;
  setState("loading");
  setStatus("Analyzing · usually 3–5 s");
  legend([]);
  els.steps.hidden = true;
  els.compare.hidden = true;

  // show the photo dimmed while the server works (browsers can't preview TIF)
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.single.hidden = false;
  els.single.alt = "Photo being analyzed";
  els.single.src = previewUrl;
  els.single.onerror = () => { els.single.hidden = true; els.placeholder.hidden = false; };
  els.placeholder.hidden = true;

  const body = new FormData();
  body.append("image", file, file.name);
  body.append("tau_acm", els.tau.value);
  body.append("min_area", els.minarea.value || "0");

  let res;
  try {
    res = await fetch("/api/analyze", { method: "POST", body });
  } catch (e) {
    return fail("Exora's server isn't responding. Check that the Exora window is still open, then try again.");
  }
  if (!res.ok) {
    let msg = "The analysis failed. Try another photo.";
    try { msg = (await res.json()).detail || msg; } catch (e) { /* keep default */ }
    return fail(msg);
  }
  current = await res.json();
  current.fileName = file.name;
  render();
}

function fail(message) {
  setState("error");
  setStatus("");
  els.errorText.textContent = message;
  els.error.hidden = false;
  els.single.hidden = true;
  els.placeholder.hidden = false;
  document.querySelector(".intro").hidden = false;
}

/* ---------- results ---------- */
function render() {
  const r = current, s = r.stats;
  setState("done");
  setStatus(`${r.fileName} · ${s.seconds} s`);
  document.querySelector(".intro").hidden = true;
  els.report.hidden = false;
  els.reportFile.textContent = r.fileName;

  els.verdict.classList.toggle("is-clear", !r.positive);
  if (r.positive) {
    els.verdictTitle.textContent = "Exudates found";
    els.verdictSub.textContent = `${fmt(s.marked_px)} pixels marked, above the ${fmt(s.min_area)} px needed to call it.`;
  } else {
    els.verdictTitle.textContent = "No exudates found";
    els.verdictSub.textContent = s.marked_px === 0
      ? "Nothing in this photo was marked."
      : `${fmt(s.marked_px)} pixels marked, below the ${fmt(s.min_area)} px needed to call it.`;
  }

  els.stats.innerHTML = [
    ["Marked area", `${fmt(s.marked_px)} px <small>${s.retina_pct}% of retina</small>`],
    ["Lesion regions", fmt(s.regions)],
    ["Color threshold τ", s.tau_acm],
    ["Analysis time", `${s.seconds} s`],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  const ex = r.expert;
  els.tabExpert.hidden = !ex;
  els.expert.hidden = !ex;
  if (ex) {
    els.expert.classList.toggle("is-off", !ex.agrees);
    const said = ex.has_exudates ? "exudates present" : "no exudates";
    let text = `<strong>${ex.source}:</strong> ${said}. ${ex.agrees ? "Same conclusion as Exora." : "Exora reached a different conclusion."}`;
    if (ex.has_exudates) {
      text += ` Exora covered ${ex.found_pct}% of the area the experts marked, and ${ex.correct_pct}% of what it marked lies inside their marking.`;
    }
    els.expert.innerHTML = text;
  }

  const base = r.fileName.replace(/\.[^.]+$/, "");
  els.download.href = r.result_full;
  els.download.download = `exora-${base}.jpg`;

  els.before.src = r.original;
  els.after.src = r.result;
  buildStrip();
  stepIndex = Math.min(stepIndex, r.stages.length - 1);
  show("result", true);
  if (window.matchMedia("(max-width: 980px)").matches) {
    document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function show(which, animate = false) {
  view = which;
  document.body.dataset.view = which;
  for (const [tab, name] of [[els.tabResult, "result"], [els.tabSteps, "steps"], [els.tabExpert, "expert"]]) {
    tab.setAttribute("aria-selected", String(name === which));
  }
  els.placeholder.hidden = true;
  els.steps.hidden = which !== "steps";
  els.compare.hidden = which !== "result";
  els.single.hidden = which === "result";
  els.single.onerror = null;

  if (which === "result") {
    legend([["var(--detect)", "Detected exudates"]]);
    if (animate) {
      els.compare.classList.remove("reveal");
      void els.compare.offsetWidth;      // restart the reveal animation
      els.compare.classList.add("reveal");
    }
  } else if (which === "expert") {
    els.single.src = current.expert.image;
    els.single.alt = "Detection compared with the experts' annotation";
    legend([["#00FF00", "Found by both"], ["#FFFF00", "Exora only"], ["#FF0000", "Experts only"]]);
  } else {
    legend([]);
    showStep(stepIndex);
  }
}

/* ---------- step by step ---------- */
function buildStrip() {
  els.strip.innerHTML = "";
  current.stages.forEach((st, i) => {
    const li = document.createElement("li");
    if (i === 0 || current.stages[i - 1].group !== st.group) {
      li.className = "group-start";
      li.dataset.group = st.group;
    }
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<img src="${st.image}" alt="">`;
    b.setAttribute("aria-label", `Step ${i + 1}: ${st.title}`);
    b.addEventListener("click", () => showStep(i));
    li.appendChild(b);
    els.strip.appendChild(li);
  });
  els.stepTotal.textContent = current.stages.length;
}

function showStep(i) {
  const stages = current.stages;
  stepIndex = Math.max(0, Math.min(stages.length - 1, i));
  const st = stages[stepIndex];
  els.single.src = st.image;
  els.single.alt = st.title;
  els.stepGroup.textContent = st.group;
  els.stepNum.textContent = stepIndex + 1;
  els.stepTitle.textContent = st.title;
  els.stepText.innerHTML = st.text;     // trusted text from app.py
  els.prev.disabled = stepIndex === 0;
  els.next.disabled = stepIndex === stages.length - 1;
  [...els.strip.querySelectorAll("button")].forEach((b, j) => {
    if (j === stepIndex) {
      b.setAttribute("aria-current", "step");
      const li = b.parentElement, strip = els.strip;
      if (li.offsetLeft < strip.scrollLeft) strip.scrollLeft = li.offsetLeft - 8;
      else if (li.offsetLeft + li.offsetWidth > strip.scrollLeft + strip.clientWidth)
        strip.scrollLeft = li.offsetLeft + li.offsetWidth - strip.clientWidth + 8;
    } else {
      b.removeAttribute("aria-current");
    }
  });
}

/* ---------- reset ---------- */
function reset() {
  current = null;
  lastFile = null;
  els.file.value = "";
  setState("empty");
  delete document.body.dataset.view;
  setStatus("");
  legend([]);
  els.report.hidden = true;
  els.error.hidden = true;
  els.steps.hidden = true;
  els.compare.hidden = true;
  els.single.hidden = true;
  els.placeholder.hidden = false;
  document.querySelector(".intro").hidden = false;
  els.drop.focus();
}

/* ---------- events ---------- */
els.file.addEventListener("change", () => { if (els.file.files[0]) analyze(els.file.files[0]); });

// drop a photo anywhere on the page
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => {
  e.preventDefault();
  els.drop.classList.add("is-over");
}));
["dragleave", "drop"].forEach((t) => document.addEventListener(t, (e) => {
  e.preventDefault();
  if (t === "dragleave" && e.relatedTarget) return;
  els.drop.classList.remove("is-over");
}));
document.addEventListener("drop", (e) => {
  const f = e.dataTransfer && e.dataTransfer.files[0];
  if (f) analyze(f);
});

els.again.addEventListener("click", reset);
els.errorAgain.addEventListener("click", reset);

els.tabResult.addEventListener("click", () => show("result"));
els.tabSteps.addEventListener("click", () => show("steps"));
els.tabExpert.addEventListener("click", () => show("expert"));
els.prev.addEventListener("click", () => showStep(stepIndex - 1));
els.next.addEventListener("click", () => showStep(stepIndex + 1));
document.addEventListener("keydown", (e) => {
  if (view !== "steps" || !current || e.target.matches("input, textarea")) return;
  if (e.key === "ArrowRight") showStep(stepIndex + 1);
  if (e.key === "ArrowLeft") showStep(stepIndex - 1);
});

els.splitInput.addEventListener("input", () => {
  els.compare.classList.remove("reveal");
  els.compare.style.setProperty("--split", els.splitInput.value + "%");
});

els.tau.addEventListener("input", () => { els.tauOut.textContent = els.tau.value; });
els.tau.addEventListener("change", settingsChanged);
els.minarea.addEventListener("change", settingsChanged);
els.paperValues.addEventListener("click", () => {
  showSettings(config.paper_tau_acm, els.minarea.value);
  settingsChanged();
});
els.defaults.addEventListener("click", () => {
  showSettings(config.tau_acm, config.min_area);
  settingsChanged();
});

/* ---------- start ---------- */
(async function init() {
  drawTicks();
  try { config = await (await fetch("/api/config")).json(); } catch (e) { /* keep built-in defaults */ }
  showSettings(recall("tau") || config.tau_acm, recall("minarea") || config.min_area);
  loadSamples();
})();
