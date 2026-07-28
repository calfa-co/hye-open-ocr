"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  jobId: null,
  pollTimer: null,
  mode: "ocr", // "ocr" | "compare"
  result: null, // OCR /result payload
  layoutResult: null, // compare /layout payload
  pageTexts: [],
  currentPage: 0,
  totalPages: 1,
  overlayMode: "regions", // "regions" | "lines"
  overlayOn: true, // draw detections on the page image
  candidateIndex: 0,
  previewConf: 0.5, // compare-mode client-side confidence filter
  zoom: 1, // image zoom in the results pane (1 = fit to pane)
};

const ZOOM_MIN = 1;
const ZOOM_MAX = 5;
const ZOOM_STEP = 0.25;

/* ---------- helpers ---------- */

function showError(message) {
  $("error-text").textContent = message;
  $("error-banner").classList.remove("hidden");
}

function hideError() {
  $("error-banner").classList.add("hidden");
}

async function apiError(response) {
  try {
    const body = await response.json();
    return body.detail?.message || body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

function show(section) {
  $("empty-state").classList.toggle("hidden", section !== "upload");
  $("progress-card").classList.toggle("hidden", section !== "progress");
  $("results-card").classList.toggle("hidden", section !== "results");
  // leaving any workspace section also closes the docs overlay
  if (section) hideDocs();
}

/* ---------- docs panel ---------- */

function showDocs() {
  state.priorSection = currentSection();
  $("empty-state").classList.add("hidden");
  $("progress-card").classList.add("hidden");
  $("results-card").classList.add("hidden");
  $("docs-card").classList.remove("hidden");
  $("docs-toggle").classList.add("active");
}

function hideDocs() {
  $("docs-card").classList.add("hidden");
  $("docs-toggle").classList.remove("active");
}

function currentSection() {
  if (!$("results-card").classList.contains("hidden")) return "results";
  if (!$("progress-card").classList.contains("hidden")) return "progress";
  return "upload";
}

function closeDocs() {
  $("docs-card").classList.add("hidden");
  $("docs-toggle").classList.remove("active");
  show(state.priorSection || "upload");
}

/* ---------- upload ---------- */

function initDropzone() {
  const dropzone = $("dropzone");
  const input = $("file-input");

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") input.click();
  });
  input.addEventListener("change", () => selectFile(input.files[0]));

  for (const eventName of ["dragover", "dragenter"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  }
  dropzone.addEventListener("drop", (event) =>
    selectFile(event.dataTransfer.files[0])
  );
}

function selectFile(file) {
  if (!file) return;
  hideError();
  const okTypes = /\.(png|jpe?g|tiff?|pdf)$/i;
  if (!okTypes.test(file.name)) {
    showError("Unsupported file type. Use PNG, JPEG, TIFF or PDF.");
    return;
  }
  if (file.size > 30 * 1024 * 1024) {
    showError("File is larger than 30 MB.");
    return;
  }
  state.file = file;
  $("file-name").textContent = `${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
  $("file-chip").classList.remove("hidden");
}

function clearFile() {
  state.file = null;
  $("file-input").value = "";
  $("file-chip").classList.add("hidden");
}

// current radio value for a card group ("layout" | "engine")
function radioValue(group) {
  const checked = document.querySelector(`input[name="${group}"]:checked`);
  return checked ? checked.value : "";
}

// the confidence slider lives inside the *selected* layout card
function selectedLayoutConf() {
  const card = document.querySelector('#layout-cards .opt-card.selected');
  const range = card && card.querySelector('.conf-range');
  return range ? range.value : "";
}

// reading-order strategy (paddle card only; the server maps yolo -> xycut)
function selectedReadingOrder() {
  const active = document.querySelector("#reading-order-row .seg-btn.active");
  return active ? active.dataset.order : "";
}

async function submitJob(mode = "ocr") {
  if (!state.file) {
    showError("Choose a file first.");
    return;
  }
  hideError();

  state.mode = mode === "compare" ? "compare" : "ocr";
  $("mode-select").value = state.mode;

  const form = new FormData();
  form.append("file", state.file);
  form.append("mode", state.mode);
  form.append("recognizer", radioValue("engine") || "tesseract");
  form.append("layout", radioValue("layout") || "");
  form.append("conf", selectedLayoutConf());
  form.append("reading_order", selectedReadingOrder());

  $("run-btn").disabled = true;
  $("compare-btn").disabled = true;
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await apiError(response));
    const body = await response.json();
    state.jobId = body.job_id;
    state.totalPages = body.total_pages;
    show("progress");
    setProgress(null, body.queue_position);
    state.pollTimer = setInterval(pollJob, 1500);
  } catch (error) {
    showError(`Upload failed: ${error.message}`);
  } finally {
    $("run-btn").disabled = false;
    $("compare-btn").disabled = false;
  }
}

/* ---------- progress ---------- */

function setProgress(progress, queuePosition) {
  const bar = $("progress-bar");
  const label = $("progress-label");
  if (progress && progress.done_pages > 0) {
    bar.classList.remove("indeterminate");
    bar.style.width = `${(100 * progress.done_pages) / progress.total_pages}%`;
    label.textContent = `Page ${Math.min(
      progress.done_pages + 1,
      progress.total_pages
    )} / ${progress.total_pages}…`;
  } else if (queuePosition > 0) {
    bar.classList.add("indeterminate");
    label.textContent = `Queued (position ${queuePosition})…`;
  } else {
    bar.classList.add("indeterminate");
    label.textContent =
      state.totalPages > 1
        ? `Processing page 1 / ${state.totalPages}…`
        : "Processing…";
  }
}

async function pollJob() {
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`);
    if (response.status === 404) {
      stopPolling();
      showError("Job expired. Please run the document again.");
      show("upload");
      return;
    }
    if (!response.ok) return; // transient error: keep polling
    const body = await response.json();
    if (body.status === "done") {
      stopPolling();
      await loadResult();
    } else if (body.status === "error") {
      stopPolling();
      showError(`Processing failed: ${body.error?.message || "unknown error"}`);
      show("upload");
    } else {
      setProgress(body.progress, body.queue_position);
    }
  } catch {
    /* network hiccup: keep polling */
  }
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

/* ---------- results ---------- */

function pageToText(page) {
  const blocks = [];
  for (const paragraph of page.paragraphs) {
    const lines = paragraph.lines
      .map((line) => line.text)
      .filter((text) => text.length > 0);
    if (lines.length) blocks.push(lines.join("\n"));
  }
  return blocks.join("\n\n");
}

// compare mode has no transcription and no per-line overlay: hide the text
// pane + overlay toggle, reveal the candidate selector, and disable exports.
function setCompareUI(on) {
  $("overlay-mode-row").classList.toggle("hidden", on);
  $("candidate-row").classList.toggle("hidden", !on);
  $("text-pane").classList.toggle("hidden", on);
  setExportsEnabled(!on);
}

// exports are only meaningful after an OCR run produced downloadable files.
function setExportsEnabled(on) {
  document
    .querySelector(".export-list")
    .classList.toggle("disabled", !on);
  $("export-block").querySelector(".export-hint").classList.toggle("hidden", on);
}

async function loadResult() {
  if (state.mode === "compare") return loadCompareResult();
  try {
    const response = await fetch(`/api/jobs/${state.jobId}/result`);
    if (!response.ok) throw new Error(await apiError(response));
    state.result = await response.json();
    state.pageTexts = state.result.pages.map(pageToText);
    state.currentPage = 0;
    state.totalPages = state.result.pages.length;

    for (const link of document.querySelectorAll(".dl")) {
      link.href = `/api/jobs/${state.jobId}/download/${link.dataset.fmt}`;
    }

    show("results");
    setCompareUI(false);
    renderPage();
  } catch (error) {
    showError(`Could not load the result: ${error.message}`);
    show("upload");
  }
}

async function loadCompareResult() {
  try {
    const response = await fetch(`/api/jobs/${state.jobId}/layout`);
    if (!response.ok) throw new Error(await apiError(response));
    state.layoutResult = await response.json();
    state.currentPage = 0;
    state.totalPages = state.layoutResult.pages.length || 1;
    state.candidateIndex = 0;

    show("results");
    setCompareUI(true);
    renderPage();
  } catch (error) {
    showError(`Could not load the comparison: ${error.message}`);
    show("upload");
  }
}

function currentPageData() {
  if (state.mode === "compare") {
    return state.layoutResult?.pages[state.currentPage];
  }
  return state.result?.pages[state.currentPage];
}

function populateCandidates() {
  const page = currentPageData();
  const select = $("candidate-select");
  select.innerHTML = "";
  const candidates = (page && page.candidates) || [];
  candidates.forEach((cand, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = cand.error
      ? `${cand.detector} — unavailable`
      : `${cand.name} · ${cand.regions.length} regions`;
    option.disabled = Boolean(cand.error);
    select.appendChild(option);
  });
  const firstOk = candidates.findIndex((c) => !c.error);
  state.candidateIndex = firstOk >= 0 ? firstOk : 0;
  select.value = String(state.candidateIndex);
}

function renderPage() {
  $("page-indicator").textContent = `${state.currentPage + 1}/${state.totalPages}`;
  $("prev-page").disabled = state.currentPage === 0;
  $("next-page").disabled = state.currentPage === state.totalPages - 1;
  if (state.mode === "compare") {
    populateCandidates();
    renderOverlay();
    return;
  }
  $("text-output").textContent =
    state.pageTexts[state.currentPage] || "(no text recognized)";
  renderOverlay();
}

function renderOverlay() {
  const img = $("page-img");
  const wanted = `/api/jobs/${state.jobId}/pages/${state.currentPage}/image`;
  if (img.getAttribute("src") !== wanted) {
    // re-apply the current zoom width to the freshly loaded image, then draw
    img.onload = applyZoom;
    img.src = wanted;
  } else {
    drawOverlay();
  }
}

// region colours by layout label (mirrors armenian_ocr/visualize.py)
const LABEL_COLORS = {
  title: "#d32f2f",
  text: "#2e7d32",
  caption: "#1565c0",
  table: "#ef6c00",
  formula: "#7b1fa2",
  figure: "#787878",
  abandon: "#969696",
  header: "#969696",
  footer: "#969696",
  page: "#00838f",
};

function colorFor(label) {
  const l = (label || "").toLowerCase();
  for (const key of Object.keys(LABEL_COLORS)) {
    if (l.includes(key)) return LABEL_COLORS[key];
  }
  return LABEL_COLORS.text;
}

function overlayContext() {
  const img = $("page-img");
  const canvas = $("page-canvas");
  const page = currentPageData();
  if (!page || !img.clientWidth) return null;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  // the image is centred inside a padded, scrollable wrap; pin the overlay
  // canvas to the image's own position so boxes line up exactly.
  canvas.style.left = `${img.offsetLeft}px`;
  canvas.style.top = `${img.offsetTop}px`;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  return { context, page, scale: img.clientWidth / page.width };
}

// image zoom: at zoom=1 the image fits the pane (width 100%); zooming widens
// it and the padded .overlay-wrap scrolls. The canvas re-reads img.clientWidth
// in overlayContext(), so the overlay follows automatically on redraw.
function applyZoom() {
  const img = $("page-img");
  img.style.width = `${state.zoom * 100}%`;
  img.style.maxWidth = state.zoom > 1 ? "none" : "100%";
  // when zoomed past the pane, drop the auto-centering so scrolling reaches
  // both edges instead of clipping the left side.
  img.style.margin = state.zoom > 1 ? "0" : "0 auto";
  const out = $("zoom-out-label");
  if (out) out.textContent = `${Math.round(state.zoom * 100)}%`;
  $("zoom-in").disabled = state.zoom >= ZOOM_MAX;
  $("zoom-dec").disabled = state.zoom <= ZOOM_MIN;
  drawOverlay();
}

function setZoom(value) {
  state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
  applyZoom();
}

function resetZoom() {
  state.zoom = 1;
  applyZoom();
}

function drawOverlay() {
  // the "Detections" switch hides the overlay: overlayContext() clears the
  // canvas, and we skip drawing anything back onto it.
  if (!state.overlayOn) {
    overlayContext();
    return;
  }
  if (state.mode === "compare") return drawCandidate();
  if (state.overlayMode === "lines") drawLines();
  else drawRegions();
}

// shared: draw region boxes + reading-order path + numbered badges from a
// list of {box:[x1,y1,x2,y2], label}. Used for OCR paragraphs and compare
// candidates alike.
function drawRegionOverlay(context, regions, scale) {
  const centers = [];
  regions.forEach((region) => {
    const [x1, y1, x2, y2] = region.box;
    const color = colorFor(region.label);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.fillStyle = color + "1a"; // ~10% alpha
    // a tight layout polygon (PP-DocLayoutV3) hugs skewed/curved blocks;
    // fall back to the axis-aligned box when there is none.
    if (region.poly && region.poly.length >= 3) {
      context.beginPath();
      region.poly.forEach(([px, py], i) =>
        i ? context.lineTo(px * scale, py * scale) : context.moveTo(px * scale, py * scale)
      );
      context.closePath();
      context.fill();
      context.stroke();
    } else {
      context.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      context.fillRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    }
    centers.push([((x1 + x2) / 2) * scale, ((y1 + y2) / 2) * scale]);
  });

  context.strokeStyle = "rgba(30, 30, 30, 0.85)";
  context.lineWidth = 1.5;
  context.beginPath();
  centers.forEach(([cx, cy], i) => (i ? context.lineTo(cx, cy) : context.moveTo(cx, cy)));
  context.stroke();

  regions.forEach((region, index) => {
    const [x1, y1] = region.box;
    const [bx, by] = [x1 * scale + 11, y1 * scale + 11];
    context.beginPath();
    context.arc(bx, by, 11, 0, 2 * Math.PI);
    context.fillStyle = colorFor(region.label);
    context.fill();
    context.fillStyle = "#fff";
    context.font = "bold 12px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(index + 1), bx, by);
  });
}

// draw the detected text-line boxes (this is the `det` output — for Paddle a
// line is one box; for Tesseract it groups its words). This is where you see
// the line detection, whether it came from a block region or the full page.
function drawLines() {
  const ctx = overlayContext();
  if (!ctx) return;
  const { context, page, scale } = ctx;
  context.strokeStyle = "rgba(176, 74, 60, 0.85)";
  context.lineWidth = 1.2;
  for (const paragraph of page.paragraphs) {
    for (const line of paragraph.lines) {
      // tight det polygon hugs skewed lines; else the axis-aligned box
      if (line.poly && line.poly.length >= 3) {
        context.beginPath();
        line.poly.forEach(([px, py], i) =>
          i ? context.lineTo(px * scale, py * scale) : context.moveTo(px * scale, py * scale)
        );
        context.closePath();
        context.stroke();
      } else {
        const [x1, y1, x2, y2] = line.box;
        context.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      }
    }
  }
}

function drawRegions() {
  const ctx = overlayContext();
  if (!ctx) return;
  const { context, page, scale } = ctx;
  drawRegionOverlay(context, page.paragraphs, scale);
}

function drawCandidate() {
  const ctx = overlayContext();
  if (!ctx) return;
  const { context, page, scale } = ctx;
  const candidate = page.candidates?.[state.candidateIndex];
  if (!candidate || candidate.error) return;
  // regions were detected at a low threshold; keep those at/above the preview
  // confidence (regions without a score are always kept). This is the live
  // "which regions would a higher threshold keep?" preview.
  const regions = filterByConfidence(candidate.regions);
  updateConfCount(regions.length, candidate.regions.length);
  drawRegionOverlay(context, regions, scale);
}

// keep regions whose detector score is >= the preview confidence slider
function filterByConfidence(regions) {
  const threshold = state.previewConf;
  return regions.filter(
    (region) => region.score == null || region.score >= threshold
  );
}

function updateConfCount(shown, total) {
  const out = $("preview-conf-count");
  if (out) out.textContent = total ? `${shown}/${total} regions` : "";
}

/* ---------- reset ---------- */

async function newDocument() {
  stopPolling();
  if (state.jobId) {
    fetch(`/api/jobs/${state.jobId}`, { method: "DELETE" }).catch(() => {});
  }
  state.jobId = null;
  state.result = null;
  state.layoutResult = null;
  state.zoom = 1;
  clearFile();
  $("page-img").removeAttribute("src");
  $("text-pane").classList.remove("hidden");
  setExportsEnabled(false);
  hideError();
  show("upload");
}

/* ---------- health (cold start notice) ---------- */

function applyEngineAvailability(engines, layouts) {
  if (engines) {
    const available = engines.paddle && engines.paddle.available;
    const card = document.querySelector("[data-paddle-engine]");
    const input = card && card.querySelector('input[value="paddle"]');
    const badge = card && card.querySelector(".opt-badge");
    if (input) input.disabled = !available;
    if (card) card.classList.toggle("disabled", !available);
    if (badge) {
      badge.textContent = available ? "Paddle" : "Paddle · unavailable";
      badge.classList.toggle("muted", true);
    }
    // update the tile description depending on availability
    const desc = card && card.querySelector(".opt-desc");
    if (desc) {
      desc.textContent = available
        ? "PP-OCRv6 model · Classical, Western & Eastern Armenian"
        : "Install the paddle extra to enable (pip install '.[paddle]')";
    }
  }
  if (layouts) {
    const available = layouts.paddle && layouts.paddle.available;
    const card = document.querySelector(
      '#layout-cards .opt-card:has(input[value="paddle"])'
    );
    const input = card && card.querySelector('input[value="paddle"]');
    if (input) input.disabled = !available;
    if (card) {
      card.classList.toggle("disabled", !available);
      const desc = card.querySelector(".opt-desc");
      if (desc && !available) desc.textContent = "Install the paddle extra to enable";
    }
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const body = await response.json();
    $("warming").classList.toggle("hidden", body.models_loaded);
    applyEngineAvailability(body.engines, body.layouts);
    if (!body.models_loaded) setTimeout(checkHealth, 3000);
  } catch {
    setTimeout(checkHealth, 5000);
  }
}

/* ---------- wiring ---------- */

initDropzone();
checkHealth();
setExportsEnabled(false);
show("upload");
$("file-clear").addEventListener("click", clearFile);

// radio-cards: reflect the checked option as .selected (which also reveals
// that card's inline tuning control)
function syncCards(group) {
  for (const card of document.querySelectorAll(`#${group}-cards .opt-card`)) {
    const input = card.querySelector("input");
    card.classList.toggle("selected", input.checked);
  }
}
for (const group of ["layout", "engine"]) {
  const container = $(`${group}-cards`);
  container.addEventListener("change", () => syncCards(group));
}

// live confidence read-out per layout card (the preview-conf slider in the
// compare toolbar is wired separately and is not inside an .opt-tune)
for (const range of document.querySelectorAll("#layout-cards .conf-range")) {
  const tune = range.closest(".opt-tune");
  const out = tune && tune.querySelector("[data-conf-out]");
  if (!out) continue;
  range.addEventListener("input", () => {
    out.textContent = Number(range.value).toFixed(2);
  });
}

// reading-order segmented control (paddle card). It lives inside the card's
// <label>, so stop the click from re-triggering the radio.
for (const button of document.querySelectorAll("#reading-order-row .seg-btn")) {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    for (const other of document.querySelectorAll("#reading-order-row .seg-btn")) {
      other.classList.toggle("active", other === button);
    }
  });
}

// docs panel toggle
$("docs-toggle").addEventListener("click", showDocs);
$("docs-close").addEventListener("click", closeDocs);
$("candidate-select").addEventListener("change", () => {
  state.candidateIndex = parseInt($("candidate-select").value, 10) || 0;
  drawOverlay();
});
$("preview-conf").addEventListener("input", (event) => {
  state.previewConf = Number(event.target.value);
  $("preview-conf-out").textContent = state.previewConf.toFixed(2);
  drawOverlay();
});
$("run-btn").addEventListener("click", () => submitJob("ocr"));
$("compare-btn").addEventListener("click", () => submitJob("compare"));
$("overlay-toggle").addEventListener("change", (event) => {
  state.overlayOn = event.target.checked;
  // the Regions/Lines switch only applies to OCR results, and only when the
  // overlay is shown (compare mode never shows it).
  const showModes = state.overlayOn && state.mode !== "compare";
  $("overlay-mode-row").classList.toggle("hidden", !showModes);
  drawOverlay();
});
$("error-close").addEventListener("click", hideError);
$("cancel-link").addEventListener("click", (event) => {
  event.preventDefault();
  newDocument();
});
$("new-doc").addEventListener("click", newDocument);
$("copy-btn").addEventListener("click", () => {
  navigator.clipboard.writeText($("text-output").textContent);
});
for (const button of document.querySelectorAll("#overlay-mode-row .seg-btn")) {
  button.addEventListener("click", () => {
    state.overlayMode = button.dataset.mode;
    for (const other of document.querySelectorAll("#overlay-mode-row .seg-btn")) {
      other.classList.toggle("active", other === button);
    }
    drawOverlay();
  });
}
$("zoom-in").addEventListener("click", () => setZoom(state.zoom + ZOOM_STEP));
$("zoom-dec").addEventListener("click", () => setZoom(state.zoom - ZOOM_STEP));
$("zoom-out-label").addEventListener("click", resetZoom);
$("prev-page").addEventListener("click", () => {
  if (state.currentPage > 0) {
    state.currentPage -= 1;
    renderPage();
  }
});
$("next-page").addEventListener("click", () => {
  if (state.currentPage < state.totalPages - 1) {
    state.currentPage += 1;
    renderPage();
  }
});
window.addEventListener("resize", drawOverlay);
