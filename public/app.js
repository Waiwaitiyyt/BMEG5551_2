// Implant Locator workstation — BMEG5552.
//
// Three panes, one shared study list:
//   queue    a client-side session queue; nothing is persisted server-side
//   viewer   the active study with the model's box drawn over it
//   findings the parsed response plus the knobs that shape the next run
//
// API contract
// ------------
//   GET  {API}/health
//     -> { status, model_path, model_name, classes, conf_threshold,
//          iou_threshold, imgsz, device }
//
//   POST {API}/predict     multipart/form-data
//     file  the image           (required)
//     conf  0 < conf <= 1       (optional, defaults to the server's setting)
//     iou   0 < iou  <= 1       (optional)
//     imgsz 64..1536, /32       (optional)
//     -> { detections: [{ label, confidence, box: [x1,y1,x2,y2] }],
//          image_width, image_height,
//          conf_threshold, iou_threshold, imgsz, model, classes, inference_ms }
//
//   box coordinates are pixels in the original uploaded image (top-left /
//   bottom-right), which is what drawOverlays() positions against.
//
// Served by the Express gateway (server/ts, :3000) the calls go to /api on the
// same origin and Express proxies them to FastAPI. Opened straight from disk
// there is no origin to inherit, so it falls back to FastAPI on :8000 — that
// path needs the FastAPI CORS settings to allow it.

const IS_FILE = window.location.protocol === "file:";
const API_BASE = IS_FILE ? "http://localhost:8000" : "/api";
const PREDICT_ENDPOINT = `${API_BASE}/predict`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const GATEWAY_HEALTH = IS_FILE ? null : "/healthz";
const HEALTH_INTERVAL_MS = 15000;

const $ = (id) => document.getElementById(id);

const el = {
    chipModel: $("chip-model"),
    dotApi: $("dot-api"), chipApi: $("chip-api"),
    dotWeb: $("dot-web"), chipWeb: $("chip-web"),

    queueList: $("queue-list"),
    queueBlank: $("queue-blank"),
    queueCount: $("queue-count"),
    queueFilter: $("queue-filter"),
    queueAdd: $("queue-add"),
    fileInput: $("file-input"),

    stage: $("viewer-stage"),
    stageEmpty: $("stage-empty"),
    stageFrames: $("stage-frames"),
    frameLeft: $("frame-left"),
    frameInner: $("frame-inner"),
    viewerImage: $("viewer-image"),
    viewerOverlays: $("viewer-overlays"),
    leftInner: $("left-inner"),
    leftImage: $("left-image"),
    leftOverlays: $("left-overlays"),
    leftTag: $("left-tag"),
    currentTag: $("current-tag"),
    overlaySeg: $("overlay-seg"),
    compareBtn: $("compare-btn"),
    comparePop: $("compare-pop"),
    compareHint: $("compare-hint"),
    pickLeft: $("pick-left"),
    pickRight: $("pick-right"),
    compareApply: $("compare-apply"),
    compareExit: $("compare-exit"),
    zoomReadout: $("zoom-readout"),

    cornerTl: $("corner-tl"), cornerTr: $("corner-tr"),
    cornerBl: $("corner-bl"), cornerBr: $("corner-br"),

    steps: $("steps"),
    runBtn: $("run-btn"),
    clearBtn: $("clear-btn"),

    statusPill: $("status-pill"),
    detCards: $("det-cards"),
    findingsNote: $("findings-note"),

    confRange: $("conf-range"), confOut: $("conf-out"),
    confTick: $("conf-tick"), confHint: $("conf-hint"),
    iouRange: $("iou-range"), iouOut: $("iou-out"),
    windowRange: $("window-range"), levelRange: $("level-range"), wlOut: $("wl-out"),

    metaCkpt: $("meta-ckpt"), metaImgsz: $("meta-imgsz"),
    metaClasses: $("meta-classes"), metaLatency: $("meta-latency"),
    metaEndpoint: $("meta-endpoint"),

    rawBlock: $("raw-block"), rawReq: $("raw-req"),
    rawStatus: $("raw-status"), rawBody: $("raw-body"),
    exportBtn: $("export-btn"),
};

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

/** @type {Array<Study>} studies in queue order (newest last) */
const studies = [];
let activeId = null;
let compareLeftId = null;    // left-hand study of the comparison; the right
                             // frame is always the active study
let nextId = 1;

// What the compare picker currently has selected. Seeded from the live
// comparison every time the popover opens, applied only on "Show comparison".
const pick = { left: null, right: null };

const view = {
    tool: "wl",              // wl | zoom | pan
    overlay: "box",          // box | heat | both | off
    zoom: 1,
    panX: 0,
    panY: 0,
    compare: false,
};

// Set once the reader moves a threshold slider; from then on the sliders win
// over whatever /health reports as the server-side default.
let touchedConf = false;
let touchedIou = false;

const server = {
    online: false,
    imgsz: 640,
    classes: [],
    modelName: null,
    modelPath: null,
};

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmt2 = (v) => Number(v).toFixed(2);
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

function activeStudy() {
    return studies.find((s) => s.id === activeId) ?? null;
}

function leftStudy() {
    return studies.find((s) => s.id === compareLeftId) ?? null;
}

function studyById(id) {
    return studies.find((s) => s.id === id) ?? null;
}

/**
 * What the left side falls back to when the reader has not picked one — the
 * most recent finished study other than `otherId`, or failing that the most
 * recent study of any state.
 */
function defaultLeftCandidate(otherId) {
    for (let i = studies.length - 1; i >= 0; i -= 1) {
        const s = studies[i];
        if (s.id !== otherId && s.status === "done") return s;
    }
    for (let i = studies.length - 1; i >= 0; i -= 1) {
        if (studies[i].id !== otherId) return studies[i];
    }
    return null;
}

/** Keep the live comparison legal: two different studies, both still queued. */
function reconcileCompare() {
    if (!view.compare) return;
    if (compareLeftId === activeId) compareLeftId = null;
    if (!leftStudy()) compareLeftId = defaultLeftCandidate(activeId)?.id ?? null;
    if (compareLeftId === null || activeId === null) view.compare = false;
}

function setText(node, value) {
    if (node.textContent !== value) node.textContent = value;
}

// ---------------------------------------------------------------------
// Health — drives the two service chips in the top bar
// ---------------------------------------------------------------------

async function pollHealth() {
    try {
        const res = await fetch(HEALTH_ENDPOINT, { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();

        server.online = data.status === "ok";
        server.imgsz = data.imgsz ?? server.imgsz;
        server.classes = Array.isArray(data.classes) ? data.classes : [];
        server.modelPath = data.model_path ?? null;
        server.modelName = data.model_name ?? basename(data.model_path);

        el.dotApi.className = `dot ${server.online ? "is-up" : "is-wait"}`;
        setText(el.chipApi, `FastAPI ${server.online ? "ready" : "loading"}`);
        setText(el.chipModel, `${server.modelName ?? "—"} · YOLO`);

        // Only adopt the server's thresholds until the reader touches a slider;
        // after that the sliders are the source of truth for the next run.
        if (!touchedConf && typeof data.conf_threshold === "number") {
            el.confRange.value = String(data.conf_threshold);
        }
        if (!touchedIou && typeof data.iou_threshold === "number") {
            el.iouRange.value = String(data.iou_threshold);
        }
        renderControls();
        renderMeta();
    } catch {
        server.online = false;
        el.dotApi.className = "dot is-down";
        setText(el.chipApi, "FastAPI offline");
    }

    if (GATEWAY_HEALTH) {
        try {
            const res = await fetch(GATEWAY_HEALTH, { cache: "no-store" });
            el.dotWeb.className = `dot ${res.ok ? "is-up" : "is-down"}`;
            setText(el.chipWeb, res.ok ? "Express ready" : "Express error");
        } catch {
            el.dotWeb.className = "dot is-down";
            setText(el.chipWeb, "Express offline");
        }
    } else {
        el.dotWeb.className = "dot";
        setText(el.chipWeb, "direct (file://)");
    }
}

function basename(p) {
    if (!p) return null;
    const parts = String(p).split(/[\\/]/);
    return parts[parts.length - 1] || null;
}

/** "…/server/py/weights/best.pt" -> "weights/best.pt" — the checkpoint's
 *  absolute path is too long for the metadata column and says nothing useful. */
function shortPath(p) {
    if (!p) return null;
    const parts = String(p).split(/[\\/]/).filter(Boolean);
    return parts.slice(-2).join("/") || null;
}

// ---------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------

function addFiles(fileList) {
    const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (files.length === 0) {
        noteError("Those files don't look like images. Use PNG, JPEG, or WEBP.");
        return;
    }

    let firstAdded = null;
    files.forEach((file) => {
        const study = {
            id: nextId++,
            file,
            url: URL.createObjectURL(file),
            name: file.name,
            width: 0,
            height: 0,
            kind: (file.type.split("/")[1] || "img").toUpperCase(),
            status: "queued",
            detections: null,
            response: null,
            error: null,
            latencyMs: null,
        };
        studies.push(study);
        if (firstAdded === null) firstAdded = study.id;

        // The natural size is only known once the browser has decoded it.
        const probe = new Image();
        probe.onload = () => {
            study.width = probe.naturalWidth;
            study.height = probe.naturalHeight;
            renderQueue();
            if (study.id === activeId) { renderViewer(); renderCorners(); }
        };
        probe.src = study.url;
    });

    selectStudy(firstAdded);
    renderAll();
}

function selectStudy(id) {
    if (activeId === id) return;
    activeId = id;
    resetTransform();

    // The right frame follows the active study, so a click in the queue swaps
    // the right side of an open comparison and only the left side is kept.
    reconcileCompare();

    renderAll();
}

function removeStudy(id) {
    const idx = studies.findIndex((s) => s.id === id);
    if (idx === -1) return;
    URL.revokeObjectURL(studies[idx].url);
    studies.splice(idx, 1);

    if (activeId === id) {
        activeId = studies[Math.min(idx, studies.length - 1)]?.id ?? null;
        resetTransform();
    }
    if (compareLeftId === id) compareLeftId = null;
    reconcileCompare();

    renderAll();
}

function clearActive() {
    if (activeId !== null) removeStudy(activeId);
}

function renderQueue() {
    const needle = el.queueFilter.value.trim().toLowerCase();
    const visible = needle
        ? studies.filter((s) => s.name.toLowerCase().includes(needle))
        : studies;

    setText(el.queueCount, String(studies.length));
    el.queueList.replaceChildren();

    visible.forEach((study) => {
        const li = document.createElement("li");

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `queue-item${study.id === activeId ? " is-active" : ""}`;
        btn.setAttribute("aria-current", study.id === activeId ? "true" : "false");
        btn.addEventListener("click", () => selectStudy(study.id));

        const thumb = document.createElement("span");
        thumb.className = "queue-thumb";
        const img = document.createElement("img");
        img.src = study.url;
        img.alt = "";
        thumb.appendChild(img);

        const body = document.createElement("span");
        body.className = "queue-body";

        const name = document.createElement("span");
        name.className = "queue-name";
        name.textContent = study.name;
        name.title = study.name;

        const sub = document.createElement("span");
        sub.className = "queue-sub";
        sub.textContent = study.width
            ? `${study.width} × ${study.height} · ${study.kind}`
            : study.kind;

        body.append(name, sub);
        btn.append(thumb, body, queueMarker(study));
        li.appendChild(btn);
        el.queueList.appendChild(li);
    });

    el.queueBlank.hidden = studies.length > 0;
    if (studies.length > 0 && visible.length === 0) {
        el.queueBlank.hidden = false;
        el.queueBlank.textContent = `No study matches "${el.queueFilter.value.trim()}".`;
    } else {
        el.queueBlank.textContent =
            "No studies yet. Drop X-ray images below, or anywhere on the window, to build a queue.";
    }
}

/** Confidence figure for a hit, a state pill for everything else. */
function queueMarker(study) {
    if (study.status === "done" && study.detections?.length) {
        const span = document.createElement("span");
        span.className = "queue-conf";
        span.textContent = pct(study.detections[0].confidence);
        return span;
    }

    const pill = document.createElement("span");
    pill.className = "queue-pill";
    if (study.status === "running") { pill.classList.add("is-running"); pill.textContent = "running"; }
    else if (study.status === "error") { pill.classList.add("is-error"); pill.textContent = "error"; }
    else if (study.status === "done") { pill.classList.add("is-none"); pill.textContent = "none"; }
    else pill.textContent = "queued";
    return pill;
}

// ---------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------

function renderViewer() {
    const study = activeStudy();

    el.stageEmpty.hidden = !!study;
    el.stageFrames.hidden = !study;
    if (!study) {
        el.viewerImage.removeAttribute("src");
        el.viewerOverlays.replaceChildren();
        return;
    }

    if (el.viewerImage.getAttribute("src") !== study.url) {
        el.viewerImage.src = study.url;
        el.viewerImage.alt = `X-ray under review: ${study.name}`;
    }

    const left = view.compare ? leftStudy() : null;
    el.stageFrames.classList.toggle("is-compare", !!left);
    el.stage.classList.toggle("is-compare", !!left);
    el.frameLeft.hidden = !left;
    el.currentTag.hidden = !left;

    if (left) {
        if (el.leftImage.getAttribute("src") !== left.url) el.leftImage.src = left.url;
        el.leftImage.alt = `Study A of the comparison: ${left.name}`;
        el.leftTag.innerHTML = "";
        el.leftTag.append(bold("A"), br(), text(left.name), br(),
            text(`${left.width} × ${left.height} px`));
        el.currentTag.innerHTML = "";
        el.currentTag.append(bold("B"), br(), text(study.name), br(),
            text(`${study.width} × ${study.height} px`));
    }

    applyWindowLevel();
    applyTransform();
    fitFrames();
    drawOverlays(el.viewerOverlays, study, false);
    drawOverlays(el.leftOverlays, left, true);
}

const text = (t) => document.createTextNode(t);
const br = () => document.createElement("br");
function bold(t) { const b = document.createElement("b"); b.textContent = t; return b; }

/**
 * Size each .frame-inner to the contain-fit rectangle of its image inside the
 * frame. Doing it here rather than in CSS means the overlays, positioned in
 * percentages of that box, stay pinned to the right pixels at any size, and a
 * small 331 px study still fills the viewer the way the design shows it.
 */
function fitFrames() {
    fitOne(el.frameInner, activeStudy());
    if (view.compare) fitOne(el.leftInner, leftStudy());
}

function fitOne(inner, study) {
    if (!inner || !study || !study.width || !study.height) return;
    const frame = inner.parentElement;
    const style = getComputedStyle(frame);
    const availW = frame.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availH = frame.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    if (availW <= 0 || availH <= 0) return;

    const scale = Math.min(availW / study.width, availH / study.height);
    inner.style.width = `${Math.round(study.width * scale)}px`;
    inner.style.height = `${Math.round(study.height * scale)}px`;
}

/**
 * Draw the model's boxes over one frame. `study.detections[i].box` is
 * [x1, y1, x2, y2] in the original image's pixels, so every edge becomes a
 * percentage of the image box and rides along with the zoom/pan transform.
 */
function drawOverlays(container, study, secondary) {
    if (!container) return;
    container.replaceChildren();
    if (!study || view.overlay === "off") return;
    if (study.status !== "done" || !study.detections?.length) return;

    const W = study.response?.image_width || study.width;
    const H = study.response?.image_height || study.height;
    if (!W || !H) return;

    const wantBox = view.overlay === "box" || view.overlay === "both";
    const wantHeat = view.overlay === "heat" || view.overlay === "both";

    study.detections.forEach((det, i) => {
        const [x1, y1, x2, y2] = det.box;
        const rect = {
            left: `${(x1 / W) * 100}%`,
            top: `${(y1 / H) * 100}%`,
            width: `${((x2 - x1) / W) * 100}%`,
            height: `${((y2 - y1) / H) * 100}%`,
        };

        if (wantHeat) {
            const heat = document.createElement("div");
            heat.className = "det-heat";
            Object.assign(heat.style, rect);
            container.appendChild(heat);
        }

        if (!wantBox) return;

        if (secondary) {
            const dashed = document.createElement("div");
            dashed.className = "dash-box";
            Object.assign(dashed.style, rect);
            container.appendChild(dashed);
            return;
        }

        const box = document.createElement("div");
        box.className = `det-box${i > 0 ? " is-secondary" : ""}`;
        Object.assign(box.style, rect);
        for (let c = 0; c < 4; c += 1) box.appendChild(document.createElement("i"));

        const label = document.createElement("span");
        label.className = "det-label";
        label.textContent = `${det.label ?? "implant"} · ${fmt2(det.confidence)}`;
        box.appendChild(label);

        container.appendChild(box);
    });
}

// --- window / level, zoom, pan ----------------------------------------

/**
 * Window/level as a CSS filter. With `brightness(b) contrast(c)` the output is
 * (in·b − 0.5)·c + 0.5, so contrast = 255/window sets the slope and
 * brightness = 127.5/level puts the chosen level at mid-grey. Display only —
 * the bytes posted to the model are always the untouched original file.
 */
function applyWindowLevel() {
    const win = Number(el.windowRange.value);
    const level = Math.max(1, Number(el.levelRange.value));
    const filter = `brightness(${(127.5 / level).toFixed(3)}) contrast(${(255 / win).toFixed(3)})`;
    el.viewerImage.style.filter = filter;
    el.leftImage.style.filter = filter;
}

function applyTransform() {
    const t = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
    el.frameInner.style.transform = t;
    el.leftInner.style.transform = t;
    setText(el.zoomReadout, `${Math.round(view.zoom * 100)}%`);
}

function resetTransform() {
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
}

function setTool(tool) {
    view.tool = tool;
    ["wl", "zoom", "pan"].forEach((t) => el.stage.classList.toggle(`tool-${t}`, t === tool));
    document.querySelectorAll(".tool[data-tool]").forEach((btn) => {
        const on = btn.dataset.tool === tool;
        btn.classList.toggle("is-on", on);
        btn.setAttribute("aria-pressed", String(on));
    });
}

function setOverlay(mode) {
    view.overlay = mode;
    el.overlaySeg.querySelectorAll("button").forEach((btn) => {
        const on = btn.dataset.overlay === mode;
        btn.classList.toggle("is-on", on);
        btn.setAttribute("aria-pressed", String(on));
    });
    drawOverlays(el.viewerOverlays, activeStudy(), false);
    drawOverlays(el.leftOverlays, view.compare ? leftStudy() : null, true);
    renderControls();
}

// ---------------------------------------------------------------------
// Compare picker — the reader chooses the two studies themselves
// ---------------------------------------------------------------------
//
// The right frame is always the active study, so applying a pair sets
// `compareLeftId` for the left frame and makes the right pick active.

function openPicker() {
    if (studies.length < 2) return;
    pick.right = activeStudy()?.id ?? studies[studies.length - 1].id;
    pick.left = (view.compare ? compareLeftId : null) ?? defaultLeftCandidate(pick.right)?.id ?? null;
    if (pick.left === pick.right) pick.left = null;

    el.comparePop.hidden = false;
    el.compareBtn.setAttribute("aria-expanded", "true");
    renderPicker();
}

function closePicker() {
    if (el.comparePop.hidden) return;
    el.comparePop.hidden = true;
    el.compareBtn.setAttribute("aria-expanded", "false");
}

function togglePicker() {
    if (el.comparePop.hidden) openPicker(); else closePicker();
}

function renderPicker() {
    renderPickList(el.pickLeft, "left");
    renderPickList(el.pickRight, "right");

    const left = studyById(pick.left);
    const right = studyById(pick.right);
    const ready = !!left && !!right && left.id !== right.id;

    el.compareApply.disabled = !ready;
    el.compareExit.hidden = !view.compare;
    setText(el.compareHint, ready
        ? `A ${left.name}  ·  B ${right.name}`
        : "Pick one study for each side.");
}

function renderPickList(list, side) {
    const other = side === "left" ? pick.right : pick.left;
    list.replaceChildren();

    studies.forEach((study) => {
        const li = document.createElement("li");

        const btn = document.createElement("button");
        btn.type = "button";
        const isOther = study.id === other;
        btn.className = `pick-item${study.id === pick[side] ? " is-picked" : ""}${isOther ? " is-other" : ""}`;
        btn.setAttribute("aria-pressed", String(study.id === pick[side]));
        btn.title = isOther ? "Chosen on the other side — click to swap the two sides" : study.name;
        btn.addEventListener("click", () => {
            // A study can only sit on one side, so choosing the other side's
            // pick swaps them rather than leaving a dead row.
            if (isOther) [pick.left, pick.right] = [pick.right, pick.left];
            else pick[side] = study.id;
            renderPicker();
        });

        const thumb = document.createElement("span");
        thumb.className = "queue-thumb";
        const img = document.createElement("img");
        img.src = study.url;
        img.alt = "";
        thumb.appendChild(img);

        const body = document.createElement("span");
        body.className = "queue-body";

        const name = document.createElement("span");
        name.className = "queue-name";
        name.textContent = study.name;

        const sub = document.createElement("span");
        sub.className = "queue-sub";
        sub.textContent = study.width
            ? `${study.width} × ${study.height} · ${statusLabel(study)}`
            : statusLabel(study);

        body.append(name, sub);
        btn.append(thumb, body);
        li.appendChild(btn);
        list.appendChild(li);
    });
}

function applyCompare() {
    const left = studyById(pick.left);
    const right = studyById(pick.right);
    if (!left || !right || left.id === right.id) return;

    compareLeftId = left.id;
    if (activeId !== right.id) {
        activeId = right.id;
        resetTransform();
    }
    view.compare = true;

    closePicker();
    renderAll();
}

function exitCompare() {
    view.compare = false;
    closePicker();
    renderAll();
}

// ---------------------------------------------------------------------
// Corner annotations
// ---------------------------------------------------------------------

function renderCorners() {
    const study = activeStudy();
    if (!study) {
        [el.cornerTl, el.cornerTr, el.cornerBl, el.cornerBr].forEach((n) => setText(n, ""));
        return;
    }

    const idx = studies.indexOf(study) + 1;
    const dims = study.width ? `${study.width} × ${study.height} px · ${study.kind}` : study.kind;
    setText(el.cornerTl, `${study.name}\n${dims}`);
    setText(el.cornerTr, `STUDY ${idx} / ${studies.length}\n${statusLabel(study).toUpperCase()}`);
    setText(el.cornerBl,
        `W ${el.windowRange.value}   L ${el.levelRange.value}\nZOOM ${Math.round(view.zoom * 100)}%`);

    const r = study.response;
    const imgsz = r?.imgsz ?? server.imgsz;
    const conf = r?.conf_threshold ?? Number(el.confRange.value);
    const iou = r?.iou_threshold ?? Number(el.iouRange.value);
    setText(el.cornerBr,
        `imgsz ${imgsz} · conf ${fmt2(conf)} · iou ${fmt2(iou)}\nPOST ${PREDICT_ENDPOINT}`);
}

function statusLabel(study) {
    if (!study) return "idle";
    if (study.status === "running") return "analysing";
    if (study.status === "error") return "error";
    if (study.status === "done") return study.detections?.length ? "detected" : "no detection";
    return "queued";
}

// ---------------------------------------------------------------------
// Findings pane
// ---------------------------------------------------------------------

function renderFindings() {
    const study = activeStudy();

    // status pill
    el.statusPill.className = "status-pill";
    if (!study) setText(el.statusPill, "Idle");
    else if (study.status === "running") { el.statusPill.classList.add("is-busy"); setText(el.statusPill, "Analysing…"); }
    else if (study.status === "error") { el.statusPill.classList.add("is-bad"); setText(el.statusPill, "Error"); }
    else if (study.status === "done" && study.detections?.length) { el.statusPill.classList.add("is-ok"); setText(el.statusPill, "Detected"); }
    else if (study.status === "done") { el.statusPill.classList.add("is-bad"); setText(el.statusPill, "No detection"); }
    else setText(el.statusPill, "Queued");

    // detection cards
    el.detCards.replaceChildren();
    const dets = study?.status === "done" ? (study.detections ?? []) : [];
    dets.forEach((det) => el.detCards.appendChild(detectionCard(det)));

    // the note that stands in for cards when there is nothing to show
    el.findingsNote.className = "findings-note";
    if (dets.length) {
        el.findingsNote.hidden = true;
    } else {
        el.findingsNote.hidden = false;
        if (!study) {
            el.findingsNote.textContent =
                "Select or drop a study, then run detection to see the implant bounding box and confidence here.";
        } else if (study.status === "error") {
            el.findingsNote.classList.add("is-error");
            el.findingsNote.textContent = study.error;
        } else if (study.status === "running") {
            el.findingsNote.textContent = "Running YOLO inference on the uploaded pixels…";
        } else if (study.status === "done") {
            el.findingsNote.textContent =
                `No implant scored above the ${fmt2(study.response?.conf_threshold ?? 0)} confidence threshold. ` +
                "Lower the threshold and re-run to see weaker candidates.";
        } else {
            el.findingsNote.textContent = "Ready. Run detection to locate the implant in this study.";
        }
    }

    renderControls();
    renderMeta();
    renderRaw();

    el.exportBtn.disabled = !study?.response;
}

function detectionCard(det) {
    const [x1, y1, x2, y2] = det.box;
    const conf = det.confidence ?? 0;

    const card = document.createElement("div");
    card.className = "det-card";

    const head = document.createElement("div");
    head.className = "det-card-head";

    const name = document.createElement("span");
    name.className = "det-name";
    const swatch = document.createElement("span");
    swatch.className = "det-swatch";
    name.append(swatch, text(det.label ?? "implant"));

    const value = document.createElement("span");
    value.className = "det-conf";
    value.textContent = fmt2(conf);

    head.append(name, value);

    const bar = document.createElement("div");
    bar.className = "det-bar";
    const fill = document.createElement("span");
    fill.style.width = `${clamp(conf * 100, 0, 100)}%`;
    bar.appendChild(fill);

    const coords = document.createElement("div");
    coords.className = "det-coords";
    [["x1", x1], ["y1", y1], ["x2", x2], ["y2", y2]].forEach(([key, v]) => {
        const cell = document.createElement("div");
        const b = document.createElement("b");
        b.textContent = String(Math.round(v));
        const s = document.createElement("span");
        s.textContent = key;
        cell.append(b, s);
        coords.appendChild(cell);
    });

    card.append(head, bar, coords);
    return card;
}

function renderControls() {
    const conf = Number(el.confRange.value);
    const iou = Number(el.iouRange.value);
    setText(el.confOut, fmt2(conf));
    setText(el.iouOut, fmt2(iou));
    setText(el.wlOut, `${el.windowRange.value} / ${el.levelRange.value}`);

    paintTrack(el.confRange, "var(--color-accent-600)");
    paintTrack(el.iouRange, "var(--color-accent-700)");
    paintTrack(el.windowRange, "var(--color-accent-600)");
    paintTrack(el.levelRange, "var(--color-neutral-700)");

    // Tick showing where the current top detection sits on the threshold track.
    const top = activeStudy()?.detections?.[0]?.confidence;
    if (typeof top === "number") {
        el.confTick.hidden = false;
        el.confTick.style.left = thumbOffset(el.confRange, top);
        setText(el.confHint, `Tick marks the current detection at ${fmt2(top)}.`);
    } else {
        el.confTick.hidden = true;
        setText(el.confHint, "Sent to the model on the next run.");
    }
}

/** Fraction of the way along a range input's travel, in CSS units. */
function thumbOffset(input, value) {
    const min = Number(input.min);
    const max = Number(input.max);
    const f = clamp((value - min) / (max - min), 0, 1);
    return `calc(${f} * (100% - 13px) + 6.5px)`;
}

/**
 * Chromium has no ::-moz-range-progress, so the filled part of the track is
 * painted into the track's own background from here.
 */
function paintTrack(input, color) {
    const min = Number(input.min);
    const max = Number(input.max);
    const f = clamp((Number(input.value) - min) / (max - min), 0, 1) * 100;
    input.parentElement.style.setProperty("--fill", color);
    input.style.setProperty(
        "--track-bg",
        `linear-gradient(90deg, ${color} 0 ${f}%, var(--color-neutral-900) ${f}% 100%)`,
    );
}

function renderMeta() {
    const r = activeStudy()?.response;
    setText(el.metaCkpt, shortPath(server.modelPath) ?? r?.model ?? "—");
    setText(el.metaImgsz, String(r?.imgsz ?? server.imgsz ?? "—"));

    const classes = r?.classes ?? server.classes;
    setText(el.metaClasses, classes?.length ? `${classes.length} · ${classes.join(", ")}` : "—");

    const ms = activeStudy()?.latencyMs;
    setText(el.metaLatency, ms == null ? "—" : `${Math.round(ms)} ms`);
    setText(el.metaEndpoint, PREDICT_ENDPOINT);
}

function renderRaw() {
    const study = activeStudy();
    if (!study || (!study.response && study.status !== "error")) {
        el.rawBlock.hidden = true;
        return;
    }
    el.rawBlock.hidden = false;
    setText(el.rawReq, `POST ${PREDICT_ENDPOINT}`);

    if (study.response) {
        el.rawStatus.className = "raw-status";
        setText(el.rawStatus, String(study.httpStatus ?? 200));
        setText(el.rawBody, JSON.stringify(study.response, null, 2));
    } else {
        el.rawStatus.className = "raw-status is-bad";
        setText(el.rawStatus, String(study.httpStatus ?? "—"));
        setText(el.rawBody, study.error ?? "");
    }
}

// ---------------------------------------------------------------------
// Pipeline steps + action buttons
// ---------------------------------------------------------------------

function renderSteps() {
    const study = activeStudy();
    let active = 1;
    if (study && study.status === "done" && study.detections) active = 3;
    else if (study) active = 2;

    el.steps.querySelectorAll(".step").forEach((node) => {
        const n = Number(node.dataset.step);
        node.classList.toggle("is-done", n < active);
        node.classList.toggle("is-active", n === active);
    });
}

function renderActions() {
    const study = activeStudy();
    const busy = study?.status === "running";

    el.clearBtn.disabled = !study || busy;
    el.runBtn.disabled = !study || busy;

    const label = el.runBtn.querySelector(".btn-label");
    const spinner = el.runBtn.querySelector(".spinner");
    if (busy && !spinner) el.runBtn.prepend(Object.assign(document.createElement("span"), { className: "spinner" }));
    if (!busy && spinner) spinner.remove();
    label.textContent = busy
        ? "Detecting…"
        : (study?.response ? "Re-run detection" : "Run detection");

    el.compareBtn.disabled = studies.length < 2;
    el.compareBtn.classList.toggle("is-on", view.compare);
    if (el.compareBtn.disabled) closePicker();
}

function renderAll() {
    renderQueue();
    renderViewer();
    renderCorners();
    renderFindings();
    renderSteps();
    renderActions();
    if (!el.comparePop.hidden) renderPicker();
}

function noteError(message) {
    el.findingsNote.hidden = false;
    el.findingsNote.className = "findings-note is-error";
    el.findingsNote.textContent = message;
}

// ---------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------

async function runDetection() {
    const study = activeStudy();
    if (!study || study.status === "running") return;

    const conf = Number(el.confRange.value);
    const iou = Number(el.iouRange.value);

    study.status = "running";
    study.error = null;
    renderAll();

    const started = performance.now();
    try {
        const form = new FormData();
        form.append("file", study.file);
        form.append("conf", String(conf));
        form.append("iou", String(iou));

        const res = await fetch(PREDICT_ENDPOINT, { method: "POST", body: form });
        study.httpStatus = res.status;

        if (!res.ok) {
            const detail = await res.json().catch(() => null);
            throw new Error(detail?.detail ?? `Server responded with ${res.status} ${res.statusText}`);
        }

        const data = await res.json();
        study.response = data;
        study.detections = Array.isArray(data.detections) ? data.detections : [];
        study.latencyMs = data.inference_ms ?? (performance.now() - started);
        study.status = "done";
    } catch (err) {
        study.status = "error";
        study.response = null;
        study.detections = null;
        study.error = server.online || study.httpStatus
            ? String(err.message)
            : `Could not reach the detection server at ${PREDICT_ENDPOINT}. ` +
              "Start it with ./start.sh (FastAPI on :8000, Express on :3000).";
        console.error(err);
    }

    renderAll();
}

function exportJson() {
    const study = activeStudy();
    if (!study?.response) return;

    const payload = {
        file: study.name,
        image_width: study.response.image_width,
        image_height: study.response.image_height,
        requested: { conf: Number(el.confRange.value), iou: Number(el.iouRange.value) },
        response: study.response,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${study.name.replace(/\.[^.]+$/, "")}-detection.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------

el.queueAdd.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files?.length) addFiles(el.fileInput.files);
    el.fileInput.value = "";
});
el.queueFilter.addEventListener("input", renderQueue);

// Drops land anywhere on the window, not only on the queue's dashed target.
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    document.body.classList.add("is-dragover");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) document.body.classList.remove("is-dragover");
});
window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("is-dragover");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});

document.querySelectorAll(".tool[data-tool]").forEach((btn) => {
    btn.addEventListener("click", () => setTool(btn.dataset.tool));
});

el.overlaySeg.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-overlay]");
    if (btn) setOverlay(btn.dataset.overlay);
});

el.compareBtn.addEventListener("click", togglePicker);
el.compareApply.addEventListener("click", applyCompare);
el.compareExit.addEventListener("click", exitCompare);

// Anywhere outside the popover — including the queue and the stage — closes it.
document.addEventListener("pointerdown", (e) => {
    if (el.comparePop.hidden) return;
    if (!e.target.closest(".compare-wrap")) closePicker();
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el.comparePop.hidden) {
        closePicker();
        el.compareBtn.focus();
    }
});

el.zoomReadout.addEventListener("click", () => {
    resetTransform();
    applyTransform();
    renderCorners();
});

el.runBtn.addEventListener("click", runDetection);
el.clearBtn.addEventListener("click", clearActive);
el.exportBtn.addEventListener("click", exportJson);

[el.confRange, el.iouRange].forEach((input) => {
    input.addEventListener("input", () => {
        if (input === el.confRange) touchedConf = true;
        if (input === el.iouRange) touchedIou = true;
        renderControls();
        renderCorners();
    });
});

[el.windowRange, el.levelRange].forEach((input) => {
    input.addEventListener("input", () => {
        applyWindowLevel();
        renderControls();
        renderCorners();
    });
});

// --- pointer tools on the stage ---------------------------------------

let drag = null;

el.stage.addEventListener("pointerdown", (e) => {
    if (!activeStudy() || e.button !== 0) return;
    drag = {
        x: e.clientX,
        y: e.clientY,
        zoom: view.zoom,
        panX: view.panX,
        panY: view.panY,
        win: Number(el.windowRange.value),
        level: Number(el.levelRange.value),
    };
    el.stage.setPointerCapture(e.pointerId);
    el.stage.classList.add("is-dragging");
});

el.stage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;

    if (view.tool === "pan") {
        view.panX = drag.panX + dx;
        view.panY = drag.panY + dy;
        applyTransform();
    } else if (view.tool === "zoom") {
        view.zoom = clamp(drag.zoom * Math.exp(-dy / 220), 0.2, 8);
        applyTransform();
    } else {
        // Window/level, the radiology convention: horizontal widens the
        // window, vertical shifts the level.
        el.windowRange.value = String(clamp(drag.win + dx, 20, 480));
        el.levelRange.value = String(clamp(drag.level - dy, 0, 255));
        applyWindowLevel();
        renderControls();
    }
    renderCorners();
});

["pointerup", "pointercancel"].forEach((type) => {
    el.stage.addEventListener(type, (e) => {
        drag = null;
        el.stage.classList.remove("is-dragging");
        if (el.stage.hasPointerCapture?.(e.pointerId)) el.stage.releasePointerCapture(e.pointerId);
    });
});

el.stage.addEventListener("wheel", (e) => {
    if (!activeStudy()) return;
    e.preventDefault();
    view.zoom = clamp(view.zoom * Math.exp(-e.deltaY / 500), 0.2, 8);
    applyTransform();
    renderCorners();
}, { passive: false });

window.addEventListener("resize", () => {
    fitFrames();
    renderCorners();
});

window.addEventListener("beforeunload", () => {
    studies.forEach((s) => URL.revokeObjectURL(s.url));
});

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

setTool("wl");
setOverlay("box");
applyWindowLevel();
applyTransform();
renderAll();

pollHealth();
setInterval(pollHealth, HEALTH_INTERVAL_MS);
