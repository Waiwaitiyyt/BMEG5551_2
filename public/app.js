// Implant workstation — BMEG5552.
//
// Three panes, one shared study list:
//   queue    a client-side session queue; nothing is persisted server-side
//   viewer   the active study with the model's box and heat map drawn over it
//   findings the parsed responses plus the knobs that shape the next run
//
// Three model-backed features, all driven by one "Run analysis" click:
//   localisation  YOLO bounding box            POST /predict
//   prediction    loose vs. well fixed         POST /classify
//   heat map      Grad-CAM saliency            returned by the same /classify
//
// The two calls run concurrently and settle independently, so a classifier
// that is missing or erroring still leaves localisation on screen.
//
// API contract
// ------------
//   GET  {API}/health
//     -> { status, version, model_path, model_name, classes, conf_threshold,
//          iou_threshold, imgsz, device,
//          classifier: { available, model_name, arch, classes, threshold,
//                        img_size, val_metrics } }
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
//   POST {API}/classify    multipart/form-data
//     file      the image        (required)
//     threshold 0 < t < 1        (optional, defaults to the tuned value)
//     heatmap   true | false     (optional, default true)
//     -> { label, probability, confidence, threshold, classes,
//          heatmap (PNG data URI or null), heatmap_box: [x1,y1,x2,y2],
//          image_width, image_height, model, arch, imgsz, inference_ms }
//
//   box coordinates are pixels in the original uploaded image (top-left /
//   bottom-right), which is what drawOverlays() positions against. The heat
//   map arrives as a full-size RGBA PNG, transparent outside the centre crop
//   the classifier saw, so it is simply stretched over the frame.
//
// Served by the Express gateway (server/ts, :3000) the calls go to /api on the
// same origin and Express proxies them to FastAPI. Opened straight from disk
// there is no origin to inherit, so it falls back to FastAPI on :8000 — that
// path needs the FastAPI CORS settings to allow it.

const IS_FILE = window.location.protocol === "file:";
const API_BASE = IS_FILE ? "http://localhost:8000" : "/api";
const PREDICT_ENDPOINT = `${API_BASE}/predict`;
const CLASSIFY_ENDPOINT = `${API_BASE}/classify`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const GATEWAY_HEALTH = IS_FILE ? null : "/healthz";
const HEALTH_INTERVAL_MS = 15000;

// This frontend's own version, stamped onto the PDF report. The inference API
// reports its version through /health; this one has no server to ask.
const APP_VERSION = "0.1.0";

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
    clsSlot: $("cls-slot"),
    detCards: $("det-cards"),
    findingsNote: $("findings-note"),

    confRange: $("conf-range"), confOut: $("conf-out"),
    confTick: $("conf-tick"), confHint: $("conf-hint"),
    iouRange: $("iou-range"), iouOut: $("iou-out"),
    windowRange: $("window-range"), levelRange: $("level-range"), wlOut: $("wl-out"),
    ctrlReset: $("ctrl-reset"),

    metaCkpt: $("meta-ckpt"), metaClsCkpt: $("meta-cls-ckpt"), metaImgsz: $("meta-imgsz"),
    metaClasses: $("meta-classes"), metaLatency: $("meta-latency"),
    metaEndpoint: $("meta-endpoint"),

    rawBlock: $("raw-block"), rawReq: $("raw-req"),
    rawStatus: $("raw-status"), rawBody: $("raw-body"),
    rawClsBlock: $("raw-cls-block"), rawClsReq: $("raw-cls-req"),
    rawClsStatus: $("raw-cls-status"), rawClsBody: $("raw-cls-body"),
    exportBtn: $("export-btn"),
    reportBtn: $("report-btn"),
    saveBox: $("save-box"), saveHeat: $("save-heat"), saveBoth: $("save-both"),

    reportDialog: $("report-dialog"),
    rfPid: $("rf-pid"), rfPname: $("rf-pname"), rfDate: $("rf-date"),
    rfClin: $("rf-clin"), rfNotes: $("rf-notes"), rfCancel: $("rf-cancel"),
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
    apiVersion: null,        // /health.version — stamped onto the PDF report
    // Last thresholds /health reported — what "Reset" restores the two
    // threshold sliders to, in preference to their markup defaults.
    confDefault: null,
    iouDefault: null,
    // Mirror of /health.classifier — null until the first successful poll, and
    // { available: false } when the checkpoint is not on the server.
    classifier: null,
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
        server.apiVersion = data.version ?? server.apiVersion;
        server.imgsz = data.imgsz ?? server.imgsz;
        server.classes = Array.isArray(data.classes) ? data.classes : [];
        server.modelPath = data.model_path ?? null;
        server.modelName = data.model_name ?? basename(data.model_path);
        server.classifier = data.classifier ?? null;

        el.dotApi.className = `dot ${server.online ? "is-up" : "is-wait"}`;
        setText(el.chipApi, `FastAPI ${server.online ? "ready" : "loading"}`);
        setText(el.chipModel, server.classifier?.available
            ? `YOLO + ${server.classifier.arch ?? "CNN"}`
            : `${server.modelName ?? "—"} · YOLO`);

        // Only adopt the server's thresholds until the reader touches a slider;
        // after that the sliders are the source of truth for the next run.
        if (typeof data.conf_threshold === "number") server.confDefault = data.conf_threshold;
        if (typeof data.iou_threshold === "number") server.iouDefault = data.iou_threshold;
        if (!touchedConf && server.confDefault !== null) {
            el.confRange.value = String(server.confDefault);
        }
        if (!touchedIou && server.iouDefault !== null) {
            el.iouRange.value = String(server.iouDefault);
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
            // The classification half is tracked separately so one failing
            // model never blanks the other's result.
            classification: null,
            classError: null,
            classHttpStatus: null,
            classLatencyMs: null,
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
    // Scanning a worklist, the verdict beats the box confidence; the box is
    // still shown when only the detector answered.
    if (study.status === "done" && study.classification) {
        const span = document.createElement("span");
        const loose = study.classification.label === "Loose";
        span.className = `queue-verdict ${loose ? "is-loose" : "is-control"}`;
        span.textContent = `${loose ? "loose" : "fixed"} ${pct(study.classification.confidence)}`;
        return span;
    }

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
    // Fit first: applyTransform() clamps the pan against the frame box, so it
    // needs the sizes fitFrames() writes rather than the previous study's.
    fitFrames();
    applyTransform();
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
 * Draw the model output over one frame: the detector's boxes and the
 * classifier's Grad-CAM.
 *
 * `study.detections[i].box` is [x1, y1, x2, y2] in the original image's pixels,
 * so every edge becomes a percentage of the image box and rides along with the
 * zoom/pan transform. The heat map needs no such maths — the server returns it
 * already the size of the upload, transparent outside the region the classifier
 * saw, so it is stretched edge to edge.
 */
function drawOverlays(container, study, secondary) {
    if (!container) return;
    container.replaceChildren();
    if (!study || view.overlay === "off") return;

    const wantBox = view.overlay === "box" || view.overlay === "both";
    const wantHeat = view.overlay === "heat" || view.overlay === "both";

    // Painted first so the boxes and labels stay legible on top of it.
    if (wantHeat && study.classification?.heatmap) {
        const heat = document.createElement("img");
        heat.className = "cam-heat";
        heat.src = study.classification.heatmap;
        heat.alt = "";
        container.appendChild(heat);
    }

    if (study.status !== "done" || !study.detections?.length) return;

    const W = study.response?.image_width || study.width;
    const H = study.response?.image_height || study.height;
    if (!W || !H) return;

    study.detections.forEach((det, i) => {
        const [x1, y1, x2, y2] = det.box;
        const rect = {
            left: `${(x1 / W) * 100}%`,
            top: `${(y1 / H) * 100}%`,
            width: `${((x2 - x1) / W) * 100}%`,
            height: `${((y2 - y1) / H) * 100}%`,
        };

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

/**
 * How far the image may be dragged from centre: however much of it hangs
 * outside its frame once scaled. Fitted to the frame there is nothing hidden,
 * so both limits are zero and the image stays put; zoomed in, every edge is
 * reachable and the image can never be dragged out of view.
 */
function panLimits() {
    const frame = el.frameInner.parentElement;
    if (!frame || !el.frameInner.offsetWidth) return { x: 0, y: 0 };
    return {
        x: Math.max(0, (el.frameInner.offsetWidth * view.zoom - frame.clientWidth) / 2),
        y: Math.max(0, (el.frameInner.offsetHeight * view.zoom - frame.clientHeight) / 2),
    };
}

/** True when some of the image is off-frame, i.e. dragging would show more. */
function canPan() {
    const lim = panLimits();
    return lim.x > 0.5 || lim.y > 0.5;
}

function applyTransform() {
    const lim = panLimits();
    view.panX = clamp(view.panX, -lim.x, lim.x);
    view.panY = clamp(view.panY, -lim.y, lim.y);

    const t = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
    el.frameInner.style.transform = t;
    el.leftInner.style.transform = t;
    // Grab cursor whenever a plain drag would pan, whatever tool is armed.
    el.stage.classList.toggle("is-pannable", canPan());
    setText(el.zoomReadout, `${Math.round(view.zoom * 100)}%`);
}

/**
 * Zoom by `factor` about a point on screen, keeping the pixel under that point
 * under it. With `translate(pan) scale(z)` about the frame's centre, the image
 * point at cursor offset m is u = (m − pan) / z; holding u fixed across the
 * zoom change gives the pan below.
 */
function zoomAt(factor, clientX, clientY) {
    const next = clamp(view.zoom * factor, 0.2, 8);
    const rect = el.frameInner.parentElement.getBoundingClientRect();
    const mx = clientX - (rect.left + rect.width / 2);
    const my = clientY - (rect.top + rect.height / 2);
    const k = next / view.zoom;

    view.panX = mx - k * (mx - view.panX);
    view.panY = my - k * (my - view.panY);
    view.zoom = next;
    applyTransform();
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
    const cls = study.classification;
    const verdict = cls ? `${cls.label.toUpperCase()} p=${cls.probability.toFixed(3)}` : "not classified";
    setText(el.cornerBr,
        `imgsz ${imgsz} · conf ${fmt2(conf)} · iou ${fmt2(iou)}\n${verdict}`);
}

function statusLabel(study) {
    if (!study) return "idle";
    if (study.status === "running") return "analysing";
    if (study.status === "error") return "error";
    if (study.status === "done") {
        if (study.classification) return study.classification.label.toLowerCase();
        return study.detections?.length ? "detected" : "no detection";
    }
    return "queued";
}

// ---------------------------------------------------------------------
// Findings pane
// ---------------------------------------------------------------------

function renderFindings() {
    const study = activeStudy();

    // status pill
    el.statusPill.className = "status-pill";
    const cls = study?.classification;
    if (!study) setText(el.statusPill, "Idle");
    else if (study.status === "running") { el.statusPill.classList.add("is-busy"); setText(el.statusPill, "Analysing…"); }
    else if (study.status === "done" && cls) {
        // The verdict is what a reader looks for first, so it owns the pill.
        el.statusPill.classList.add(cls.label === "Loose" ? "is-bad" : "is-ok");
        setText(el.statusPill, `${cls.label} · ${pct(cls.confidence)}`);
    }
    else if (study.status === "error") { el.statusPill.classList.add("is-bad"); setText(el.statusPill, "Error"); }
    else if (study.status === "done" && study.detections?.length) { el.statusPill.classList.add("is-ok"); setText(el.statusPill, "Detected"); }
    else if (study.status === "done") { el.statusPill.classList.add("is-bad"); setText(el.statusPill, "No detection"); }
    else setText(el.statusPill, "Queued");

    // classification card, above the detections it explains
    el.clsSlot.replaceChildren();
    if (cls) el.clsSlot.appendChild(classificationCard(cls, study));
    else if (study?.classError) el.clsSlot.appendChild(classErrorCard(study.classError));

    // detection cards
    el.detCards.replaceChildren();
    const dets = study?.status === "done" ? (study.detections ?? []) : [];
    dets.forEach((det) => el.detCards.appendChild(detectionCard(det)));

    // the note that stands in for cards when there is nothing to show
    el.findingsNote.className = "findings-note";
    if (dets.length || cls) {
        el.findingsNote.hidden = true;
    } else {
        el.findingsNote.hidden = false;
        if (!study) {
            el.findingsNote.textContent =
                "Select or drop a study, then run the analysis to see the implant bounding box, " +
                "the loosening verdict, and the Grad-CAM heat map here.";
        } else if (study.status === "error") {
            el.findingsNote.classList.add("is-error");
            el.findingsNote.textContent = study.error;
        } else if (study.status === "running") {
            el.findingsNote.textContent = "Running detection and classification on the uploaded pixels…";
        } else if (study.status === "done") {
            el.findingsNote.textContent =
                `No implant scored above the ${fmt2(study.response?.conf_threshold ?? 0)} confidence threshold. ` +
                "Lower the threshold and re-run to see weaker candidates.";
        } else {
            el.findingsNote.textContent = "Ready. Run the analysis to locate and classify the implant in this study.";
        }
    }

    renderControls();
    renderMeta();
    renderRaw();

    el.exportBtn.disabled = !study?.response && !study?.classification;
    el.reportBtn.disabled = !study?.response && !study?.classification;

    const avail = overlayAvailability(study);
    el.saveBox.disabled = !avail.box;
    el.saveHeat.disabled = !avail.heat;
    el.saveBoth.disabled = !avail.any;
}

/**
 * The loosening verdict.
 *
 * p(Loose) is shown alongside the label because the label alone hides how
 * close the call was: a 0.62 and a 0.99 both read "Loose", and only one of
 * them is worth a second look. The tuned threshold is marked on the bar so it
 * is visible how much margin the decision had.
 */
function classificationCard(cls, study) {
    const loose = cls.label === "Loose";
    const card = document.createElement("div");
    card.className = `det-card cls-card ${loose ? "is-loose" : "is-control"}`;

    const head = document.createElement("div");
    head.className = "det-card-head";

    const name = document.createElement("span");
    name.className = "det-name";
    const swatch = document.createElement("span");
    swatch.className = "det-swatch cls-swatch";
    name.append(swatch, document.createTextNode(loose ? "Aseptic loosening" : "Well fixed"));

    const value = document.createElement("span");
    value.className = "det-conf";
    value.textContent = pct(cls.confidence);
    value.title = `Confidence in "${cls.label}"`;
    head.append(name, value);

    const bar = document.createElement("div");
    bar.className = "det-bar cls-bar";
    bar.title = `p(loose) = ${cls.probability.toFixed(3)} on a 0-1 axis; the tick marks the ${fmt2(cls.threshold)} decision threshold`;
    const fill = document.createElement("span");
    fill.style.width = `${clamp(cls.probability, 0, 1) * 100}%`;
    const mark = document.createElement("i");
    mark.className = "cls-thresh";
    mark.style.left = `${clamp(cls.threshold, 0, 1) * 100}%`;
    mark.title = `Decision threshold ${fmt2(cls.threshold)}`;
    bar.append(fill, mark);

    const legend = document.createElement("div");
    legend.className = "cls-legend";
    // Three cells, not four: the architecture is already named in the run
    // metadata below, and a fourth column truncated the values.
    legend.append(
        labelled("p(loose)", cls.probability.toFixed(3)),
        labelled("threshold", fmt2(cls.threshold)),
        labelled("latency", `${Math.round(study?.classLatencyMs ?? cls.inference_ms ?? 0)} ms`),
    );

    const hint = document.createElement("p");
    hint.className = "cls-hint";
    if (!cls.heatmap) {
        hint.textContent = "Heat map was not requested for this run.";
    } else if (loose) {
        hint.textContent =
            "Switch the overlay to Heatmap to see the Grad-CAM region that drove this call.";
    } else {
        // Grad-CAM is renormalised per image, so the map always has a hot spot
        // even when the underlying activation is negligible. On a well-fixed
        // implant that spot is the strongest evidence the model could find for
        // loosening — which is near zero here, and is not a finding.
        hint.textContent =
            "Switch the overlay to Heatmap to see where the weak loosening evidence sat. " +
            "Grad-CAM is rescaled per image, so a hot spot appears even at p(loose) " +
            `${cls.probability.toFixed(3)} — read it as the model's best candidate, not a finding.`;
    }

    card.append(head, bar, legend, hint);
    return card;
}

function labelled(term, value) {
    const wrap = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = value;
    const span = document.createElement("span");
    span.textContent = term;
    wrap.append(b, span);
    return wrap;
}

/** Shown when /classify failed but detection may still have succeeded. */
function classErrorCard(message) {
    const card = document.createElement("div");
    card.className = "det-card cls-card is-error";
    const head = document.createElement("div");
    head.className = "det-card-head";
    const name = document.createElement("span");
    name.className = "det-name";
    name.textContent = "Classification unavailable";
    head.appendChild(name);
    const body = document.createElement("p");
    body.className = "cls-hint";
    body.textContent = message;
    card.append(head, body);
    return card;
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

/**
 * Where "Reset" puts the four knobs: the server's thresholds once /health has
 * answered, otherwise the value= attributes in the markup, which the window
 * and level sliders always use since neither is a server-side setting.
 */
function controlDefaults() {
    return {
        conf: server.confDefault ?? Number(el.confRange.defaultValue),
        iou: server.iouDefault ?? Number(el.iouRange.defaultValue),
        win: Number(el.windowRange.defaultValue),
        level: Number(el.levelRange.defaultValue),
    };
}

/** True when nothing is left to reset — the button greys out on that. */
function controlsAreDefault() {
    const d = controlDefaults();
    // The sliders snap to their step, so a server threshold that falls between
    // two steps never compares exactly equal; a half-step tolerance does.
    const near = (a, b) => Math.abs(a - b) < 0.005;
    return near(Number(el.confRange.value), d.conf)
        && near(Number(el.iouRange.value), d.iou)
        && Number(el.windowRange.value) === d.win
        && Number(el.levelRange.value) === d.level;
}

/**
 * Back to the defaults. Clearing the touched flags also hands the thresholds
 * back to /health, so a server restarted with different settings is adopted
 * again on the next poll.
 */
function resetControls() {
    const d = controlDefaults();
    touchedConf = false;
    touchedIou = false;
    el.confRange.value = String(d.conf);
    el.iouRange.value = String(d.iou);
    el.windowRange.value = String(d.win);
    el.levelRange.value = String(d.level);
    applyWindowLevel();
    renderControls();
    renderCorners();
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

    el.ctrlReset.disabled = controlsAreDefault();

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
    const study = activeStudy();
    const r = study?.response;
    const cls = study?.classification;
    setText(el.metaCkpt, shortPath(server.modelPath) ?? r?.model ?? "—");

    const clsInfo = server.classifier;
    setText(el.metaClsCkpt, clsInfo?.available
        ? `${shortPath(clsInfo.model_path) ?? clsInfo.model_name} · ${clsInfo.arch}`
        : (clsInfo ? "not loaded" : "—"));

    // Two models, two input sizes — showing one would misdescribe the other.
    const sizes = [r?.imgsz ?? server.imgsz, cls?.imgsz ?? clsInfo?.img_size]
        .filter((v) => v != null);
    setText(el.metaImgsz, sizes.length ? [...new Set(sizes)].join(" · ") : "—");

    const detClasses = r?.classes ?? server.classes ?? [];
    const clsClasses = cls?.classes ?? clsInfo?.classes ?? [];
    const allClasses = [...detClasses, ...clsClasses];
    setText(el.metaClasses, allClasses.length
        ? `${allClasses.length} · ${allClasses.join(", ")}`
        : "—");

    // The wall-clock cost of a run is the slower of two concurrent calls.
    const parts = [study?.latencyMs, study?.classLatencyMs].filter((v) => v != null);
    setText(el.metaLatency, parts.length ? `${Math.round(Math.max(...parts))} ms` : "—");
    setText(el.metaEndpoint, `${PREDICT_ENDPOINT} + ${CLASSIFY_ENDPOINT}`);
}

function renderRaw() {
    const study = activeStudy();

    const showDetect = study && (study.response || study.status === "error");
    el.rawBlock.hidden = !showDetect;
    if (showDetect) {
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

    const showClassify = study && (study.classification || study.classError);
    el.rawClsBlock.hidden = !showClassify;
    if (showClassify) {
        setText(el.rawClsReq, `POST ${CLASSIFY_ENDPOINT}`);
        if (study.classification) {
            el.rawClsStatus.className = "raw-status";
            setText(el.rawClsStatus, String(study.classHttpStatus ?? 200));
            // The base64 PNG is tens of kilobytes of noise in a debug panel;
            // its size is the only part worth reading.
            const { heatmap, ...rest } = study.classification;
            setText(el.rawClsBody, JSON.stringify({
                ...rest,
                heatmap: heatmap ? `<data:image/png;base64, ${Math.round(heatmap.length / 1024)} KB>` : null,
            }, null, 2));
        } else {
            el.rawClsStatus.className = "raw-status is-bad";
            setText(el.rawClsStatus, String(study.classHttpStatus ?? "—"));
            setText(el.rawClsBody, study.classError ?? "");
        }
    }
}

// ---------------------------------------------------------------------
// Pipeline steps + action buttons
// ---------------------------------------------------------------------

function renderSteps() {
    const study = activeStudy();
    let active = 1;
    if (study && study.status === "done" && (study.detections || study.classification)) active = 3;
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
        ? "Analysing…"
        : (study?.response || study?.classification ? "Re-run analysis" : "Run analysis");

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
// Inference
// ---------------------------------------------------------------------

/** POST one study to one endpoint; resolves to { status, data } or throws. */
async function postStudy(endpoint, study, fields) {
    const form = new FormData();
    form.append("file", study.file);
    Object.entries(fields).forEach(([key, value]) => form.append(key, String(value)));

    const res = await fetch(endpoint, { method: "POST", body: form });
    if (!res.ok) {
        const detail = await res.json().catch(() => null);
        const error = new Error(
            detail?.detail ?? `Server responded with ${res.status} ${res.statusText}`,
        );
        error.httpStatus = res.status;
        throw error;
    }
    return { status: res.status, data: await res.json() };
}

/**
 * Run both models on the active study.
 *
 * The calls go out together and are settled independently: the classifier is
 * an order of magnitude slower than the detector, and a server without the
 * classifier checkpoint answers /classify with 503 while /predict keeps
 * working. Either half failing must still leave the other's result on screen.
 */
async function runAnalysis() {
    const study = activeStudy();
    if (!study || study.status === "running") return;

    const conf = Number(el.confRange.value);
    const iou = Number(el.iouRange.value);

    study.status = "running";
    study.error = null;
    study.classError = null;
    renderAll();

    const started = performance.now();
    const [detect, classify] = await Promise.allSettled([
        postStudy(PREDICT_ENDPOINT, study, { conf, iou }),
        postStudy(CLASSIFY_ENDPOINT, study, { heatmap: true }),
    ]);

    if (detect.status === "fulfilled") {
        const { status, data } = detect.value;
        study.httpStatus = status;
        study.response = data;
        study.detections = Array.isArray(data.detections) ? data.detections : [];
        study.latencyMs = data.inference_ms ?? (performance.now() - started);
    } else {
        study.httpStatus = detect.reason.httpStatus ?? null;
        study.response = null;
        study.detections = null;
        study.error = server.online || study.httpStatus
            ? String(detect.reason.message)
            : `Could not reach the inference server at ${PREDICT_ENDPOINT}. ` +
              "Start it with ./start.sh (FastAPI on :8000, Express on :3000).";
        console.error(detect.reason);
    }

    if (classify.status === "fulfilled") {
        const { status, data } = classify.value;
        study.classHttpStatus = status;
        study.classification = data;
        study.classLatencyMs = data.inference_ms ?? (performance.now() - started);
    } else {
        study.classHttpStatus = classify.reason.httpStatus ?? null;
        study.classification = null;
        study.classError = String(classify.reason.message);
        console.error(classify.reason);
    }

    // "error" only when nothing at all came back; a half-run still has
    // something worth showing, and the failing half explains itself in place.
    study.status = detect.status === "rejected" && classify.status === "rejected" ? "error" : "done";

    renderAll();
}

function exportJson() {
    const study = activeStudy();
    if (!study?.response && !study?.classification) return;

    // The heat map is dropped: a base64 PNG would dwarf the numbers this file
    // exists to carry, and it is reproducible from the same upload.
    const classification = study.classification
        ? (({ heatmap, ...rest }) => rest)(study.classification)
        : null;

    const payload = {
        file: study.name,
        image_width: study.response?.image_width ?? study.classification?.image_width,
        image_height: study.response?.image_height ?? study.classification?.image_height,
        requested: { conf: Number(el.confRange.value), iou: Number(el.iouRange.value) },
        detection: study.response ?? { error: study.error },
        classification: classification ?? { error: study.classError },
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${study.name.replace(/\.[^.]+$/, "")}-analysis.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------
// PDF report — a self-contained page opened in a new window, printed to
// PDF by the browser. No libraries, so it also works from file://.
// ---------------------------------------------------------------------

function esc(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("an image for the report failed to load"));
        img.src = src;
    });
}

/** Decode the study's X-ray and, if present, its Grad-CAM PNG once. */
async function studyAssets(study) {
    const base = await loadImage(study.url);
    let heat = null;
    if (study.classification?.heatmap) {
        heat = await loadImage(study.classification.heatmap).catch(() => null);
    }
    return { base, heat };
}

/**
 * Draw one study onto a fresh canvas at the X-ray's native resolution, with the
 * window/level the viewer is currently showing baked in — so a saved image
 * matches what was on screen rather than the raw file. `opts.heat` adds the
 * Grad-CAM overlay, `opts.box` adds the detector rectangles; pass neither for
 * the clean image.
 */
function renderStudyCanvas(study, base, heat, opts) {
    const W = study.response?.image_width || study.classification?.image_width || study.width;
    const H = study.response?.image_height || study.classification?.image_height || study.height;

    const win = Number(el.windowRange.value);
    const level = Math.max(1, Number(el.levelRange.value));

    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.filter = `brightness(${(127.5 / level).toFixed(3)}) contrast(${(255 / win).toFixed(3)})`;
    ctx.drawImage(base, 0, 0, W, H);
    ctx.filter = "none";

    if (opts.heat && heat) ctx.drawImage(heat, 0, 0, W, H);

    if (opts.box) {
        const dets = study.status === "done" ? (study.detections ?? []) : [];
        const fontPx = Math.max(11, Math.round(Math.min(W, H) / 32));
        ctx.lineWidth = Math.max(2, Math.round(Math.min(W, H) / 220));
        ctx.font = `600 ${fontPx}px "Inter", system-ui, sans-serif`;
        ctx.textBaseline = "bottom";
        dets.forEach((det, i) => {
            const [x1, y1, x2, y2] = det.box;
            const colour = i === 0 ? "#2fb46b" : "#e0a23c";
            ctx.strokeStyle = colour;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const tag = `${det.label ?? "implant"} ${Number(det.confidence ?? 0).toFixed(2)}`;
            const tw = ctx.measureText(tag).width;
            const ty = y1 > fontPx + 6 ? y1 : y1 + fontPx + 8;
            ctx.fillStyle = colour;
            ctx.fillRect(x1, ty - fontPx - 4, tw + 10, fontPx + 4);
            ctx.fillStyle = "#0b0c12";
            ctx.fillText(tag, x1 + 5, ty - 2);
        });
    }
    return canvas;
}

/**
 * Bake the review into two PNG data URIs for the report: the X-ray on its own,
 * and the same frame with the boxes and the Grad-CAM heat map drawn on.
 */
async function buildReportImages(study) {
    const { base, heat } = await studyAssets(study);
    const canvas = renderStudyCanvas(study, base, heat, {});
    return {
        width: canvas.width,
        height: canvas.height,
        original: canvas.toDataURL("image/png"),
        composite: renderStudyCanvas(study, base, heat, { box: true, heat: true }).toDataURL("image/png"),
    };
}

// The three overlay flavours the "Save image" buttons can write.
const OVERLAY_KINDS = {
    box: { opts: { box: true }, suffix: "box", needs: "box" },
    heat: { opts: { heat: true }, suffix: "heatmap", needs: "heat" },
    both: { opts: { box: true, heat: true }, suffix: "box-heatmap", needs: "any" },
};

function overlayAvailability(study) {
    const box = !!(study?.status === "done" && study.detections?.length);
    const heat = !!study?.classification?.heatmap;
    return { box, heat, any: box || heat };
}

/** Save the active study's X-ray as a PNG with the chosen overlay(s) drawn on. */
async function saveOverlayImage(kind) {
    const study = activeStudy();
    const spec = OVERLAY_KINDS[kind];
    if (!study || !spec) return;

    if (!overlayAvailability(study)[spec.needs]) {
        noteError(
            spec.needs === "box"
                ? "No detection to save yet — run the analysis first."
                : spec.needs === "heat"
                  ? "No heat map to save — run the analysis with the classifier available."
                  : "Nothing to save yet — run the analysis first.",
        );
        return;
    }

    try {
        const { base, heat } = await studyAssets(study);
        const canvas = renderStudyCanvas(study, base, heat, spec.opts);
        const name = `${study.name.replace(/\.[^.]+$/, "")}-${spec.suffix}.png`;
        canvas.toBlob((blob) => {
            if (!blob) {
                noteError("Could not encode the image as PNG.");
                return;
            }
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = name;
            a.click();
            URL.revokeObjectURL(url);
        }, "image/png");
    } catch (err) {
        console.error(err);
        noteError(`Could not save the image: ${err.message}`);
    }
}

function collectReportMeta() {
    const val = (node) => (node?.value ?? "").trim();
    return {
        patientId: val(el.rfPid),
        patientName: val(el.rfPname),
        studyDate: val(el.rfDate),
        clinician: val(el.rfClin),
        notes: val(el.rfNotes),
    };
}

/** One `<dt>/<dd>` row, or "" when the value is blank so the grid stays tidy. */
function kvRow(term, value) {
    if (value === "" || value == null) return "";
    return `<div class="kv"><dt>${esc(term)}</dt><dd>${esc(value)}</dd></div>`;
}

function renderReportHtml(study, meta, images) {
    const now = new Date();
    const cls = study.classification;
    const r = study.response;
    const clsInfo = server.classifier;
    const num = (v) => (typeof v === "number" ? v.toFixed(3) : "—");

    // --- diagnosis -----------------------------------------------------
    let verdict;
    let tone;
    let detail;
    if (cls) {
        const loose = cls.label === "Loose";
        verdict = loose ? "Aseptic loosening — suspected" : "Well fixed — no loosening indicated";
        tone = loose ? "bad" : "ok";
        const margin = Math.abs((cls.probability ?? 0) - (cls.threshold ?? 0));
        detail =
            `p(loose) = ${(cls.probability ?? 0).toFixed(3)} against a decision threshold of ` +
            `${(cls.threshold ?? 0).toFixed(4)}. Reported confidence in "${esc(cls.label)}" is ` +
            `${pct(cls.confidence)}. Margin to threshold ${margin.toFixed(3)}` +
            (margin < 0.1 ? " — a marginal call that warrants a second read." : ".");
    } else if (study.classError) {
        verdict = "Classification unavailable";
        tone = "warn";
        detail = esc(study.classError);
    } else {
        verdict = "Not classified";
        tone = "warn";
        detail = "The loosening classifier did not return a result for this study.";
    }

    // --- detections --------------------------------------------------------
    const dets = study.status === "done" ? (study.detections ?? []) : [];
    const detTable = dets.length
        ? `<table class="tbl"><thead><tr><th>#</th><th>Label</th><th>Confidence</th>` +
          `<th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead><tbody>` +
          dets
              .map((d, i) => {
                  const [x1, y1, x2, y2] = d.box.map((v) => Math.round(v));
                  return (
                      `<tr><td>${i + 1}</td><td>${esc(d.label ?? "implant")}</td>` +
                      `<td>${Number(d.confidence ?? 0).toFixed(3)}</td>` +
                      `<td>${x1}</td><td>${y1}</td><td>${x2}</td><td>${y2}</td></tr>`
                  );
              })
              .join("") +
          `</tbody></table>`
        : `<p class="muted">No implant scored above the ` +
          `${fmt2(r?.conf_threshold ?? Number(el.confRange.value))} confidence threshold.</p>`;

    // --- case + software / model metadata --------------------------------
    const caseRows = [
        kvRow("Patient ID", meta.patientId),
        kvRow("Patient name", meta.patientName),
        kvRow("Study date", meta.studyDate),
        kvRow("Reporting clinician", meta.clinician),
        kvRow("Source file", study.name),
        kvRow("Image", `${images.width} × ${images.height} px · ${study.kind}`),
        kvRow("Analysis run", now.toLocaleString()),
    ].join("");

    const detLatency =
        study.latencyMs != null ? study.latencyMs : r?.inference_ms;
    const clsLatency =
        study.classLatencyMs != null ? study.classLatencyMs : cls?.inference_ms;
    const vm = clsInfo?.val_metrics;
    const metaRows = [
        kvRow("Frontend software", `Implant Locator Workstation v${APP_VERSION}`),
        kvRow("Inference API", `Implant Locator Inference API v${server.apiVersion ?? "unknown"}`),
        kvRow("Detector checkpoint", shortPath(server.modelPath) ?? r?.model ?? "—"),
        kvRow("Detector classes", (r?.classes ?? server.classes ?? []).join(", ") || "—"),
        kvRow("Detector image size", `${r?.imgsz ?? server.imgsz} px`),
        kvRow("Confidence threshold", (r?.conf_threshold ?? Number(el.confRange.value)).toFixed(2)),
        kvRow("IoU / NMS threshold", (r?.iou_threshold ?? Number(el.iouRange.value)).toFixed(2)),
        kvRow("Detector latency", detLatency != null ? `${Math.round(detLatency)} ms` : ""),
        kvRow(
            "Classifier checkpoint",
            clsInfo?.available
                ? `${shortPath(clsInfo.model_path) ?? clsInfo.model_name} · ${clsInfo.arch ?? cls?.arch ?? "CNN"}`
                : cls
                  ? `${cls.model ?? "classifier"} · ${cls.arch ?? "CNN"}`
                  : "not loaded",
        ),
        kvRow("Classifier classes", (cls?.classes ?? clsInfo?.classes ?? []).join(", ") || "—"),
        kvRow(
            "Classifier image size",
            cls?.imgsz ?? clsInfo?.img_size ? `${cls?.imgsz ?? clsInfo?.img_size} px` : "",
        ),
        kvRow(
            "Decision threshold",
            cls ? Number(cls.threshold ?? clsInfo?.threshold ?? 0).toFixed(4) : "",
        ),
        kvRow("Classifier latency", clsLatency != null ? `${Math.round(clsLatency)} ms` : ""),
        kvRow(
            "Model validation",
            vm
                ? `AUC ${num(vm.auc)}, accuracy ${num(vm.accuracy)}, sensitivity ${num(vm.sensitivity)}, ` +
                  `specificity ${num(vm.specificity)} (validation split, not this case)`
                : "",
        ),
        kvRow("Endpoints", `${PREDICT_ENDPOINT} · ${CLASSIFY_ENDPOINT}`),
        kvRow("Window / level applied", `${el.windowRange.value} / ${el.levelRange.value}`),
    ].join("");

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Loosening report — ${esc(study.name)}</title>
<style>
  @page { size: A4; margin: 16mm; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.5 "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #14161f; background: #f4f5f8; }
  .sheet { max-width: 820px; margin: 24px auto; background: #fff; padding: 32px 36px; box-shadow: 0 1px 4px rgba(0,0,0,.15); }
  header.rep { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; border-bottom: 2px solid #14161f; padding-bottom: 12px; }
  header.rep h1 { font-size: 18px; margin: 0 0 2px; }
  header.rep .sub { color: #5b6070; font-size: 12px; }
  .proto { flex: none; font-size: 10.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: #a23c3c; border: 1px solid #d9a9a9; border-radius: 4px; padding: 3px 7px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  h2 { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: #5b6070; margin: 22px 0 8px; }
  section { break-inside: avoid; }
  .verdict { border: 1px solid; border-radius: 8px; padding: 12px 14px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .verdict.ok { border-color: #b7e0c6; background: #eef8f1; }
  .verdict.bad { border-color: #e6b6b6; background: #fbeeee; }
  .verdict.warn { border-color: #e6d3a8; background: #fbf4e4; }
  .verdict b { display: block; font-size: 15px; margin-bottom: 3px; }
  .verdict p { margin: 0; font-size: 12px; color: #3a3f4d; }
  .kv-list { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; margin: 0; }
  .kv { display: flex; gap: 8px; border-bottom: 1px solid #eceef2; padding: 4px 0; font-size: 12px; }
  .kv dt { color: #5b6070; min-width: 132px; flex: none; }
  .kv dd { margin: 0; font-weight: 500; word-break: break-word; }
  figure { margin: 0; }
  figure img { width: 100%; border: 1px solid #ccced8; display: block; }
  .fig-aside { max-width: 280px; }
  figcaption { font-size: 11px; color: #5b6070; margin-top: 4px; }
  table.tbl { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  table.tbl th, table.tbl td { border: 1px solid #dcdfe6; padding: 4px 7px; text-align: right; }
  table.tbl th { background: #f2f3f6; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  table.tbl th:nth-child(2), table.tbl td:nth-child(2) { text-align: left; }
  .muted { color: #5b6070; font-size: 12px; }
  .notes { white-space: pre-wrap; border: 1px solid #eceef2; border-radius: 6px; padding: 10px 12px; font-size: 12px; }
  footer.rep { margin-top: 26px; border-top: 1px solid #dcdfe6; padding-top: 10px; font-size: 10.5px; color: #6b7080; }
  .bar { display: flex; gap: 8px; justify-content: center; padding: 10px; background: #14161f; }
  .bar button { font: inherit; font-size: 12px; padding: 7px 14px; border: 0; border-radius: 6px; background: #9184d9; color: #fff; cursor: pointer; }
  @media print { body { background: #fff; } .sheet { box-shadow: none; margin: 0; max-width: none; padding: 0; } .bar { display: none; } }
</style>
</head>
<body>
<div class="bar">
  <button type="button" onclick="window.print()">Print / Save as PDF</button>
  <button type="button" onclick="window.close()">Close</button>
</div>
<div class="sheet">
  <header class="rep">
    <div>
      <h1>Implant Loosening Analysis Report</h1>
      <div class="sub">BMEG5552 · University of Western Australia · automated screening prototype</div>
    </div>
    <span class="proto">Not for clinical use</span>
  </header>

  <section>
    <h2>Diagnosis</h2>
    <div class="verdict ${tone}"><b>${esc(verdict)}</b><p>${detail}</p></div>
  </section>

  <section>
    <h2>Case</h2>
    <dl class="kv-list">${caseRows}</dl>
    ${meta.notes ? `<h2>Notes</h2><div class="notes">${esc(meta.notes)}</div>` : ""}
  </section>

  <section>
    <h2>Imaging</h2>
    <figure>
      <img alt="X-ray with implant bounding box and Grad-CAM overlay" src="${images.composite}">
      <figcaption>YOLO bounding box (green) and Grad-CAM saliency from the ResNet50 classifier, drawn over the X-ray at window/level ${esc(el.windowRange.value)}/${esc(el.levelRange.value)} as reviewed. Grad-CAM is renormalised per image, so a hot spot appears even when the underlying activation is negligible.</figcaption>
    </figure>
    <figure class="fig-aside" style="margin-top:14px">
      <img alt="Original X-ray" src="${images.original}">
      <figcaption>Source X-ray, overlays removed.</figcaption>
    </figure>
  </section>

  <section>
    <h2>Detections</h2>
    ${detTable}
  </section>

  <section>
    <h2>Software &amp; model metadata</h2>
    <dl class="kv-list">${metaRows}</dl>
  </section>

  <footer class="rep">
    Generated by Implant Locator Workstation v${esc(APP_VERSION)} on ${esc(now.toLocaleString())}.
    The loosening classifier was trained on 206 images from a single public database — far too small to support a clinical claim.
    This report is a demonstration artefact and must not be used to guide patient care.
  </footer>
</div>
<script>window.addEventListener("load", function () { setTimeout(function () { try { window.focus(); window.print(); } catch (e) {} }, 400); });</script>
</body>
</html>`;
}

async function exportPdfReport() {
    const study = activeStudy();
    if (!study || (!study.response && !study.classification)) return;

    const label = el.reportBtn.textContent;
    el.reportBtn.disabled = true;
    el.reportBtn.textContent = "Preparing…";
    try {
        const meta = collectReportMeta();
        const images = await buildReportImages(study);
        const win = window.open("", "_blank");
        if (!win) {
            noteError(
                "The report could not open a new tab. Allow pop-ups for this page, then try again.",
            );
            return;
        }
        win.document.open();
        win.document.write(renderReportHtml(study, meta, images));
        win.document.close();
    } catch (err) {
        console.error(err);
        noteError(`Could not build the PDF report: ${err.message}`);
    } finally {
        el.reportBtn.textContent = label;
        renderFindings();
    }
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

el.runBtn.addEventListener("click", runAnalysis);
el.clearBtn.addEventListener("click", clearActive);
el.exportBtn.addEventListener("click", exportJson);

// The PDF report collects a few optional case fields first; browsers without
// <dialog> support skip straight to the report.
el.reportBtn.addEventListener("click", () => {
    if (typeof el.reportDialog?.showModal === "function") el.reportDialog.showModal();
    else exportPdfReport();
});
el.rfCancel.addEventListener("click", () => el.reportDialog.close("cancel"));
el.reportDialog.addEventListener("close", () => {
    if (el.reportDialog.returnValue === "go") exportPdfReport();
});

el.saveBox.addEventListener("click", () => saveOverlayImage("box"));
el.saveHeat.addEventListener("click", () => saveOverlayImage("heat"));
el.saveBoth.addEventListener("click", () => saveOverlayImage("both"));

[el.confRange, el.iouRange].forEach((input) => {
    input.addEventListener("input", () => {
        if (input === el.confRange) touchedConf = true;
        if (input === el.iouRange) touchedIou = true;
        renderControls();
        renderCorners();
    });
});

el.ctrlReset.addEventListener("click", resetControls);

[el.windowRange, el.levelRange].forEach((input) => {
    input.addEventListener("input", () => {
        applyWindowLevel();
        renderControls();
        renderCorners();
    });
});

// --- pointer tools on the stage ---------------------------------------

let drag = null;

/**
 * What a drag starting on this event does. Panning is what a reader reaches
 * for once the image is bigger than its frame, so past that point a plain drag
 * pans whichever tool is armed; window/level stays a Shift-drag away, and the
 * middle button pans at any zoom.
 */
function dragMode(e) {
    if (e.button === 1 || view.tool === "pan") return "pan";
    if (view.tool === "zoom") return "zoom";
    if (canPan() && !e.shiftKey) return "pan";
    return "wl";
}

el.stage.addEventListener("pointerdown", (e) => {
    if (!activeStudy() || (e.button !== 0 && e.button !== 1)) return;
    e.preventDefault();   // no middle-click autoscroll, no image drag ghost
    drag = {
        mode: dragMode(e),
        x: e.clientX,
        y: e.clientY,
        zoom: view.zoom,
        panX: view.panX,
        panY: view.panY,
        win: Number(el.windowRange.value),
        level: Number(el.levelRange.value),
    };
    el.stage.setPointerCapture(e.pointerId);
    el.stage.classList.add("is-dragging", `drag-${drag.mode}`);
});

el.stage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;

    if (drag.mode === "pan") {
        view.panX = drag.panX + dx;
        view.panY = drag.panY + dy;
        applyTransform();
    } else if (drag.mode === "zoom") {
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
        el.stage.classList.remove("is-dragging", "drag-pan", "drag-zoom", "drag-wl");
        if (el.stage.hasPointerCapture?.(e.pointerId)) el.stage.releasePointerCapture(e.pointerId);
    });
});

el.stage.addEventListener("wheel", (e) => {
    if (!activeStudy()) return;
    e.preventDefault();
    zoomAt(Math.exp(-e.deltaY / 500), e.clientX, e.clientY);
    renderCorners();
}, { passive: false });

window.addEventListener("resize", () => {
    fitFrames();
    applyTransform();   // the frames changed size, so the pan limits did too
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
