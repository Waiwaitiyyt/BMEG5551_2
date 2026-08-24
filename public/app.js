// Frontend logic for the Implant Loosening Detection demo (BMEG5552).
//
// API contract:
//
//   POST {API_BASE_URL}/predict
//   Content-Type: multipart/form-data, field name "file"
//
//   Response JSON:
//   {
//     "detections": [
//       { "label": "Implant", "confidence": 0.94, "box": [x1, y1, x2, y2] }
//     ],
//     "image_width": 1024,   // px — original image the box coords refer to
//     "image_height": 768    // px
//   }
//
//   box: [x1, y1, x2, y2] in pixel coordinates of the original uploaded image
//   (top-left / bottom-right corners).
//
// When the page is served by the Express gateway (server/ts, port 3000) the
// requests go to the same origin under /api, which Express proxies to the
// FastAPI inference server. Opening index.html straight from disk (file://)
// has no origin to inherit, so it falls back to FastAPI on localhost:8000 —
// that path needs the FastAPI CORS settings to allow it.

const API_BASE_URL =
    window.location.protocol === "file:" ? "http://localhost:8000" : "/api";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const imageStage = document.getElementById("image-stage");
const previewImage = document.getElementById("preview-image");
const overlayCanvas = document.getElementById("overlay-canvas");

const uploadMeta = document.getElementById("upload-meta");
const fileNameEl = document.getElementById("file-name");
const fileDimsEl = document.getElementById("file-dims");

const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");

const statusPill = document.getElementById("status-pill");
const resultsEmpty = document.getElementById("results-empty");
const resultsError = document.getElementById("results-error");
const errorMessage = document.getElementById("error-message");
const resultsContent = document.getElementById("results-content");
const detectionList = document.getElementById("detection-list");

let currentObjectUrl = null;
let currentFile = null;
let lastDetections = null;

const BOX_COLOR = "#ff5a5f";

// ---------------------------------------------------------------------
// Upload interactions
// ---------------------------------------------------------------------

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragover");
    });
});

["dragleave", "dragend"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragover");
    });
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer.files?.[0];
    if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
});

resetBtn.addEventListener("click", resetAll);
analyzeBtn.addEventListener("click", runDetection);

window.addEventListener("resize", () => {
    if (!imageStage.hidden) {
        resizeCanvasToImage();
        if (lastDetections) drawDetections(lastDetections);
    }
});

// ---------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("That file doesn't look like an image. Please choose a PNG, JPEG, or WEBP X-ray.");
        return;
    }

    clearResults();

    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentFile = file;
    currentObjectUrl = URL.createObjectURL(file);

    previewImage.onload = () => {
        dropzone.classList.add("has-image");
        dropzoneEmpty.hidden = true;
        imageStage.hidden = false;

        uploadMeta.hidden = false;
        fileNameEl.textContent = file.name;
        fileDimsEl.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight}px`;

        resizeCanvasToImage();

        analyzeBtn.disabled = false;
        resetBtn.disabled = false;
    };
    previewImage.src = currentObjectUrl;
}

function resetAll() {
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
    currentFile = null;
    lastDetections = null;

    fileInput.value = "";
    previewImage.src = "";
    dropzone.classList.remove("has-image");
    dropzoneEmpty.hidden = false;
    imageStage.hidden = true;
    uploadMeta.hidden = true;

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    analyzeBtn.disabled = true;
    resetBtn.disabled = true;

    clearResults();
}

// ---------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------

async function runDetection() {
    if (!currentFile) return;

    setStatus("loading");
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("is-loading");
    resetBtn.disabled = true;

    resultsEmpty.hidden = true;
    resultsError.hidden = true;
    resultsContent.hidden = true;

    try {
        const formData = new FormData();
        formData.append("file", currentFile);

        const response = await fetch(PREDICT_ENDPOINT, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const detections = Array.isArray(data.detections) ? data.detections : [];

        lastDetections = {
            items: detections,
            imageWidth: data.image_width || previewImage.naturalWidth,
            imageHeight: data.image_height || previewImage.naturalHeight,
        };

        renderDetections(lastDetections);
        drawDetections(lastDetections);
        setStatus(detections.length ? "success" : "idle", detections.length ? undefined : "No implant detected");
    } catch (err) {
        console.error(err);
        setStatus("error");
        showError(
            `Could not reach the detection server at ${PREDICT_ENDPOINT}. ` +
            `Make sure the FastAPI backend is running, and that API_BASE_URL in app.js points to it. ` +
            `(${err.message})`
        );
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove("is-loading");
        resetBtn.disabled = false;
    }
}

// ---------------------------------------------------------------------
// Canvas overlay
// ---------------------------------------------------------------------

function resizeCanvasToImage() {
    const width = previewImage.clientWidth;
    const height = previewImage.clientHeight;
    const dpr = window.devicePixelRatio || 1;

    overlayCanvas.style.width = `${width}px`;
    overlayCanvas.style.height = `${height}px`;
    overlayCanvas.width = Math.round(width * dpr);
    overlayCanvas.height = Math.round(height * dpr);

    const ctx = overlayCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawDetections(result) {
    const ctx = overlayCanvas.getContext("2d");
    const width = previewImage.clientWidth;
    const height = previewImage.clientHeight;
    ctx.clearRect(0, 0, width, height);

    const scaleX = width / result.imageWidth;
    const scaleY = height / result.imageHeight;

    result.items.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.box;
        const bx = x1 * scaleX;
        const by = y1 * scaleY;
        const bw = (x2 - x1) * scaleX;
        const bh = (y2 - y1) * scaleY;

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = BOX_COLOR;
        ctx.strokeRect(bx, by, bw, bh);

        const label = `${detection.label ?? "implant"} ${formatConfidence(detection.confidence)}`;
        ctx.font = "600 12px Inter, sans-serif";
        const textWidth = ctx.measureText(label).width;
        const tagHeight = 18;
        const tagY = by > tagHeight ? by - tagHeight : by;

        ctx.fillStyle = BOX_COLOR;
        ctx.fillRect(bx, tagY, textWidth + 12, tagHeight);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, bx + 6, tagY + 13);
    });
}

// ---------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------

function renderDetections(result) {
    detectionList.innerHTML = "";

    if (result.items.length === 0) {
        resultsEmpty.hidden = false;
        resultsEmpty.querySelector("p").textContent = "No implant was detected in this image.";
        resultsContent.hidden = true;
        return;
    }

    result.items.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.box;
        const confidencePct = Math.round((detection.confidence ?? 0) * 100);

        const card = document.createElement("div");
        card.className = "detection-card";
        card.innerHTML = `
            <div class="detection-card-head">
                <span class="detection-label">${escapeHtml(detection.label ?? "implant")}</span>
                <span class="confidence-value">${confidencePct}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width: ${confidencePct}%"></div>
            </div>
            <div class="detection-coords">
                <span><b>${Math.round(x1)}</b>x1</span>
                <span><b>${Math.round(y1)}</b>y1</span>
                <span><b>${Math.round(x2)}</b>x2</span>
                <span><b>${Math.round(y2)}</b>y2</span>
            </div>
        `;
        detectionList.appendChild(card);
    });

    resultsEmpty.hidden = true;
    resultsContent.hidden = false;
}

function clearResults() {
    lastDetections = null;
    resultsEmpty.hidden = false;
    resultsEmpty.querySelector("p").textContent =
        "Upload an X-ray and run detection to see the implant bounding box and confidence score here.";
    resultsError.hidden = true;
    resultsContent.hidden = true;
    detectionList.innerHTML = "";
    setStatus("idle");
}

function showError(message) {
    resultsEmpty.hidden = true;
    resultsContent.hidden = true;
    resultsError.hidden = false;
    errorMessage.textContent = message;
}

function setStatus(state, label) {
    const labels = {
        idle: "Idle",
        loading: "Analyzing…",
        success: "Detected",
        error: "Error",
    };
    statusPill.className = `status-pill status-${state}`;
    statusPill.textContent = label ?? labels[state];
}

function formatConfidence(value) {
    if (typeof value !== "number") return "";
    return `${Math.round(value * 100)}%`;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
