"""FastAPI inference server for the BMEG5552 implant workstation.

Two models are loaded once at startup and serve the frontend's three features:

    localisation   YOLO detector    -> POST /predict    (bounding box)
    prediction     ResNet50 CNN     -> POST /classify   (loose vs. well fixed)
    heat map       Grad-CAM         -> POST /classify   (returned with the call)

Prediction and its heat map share one endpoint because the viewer's overlay
switch (Box / Heatmap / Both / Off) is a display toggle, not a new run: one
upload yields the probability and the saliency map together, and flipping the
switch never costs another request.

    GET  /health
    ->  {
          "status": "ok",
          "model_path": "server/py/weights/best.pt",
          "model_name": "best.pt",
          "classes": ["Implant"],
          "conf_threshold": 0.5,
          "iou_threshold": 0.7,
          "imgsz": 640,
          "device": "cpu",
          "classifier": {
            "available": true,
            "model_name": "classifier.pt",
            "arch": "resnet50",
            "classes": ["Control", "Loose"],
            "threshold": 0.6132,
            "img_size": 320,
            "val_metrics": {...}
          }
        }

    POST /classify  multipart/form-data
        file       the image                              (required)
        threshold  per-request decision threshold         (optional, 0 < t < 1)
        heatmap    compute the Grad-CAM overlay           (optional, default true)
    ->  {
          "label": "Loose",
          "probability": 0.9971,          # p(Loose), threshold free
          "confidence": 0.9971,           # probability of the reported label
          "threshold": 0.6132,
          "classes": ["Control", "Loose"],
          "heatmap": "data:image/png;base64,...",   # null when heatmap=false
          "heatmap_box": [20.0, 20.0, 311.0, 311.0],
          "image_width": 331,
          "image_height": 331,
          "model": "classifier.pt",
          "arch": "resnet50",
          "imgsz": 320,
          "inference_ms": 498.2
        }

    The heat map is an RGBA PNG the size of the upload, transparent outside the
    centre crop the model actually saw, so the frontend can stretch it over the
    image at natural size with no coordinate maths. `heatmap_box` reports that
    crop in source pixels for anyone who wants to draw its outline.

    POST /predict   multipart/form-data
        file    the image                          (required)
        conf    per-request confidence threshold   (optional, 0 < conf <= 1)
        iou     per-request NMS IoU threshold      (optional, 0 < iou  <= 1)
        imgsz   per-request inference size         (optional, 64..1536)
    ->  {
          "detections": [
            {"label": "Implant", "confidence": 0.94, "box": [x1, y1, x2, y2]}
          ],
          "image_width": 1024,
          "image_height": 768,
          "conf_threshold": 0.5,
          "iou_threshold": 0.7,
          "imgsz": 640,
          "model": "best.pt",
          "classes": ["Implant"],
          "inference_ms": 41.7
        }

The three optional fields back the workstation's "Inference controls" sliders:
they override the process-wide defaults for one request only, and the response
echoes the values actually used so the viewer's corner annotation and the run
metadata always describe the run that produced the boxes on screen.

Box coordinates are pixels in the original uploaded image (top-left /
bottom-right corners), which is exactly what public/app.js draws.
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

logger = logging.getLogger("implant.server")

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# --- Configuration (override with environment variables) ------------------
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "weights" / "best.pt"))
CLASSIFIER_PATH = Path(
    os.environ.get("CLASSIFIER_PATH", BASE_DIR / "weights" / "classifier.pt")
)
CONF_THRESHOLD = _env_float("CONF_THRESHOLD", 0.5)
IOU_THRESHOLD = _env_float("IOU_THRESHOLD", 0.7)
IMGSZ = _env_int("IMGSZ", 640)
DEVICE = os.environ.get("DEVICE") or None  # e.g. "0", "0,1", "cpu"
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# Bounds for the per-request overrides. Ultralytics wants an imgsz that is a
# multiple of 32; anything else is silently rounded up, so round it here and
# report the value actually used.
MIN_IMGSZ = 64
MAX_IMGSZ = 1536
IMGSZ_STRIDE = 32

# Loaded during the lifespan startup hook.
state: dict[str, Any] = {"model": None, "classifier": None}


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]


class ClassifyResponse(BaseModel):
    label: str
    probability: float
    confidence: float
    threshold: float
    classes: list[str]
    heatmap: str | None
    heatmap_box: list[float] | None
    image_width: int
    image_height: int
    model: str
    arch: str
    imgsz: int
    inference_ms: float


class PredictResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
    # Echo of the settings this run actually used, so the frontend never has to
    # assume its sliders and the server agree.
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    model: str
    classes: list[str]
    inference_ms: float


def _validate_ratio(name: str, value: float | None, default: float) -> float:
    if value is None:
        return default
    if not 0.0 < value <= 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be greater than 0 and at most 1 (got {value}).",
        )
    return float(value)


def _validate_imgsz(value: int | None) -> int:
    if value is None:
        return IMGSZ
    if not MIN_IMGSZ <= value <= MAX_IMGSZ:
        raise HTTPException(
            status_code=422,
            detail=f"imgsz must be between {MIN_IMGSZ} and {MAX_IMGSZ} (got {value}).",
        )
    # Round up to the model's stride so the echoed value matches reality.
    return -(-int(value) // IMGSZ_STRIDE) * IMGSZ_STRIDE


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Train a model with tools/train.py and copy runs/detect/implant/weights/best.pt "
            "to server/py/weights/best.pt, or set MODEL_PATH to its location."
        )

    from ultralytics import YOLO  # imported lazily; pulls in torch

    logger.info("Loading YOLO checkpoint from %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    if DEVICE:
        model.to(DEVICE)
    state["model"] = model
    logger.info("Detector ready. Classes: %s", model.names)

    # The classifier is optional: a missing checkpoint degrades /classify to a
    # 503 but leaves localisation working, which keeps the demo usable on a
    # machine where only the YOLO weights were copied over.
    if CLASSIFIER_PATH.is_file():
        from classifier import LooseningClassifier

        logger.info("Loading loosening classifier from %s", CLASSIFIER_PATH)
        classifier = LooseningClassifier(CLASSIFIER_PATH, device=DEVICE)
        state["classifier"] = classifier
        logger.info(
            "Classifier ready. arch=%s classes=%s threshold=%.4f device=%s",
            classifier.arch,
            classifier.class_names,
            classifier.threshold,
            classifier.device,
        )
    else:
        logger.warning(
            "Classifier checkpoint not found at %s - /classify will return 503. "
            "Copy resnet50_models/fold1/best.pt there, or set CLASSIFIER_PATH.",
            CLASSIFIER_PATH,
        )

    yield
    state["model"] = None
    state["classifier"] = None


app = FastAPI(
    title="Implant Locator Inference API",
    description="YOLO-based implant localisation for the BMEG5552 demo prototype.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def _read_image(file: UploadFile) -> Image.Image:
    """Decode an upload into a PIL image, or raise the right 4xx for the frontend."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the upload as an image. Use PNG, JPEG, or WEBP.",
        ) from exc
    return image


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness plus the defaults the frontend seeds its controls from."""
    model = state["model"]
    classifier = state["classifier"]
    return {
        "status": "ok" if model is not None else "loading",
        "model_path": str(MODEL_PATH),
        "model_name": MODEL_PATH.name,
        "classes": list(model.names.values()) if model is not None else [],
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": IMGSZ,
        "device": DEVICE or "auto",
        "classifier": (
            {"available": True, **classifier.info()}
            if classifier is not None
            else {"available": False, "model_path": str(CLASSIFIER_PATH)}
        ),
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    file: UploadFile = File(...),
    threshold: float | None = Form(None),
    heatmap: bool = Form(True),
) -> ClassifyResponse:
    """Loose vs. well-fixed, with the Grad-CAM overlay that explains the call."""
    classifier = state["classifier"]
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Classifier checkpoint not loaded ({CLASSIFIER_PATH}). "
                "Copy resnet50_models/fold1/best.pt to server/py/weights/classifier.pt."
            ),
        )
    # A threshold of exactly 0 or 1 collapses the decision to one class, which
    # is never what a slider drag means.
    if threshold is not None and not 0.0 < threshold < 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"threshold must be strictly between 0 and 1 (got {threshold}).",
        )

    image = await _read_image(file)

    # Grad-CAM needs a backward pass, so this is real work, not a lookup; run it
    # off the event loop to keep /health and concurrent uploads responsive.
    from anyio import to_thread
    from classifier import png_to_data_uri

    result = await to_thread.run_sync(
        lambda: classifier.classify(image, want_cam=heatmap, threshold=threshold)
    )

    logger.info(
        "%s -> %s p=%.4f (thr=%.3f, heatmap=%s) in %.1f ms",
        file.filename,
        result.label,
        result.probability,
        result.threshold,
        heatmap,
        result.inference_ms,
    )

    return ClassifyResponse(
        label=result.label,
        probability=result.probability,
        confidence=result.confidence,
        threshold=result.threshold,
        classes=classifier.class_names,
        heatmap=png_to_data_uri(result.cam_png) if result.cam_png else None,
        heatmap_box=result.cam_box,
        image_width=image.width,
        image_height=image.height,
        model=CLASSIFIER_PATH.name,
        arch=classifier.arch,
        imgsz=classifier.img_size,
        inference_ms=result.inference_ms,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
    imgsz: int | None = Form(None),
) -> PredictResponse:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Per-request overrides for the workstation's inference sliders. Each one
    # applies to this call only — the process-wide defaults are never mutated,
    # so concurrent requests cannot see each other's settings.
    eff_conf = _validate_ratio("conf", conf, CONF_THRESHOLD)
    eff_iou = _validate_ratio("iou", iou, IOU_THRESHOLD)
    eff_imgsz = _validate_imgsz(imgsz)

    image = await _read_image(file)

    # X-rays are usually greyscale or 16-bit; YOLO expects 3-channel 8-bit RGB.
    image = image.convert("RGB")

    started = time.perf_counter()
    results = model.predict(
        source=image,
        conf=eff_conf,
        iou=eff_iou,
        imgsz=eff_imgsz,
        device=DEVICE,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - started) * 1000.0

    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            # `conf=` above already filters, but keep the guard against this
            # request's threshold rather than the process default — otherwise a
            # slider set below CONF_THRESHOLD would drop the very detections it
            # was lowered to reveal.
            if float(box.conf[0]) >= eff_conf:
                detections.append(
                    Detection(
                        label=names.get(class_id, str(class_id)),
                        confidence=round(float(box.conf[0]), 4),
                        box=[x1, y1, x2, y2],
                    )
                )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    logger.info(
        "%s -> %d detection(s) in %.1f ms (conf=%.2f iou=%.2f imgsz=%d)",
        file.filename,
        len(detections),
        inference_ms,
        eff_conf,
        eff_iou,
        eff_imgsz,
    )

    return PredictResponse(
        detections=detections,
        image_width=image.width,
        image_height=image.height,
        conf_threshold=eff_conf,
        iou_threshold=eff_iou,
        imgsz=eff_imgsz,
        model=MODEL_PATH.name,
        classes=list(model.names.values()),
        inference_ms=round(inference_ms, 1),
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        reload=os.environ.get("RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
