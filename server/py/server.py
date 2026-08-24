"""FastAPI inference server for implant localisation (BMEG5552 demo prototype).

Loads the trained YOLO checkpoint once at startup and exposes a single
inference endpoint used by the frontend:

    POST /predict   multipart/form-data, field "file"
    ->  {
          "detections": [
            {"label": "Implant", "confidence": 0.94, "box": [x1, y1, x2, y2]}
          ],
          "image_width": 1024,
          "image_height": 768
        }

Box coordinates are pixels in the original uploaded image (top-left /
bottom-right corners), which is exactly what public/app.js draws.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
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

# Loaded during the lifespan startup hook.
state: dict[str, Any] = {"model": None}


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int


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
    logger.info("Model ready. Classes: %s", model.names)
    yield
    state["model"] = None


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


@app.get("/health")
async def health() -> dict[str, Any]:
    model = state["model"]
    return {
        "status": "ok" if model is not None else "loading",
        "model_path": str(MODEL_PATH),
        "classes": list(model.names.values()) if model is not None else [],
        "conf_threshold": CONF_THRESHOLD,
        "imgsz": IMGSZ,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

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

    # X-rays are usually greyscale or 16-bit; YOLO expects 3-channel 8-bit RGB.
    image = image.convert("RGB")

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )

    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            if box.conf[0] > CONF_THRESHOLD:
                detections.append(
                    Detection(
                        label=names.get(class_id, str(class_id)),
                        confidence=round(float(box.conf[0]), 4),
                        box=[x1, y1, x2, y2],
                    )
                )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    logger.info("%s -> %d detection(s)", file.filename, len(detections))

    return PredictResponse(
        detections=detections,
        image_width=image.width,
        image_height=image.height,
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
