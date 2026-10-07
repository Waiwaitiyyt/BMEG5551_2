"""Shared fixtures for the inference-server tests.

The API tests load the real checkpoints in weights/ through the app's lifespan
hook, once per session — that is the slow part (a few seconds on CPU), so every
test reuses the same client.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

PY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PY_DIR.parents[1]
DATASET_DIR = (
    REPO_ROOT
    / "datasets/tawsifurrahman/aseptic-loose-hip-implant-xray-database/versions/1/Data"
)


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    if not (PY_DIR / "weights" / "best.pt").is_file():
        pytest.skip("detector checkpoint weights/best.pt is not present")
    from server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def synthetic_xray() -> bytes:
    """A deterministic radiograph-like greyscale image: dark field, bright stem.

    It exercises decoding and the full model path without shipping patient
    data; tests that need a real X-ray use the dataset and skip without it.
    """
    rng = np.random.default_rng(5552)
    h, w = 331, 331
    yy, xx = np.mgrid[0:h, 0:w]
    field = 60 + 40 * (xx / w) + rng.normal(0, 6, (h, w))
    stem = (np.abs(xx - (130 + 0.25 * yy)) < 18) & (yy > 70)
    head = (xx - 190) ** 2 + (yy - 70) ** 2 < 45**2
    field[stem | head] = 235
    return png_bytes(Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), mode="L"))


def dataset_image(group: str, name: str) -> Path:
    path = DATASET_DIR / group / name
    if not path.is_file():
        pytest.skip(f"dataset image not available: {path.relative_to(REPO_ROOT)}")
    return path
