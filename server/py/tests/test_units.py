"""Pure-function tests: parameter validation and classifier helpers."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi import HTTPException

import classifier
import server


class TestValidateRatio:
    def test_none_falls_back_to_default(self):
        assert server._validate_ratio("conf", None, 0.5) == 0.5

    @pytest.mark.parametrize("value", [0.01, 0.5, 1.0])
    def test_accepts_values_in_range(self, value):
        assert server._validate_ratio("conf", value, 0.5) == value

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.01, 5])
    def test_rejects_values_out_of_range(self, value):
        with pytest.raises(HTTPException) as exc:
            server._validate_ratio("iou", value, 0.7)
        assert exc.value.status_code == 422
        assert "iou" in exc.value.detail


class TestValidateImgsz:
    def test_none_falls_back_to_default(self):
        assert server._validate_imgsz(None) == server.IMGSZ

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(64, 64), (65, 96), (100, 128), (640, 640), (641, 672), (1536, 1536)],
    )
    def test_rounds_up_to_stride(self, value, expected):
        assert server._validate_imgsz(value) == expected

    @pytest.mark.parametrize("value", [0, 63, 1537, 4096])
    def test_rejects_values_out_of_range(self, value):
        with pytest.raises(HTTPException) as exc:
            server._validate_imgsz(value)
        assert exc.value.status_code == 422


class TestColourise:
    def test_shape_and_dtype(self):
        cam = np.linspace(0, 1, 64).reshape(8, 8)
        rgba = classifier._colourise(cam)
        assert rgba.shape == (8, 8, 4)
        assert rgba.dtype == np.uint8

    def test_cold_regions_are_transparent_and_hot_regions_are_red(self):
        rgba = classifier._colourise(np.array([[0.0, 0.35, 1.0]]))
        assert rgba[0, 0, 3] == 0
        assert rgba[0, 1, 3] == 0
        assert tuple(rgba[0, 2, :3]) == (255, 0, 0)
        assert rgba[0, 2, 3] == 200

    def test_alpha_is_monotonic(self):
        alpha = classifier._colourise(np.linspace(0, 1, 50)[None, :])[0, :, 3]
        assert np.all(np.diff(alpha.astype(int)) >= 0)


def test_png_to_data_uri_round_trips():
    payload = b"\x89PNG\r\n\x1a\nnot-really-a-png"
    uri = classifier.png_to_data_uri(payload)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == payload


def test_build_backbone_rejects_unknown_arch():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        classifier._build_backbone("vgg16", 0.3)


def test_build_backbone_has_single_logit_head():
    import torch

    model = classifier._build_backbone("resnet18", 0.3).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1)
