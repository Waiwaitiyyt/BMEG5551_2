"""HTTP tests for /health, /predict and /classify against the real checkpoints."""

from __future__ import annotations

import pytest

from conftest import dataset_image


def upload(data: bytes, name: str = "study.png", mime: str = "image/png"):
    return {"file": (name, data, mime)}


# --- /health ---------------------------------------------------------------


def test_health_reports_loaded_models(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["classes"] == ["Implant"]
    assert 0 < body["conf_threshold"] <= 1
    assert 0 < body["iou_threshold"] <= 1
    cls = body["classifier"]
    assert cls["available"] is True
    assert cls["classes"] == ["Control", "Loose"]
    assert 0 < cls["threshold"] < 1


# --- /predict --------------------------------------------------------------


def test_predict_returns_the_documented_schema(client, synthetic_xray):
    res = client.post("/predict", files=upload(synthetic_xray))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["image_width"] == 331 and body["image_height"] == 331
    assert body["model"] == "best.pt"
    assert body["classes"] == ["Implant"]
    assert body["inference_ms"] >= 0
    for det in body["detections"]:
        assert det["label"] == "Implant"
        assert det["confidence"] >= body["conf_threshold"]
        x1, y1, x2, y2 = det["box"]
        assert 0 <= x1 < x2 <= 331 and 0 <= y1 < y2 <= 331
    confs = [d["confidence"] for d in body["detections"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_echoes_per_request_settings(client, synthetic_xray):
    res = client.post(
        "/predict", files=upload(synthetic_xray), data={"conf": "0.3", "iou": "0.5", "imgsz": "100"}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["conf_threshold"] == 0.3
    assert body["iou_threshold"] == 0.5
    assert body["imgsz"] == 128  # rounded up to the stride of 32


def test_predict_settings_do_not_leak_between_requests(client, synthetic_xray):
    client.post("/predict", files=upload(synthetic_xray), data={"conf": "0.1"})
    default = client.get("/health").json()["conf_threshold"]
    res = client.post("/predict", files=upload(synthetic_xray))
    assert res.json()["conf_threshold"] == default


@pytest.mark.parametrize(
    "data",
    [{"conf": "0"}, {"conf": "1.5"}, {"iou": "-0.2"}, {"imgsz": "32"}, {"imgsz": "4000"}],
)
def test_predict_rejects_bad_settings(client, synthetic_xray, data):
    res = client.post("/predict", files=upload(synthetic_xray), data=data)
    assert res.status_code == 422


def test_predict_rejects_empty_upload(client):
    res = client.post("/predict", files=upload(b""))
    assert res.status_code == 400
    assert "empty" in res.json()["detail"]


def test_predict_rejects_non_images(client):
    res = client.post("/predict", files=upload(b"definitely not an image", "notes.txt", "text/plain"))
    assert res.status_code == 400
    assert "decode" in res.json()["detail"]


def test_predict_requires_a_file(client):
    assert client.post("/predict").status_code == 422


# --- /classify -------------------------------------------------------------


def test_classify_returns_verdict_and_heatmap(client, synthetic_xray):
    res = client.post("/classify", files=upload(synthetic_xray))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["label"] in body["classes"] == ["Control", "Loose"]
    assert 0 <= body["probability"] <= 1
    assert body["confidence"] >= 0.5
    assert (body["label"] == "Loose") == (body["probability"] >= body["threshold"])
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert len(body["heatmap_box"]) == 4
    assert body["arch"] == "resnet50"
    assert (body["image_width"], body["image_height"]) == (331, 331)


def test_classify_without_heatmap(client, synthetic_xray):
    res = client.post("/classify", files=upload(synthetic_xray), data={"heatmap": "false"})
    assert res.status_code == 200
    body = res.json()
    assert body["heatmap"] is None
    assert body["heatmap_box"] is None


def test_classify_threshold_override_flips_the_decision(client, synthetic_xray):
    low = client.post("/classify", files=upload(synthetic_xray), data={"threshold": "0.0001", "heatmap": "false"})
    high = client.post("/classify", files=upload(synthetic_xray), data={"threshold": "0.9999", "heatmap": "false"})
    assert low.json()["threshold"] == 0.0001
    assert high.json()["threshold"] == 0.9999
    p = low.json()["probability"]
    assert low.json()["label"] == ("Loose" if p >= 0.0001 else "Control")
    assert high.json()["label"] == ("Loose" if p >= 0.9999 else "Control")


@pytest.mark.parametrize("threshold", ["0", "1", "1.2"])
def test_classify_rejects_degenerate_thresholds(client, synthetic_xray, threshold):
    res = client.post("/classify", files=upload(synthetic_xray), data={"threshold": threshold})
    assert res.status_code == 422


def test_classify_accepts_rgb_jpeg(client, synthetic_xray):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(synthetic_xray)).convert("RGB").save(buf, format="JPEG")
    res = client.post("/classify", files=upload(buf.getvalue(), "study.jpg", "image/jpeg"))
    assert res.status_code == 200


# --- real X-rays (skipped when the gitignored dataset is absent, e.g. in CI) --


@pytest.mark.parametrize(
    ("group", "name", "label"),
    [("Loose", "loose (1).png", "Loose"), ("Control", "control (12).png", "Control")],
)
def test_reference_cases_from_the_dataset(client, group, name, label):
    path = dataset_image(group, name)
    data = path.read_bytes()

    det = client.post("/predict", files=upload(data, name)).json()
    assert det["detections"], "expected the implant to be localised"
    assert det["detections"][0]["confidence"] >= 0.5

    cls = client.post("/classify", files=upload(data, name), data={"heatmap": "false"}).json()
    assert cls["label"] == label
