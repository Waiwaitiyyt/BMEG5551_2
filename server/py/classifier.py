"""ResNet50 loosening classifier with Grad-CAM, for the inference server.

Self-contained on purpose: the server must run without `tools/` on the path.
The checkpoint is produced by `tools/implant_loosening/train.py` and carries the
architecture, input size, and the decision threshold tuned on validation data,
so nothing here has to be kept in sync by hand.

Deployed weights: fold1 of the 5-fold run (highest validation AUC, 0.993;
held-out test AUC 0.988, accuracy 0.976, sensitivity 0.955, specificity 1.000).
"""

from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The eval transform resizes the short side by this factor before centre
# cropping; it must match tools/implant_loosening/data.build_transforms.
RESIZE_FACTOR = 1.14

_ARCH_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "densenet121": models.densenet121,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
    "convnext_tiny": models.convnext_tiny,
}
_ARCH_HEADS = {
    "resnet18": "fc",
    "resnet50": "fc",
    "densenet121": "classifier",
    "efficientnet_b0": "classifier",
    "efficientnet_b3": "classifier",
    "convnext_tiny": "classifier",
}


@dataclass
class ClassificationResult:
    label: str
    probability: float          # p(Loose)
    threshold: float
    confidence: float           # probability of the reported label
    cam_png: bytes | None       # RGBA overlay, same pixel size as the upload
    cam_box: list[float] | None  # [x1, y1, x2, y2] region the model actually saw
    inference_ms: float


def _build_backbone(arch: str, dropout: float) -> nn.Module:
    """Rebuild the training-time architecture: backbone + dropout + 1 logit."""
    if arch not in _ARCH_BUILDERS:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch!r}")
    model = _ARCH_BUILDERS[arch](weights=None)
    head_name = _ARCH_HEADS[arch]
    head = getattr(model, head_name)
    if isinstance(head, nn.Sequential):
        index = max(i for i, layer in enumerate(head) if isinstance(layer, nn.Linear))
        head[index] = nn.Sequential(nn.Dropout(dropout), nn.Linear(head[index].in_features, 1))
    else:
        setattr(model, head_name, nn.Sequential(nn.Dropout(dropout), nn.Linear(head.in_features, 1)))
    return model


def _cam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last spatial block, whose activations Grad-CAM weights by their gradients.

    Hooking the block output rather than the final Conv2d matters: it is taken
    after the residual addition and ReLU, which is what the pooling layer -- and
    therefore the logit -- actually consumes.
    """
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        return model.features
    conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv:
        raise ValueError(f"No Grad-CAM target layer found for {arch}")
    return conv[-1]


def _colourise(cam: np.ndarray) -> np.ndarray:
    """Blue-cyan-green-yellow-red ramp with activation-proportional alpha.

    Alpha rises from zero so cold regions stay fully transparent and the
    radiograph underneath remains readable; only genuine activation is painted.
    """
    anchors = np.array(
        [
            [0.0, 0, 0, 255],
            [0.25, 0, 255, 255],
            [0.5, 0, 255, 0],
            [0.75, 255, 255, 0],
            [1.0, 255, 0, 0],
        ]
    )
    rgb = np.stack(
        [np.interp(cam, anchors[:, 0], anchors[:, channel + 1]) for channel in range(3)], axis=-1
    )
    alpha = np.clip((cam - 0.35) / 0.65, 0.0, 1.0) ** 0.8 * 200.0
    return np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)


class LooseningClassifier:
    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint: dict[str, Any] = torch.load(self.path, map_location="cpu", weights_only=False)

        self.arch: str = checkpoint["arch"]
        self.img_size: int = int(checkpoint.get("img_size", 320))
        self.class_names: list[str] = list(checkpoint.get("class_names", ["Control", "Loose"]))
        self.threshold: float = float(checkpoint.get("threshold", 0.5))
        self.val_metrics: dict[str, Any] = checkpoint.get("val_metrics", {})

        self.model = _build_backbone(self.arch, float(checkpoint.get("dropout", 0.3)))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.target_layer = _cam_target_layer(self.model, self.arch)

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.img_size * RESIZE_FACTOR)),
                transforms.CenterCrop(self.img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        # Grad-CAM registers hooks and runs a backward pass, so two concurrent
        # requests would interleave their activations. FastAPI dispatches sync
        # handlers onto a threadpool, hence the lock.
        self._cam_lock = threading.Lock()

    # -- geometry ----------------------------------------------------------
    def crop_box(self, width: int, height: int) -> list[float]:
        """Region of the original image the centre crop keeps, in source pixels."""
        scale = int(self.img_size * RESIZE_FACTOR) / min(width, height)
        side = self.img_size / scale
        return [
            round((width - side) / 2, 2),
            round((height - side) / 2, 2),
            round((width + side) / 2, 2),
            round((height + side) / 2, 2),
        ]

    # -- inference ---------------------------------------------------------
    def _forward_probability(self, tensor: torch.Tensor, tta: bool) -> float:
        with torch.no_grad():
            probability = torch.sigmoid(self.model(tensor).squeeze()).item()
            if tta:
                flipped = torch.sigmoid(self.model(torch.flip(tensor, dims=[3])).squeeze()).item()
                probability = (probability + flipped) / 2
        return probability

    def _grad_cam(self, tensor: torch.Tensor) -> np.ndarray:
        captured: dict[str, torch.Tensor] = {}
        handles = [
            self.target_layer.register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("activation", output)
            ),
            self.target_layer.register_full_backward_hook(
                lambda _m, _gi, grad_output: captured.__setitem__("gradient", grad_output[0])
            ),
        ]
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                # Gradient of the Loose logit: positive activations are evidence
                # for loosening, which is the direction worth showing.
                self.model(tensor).squeeze().backward()
            weights = captured["gradient"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * captured["activation"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
        span = float(cam.max() - cam.min())
        # A flat map means the layer produced no positive evidence; returning
        # zeros keeps the overlay empty instead of amplifying numerical noise.
        return (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

    def _cam_to_png(self, cam: np.ndarray, image: Image.Image) -> bytes:
        """Paint the CAM into the crop rectangle of a full-size RGBA overlay.

        The frontend can then stretch the PNG over the image at natural size
        with no coordinate maths, and the transparent border makes it obvious
        that the centre crop is all the model looked at.
        """
        x1, y1, x2, y2 = (int(round(v)) for v in self.crop_box(image.width, image.height))
        crop_width = max(x2 - x1, 1)
        crop_height = max(y2 - y1, 1)
        heat = Image.fromarray(_colourise(cam), mode="RGBA").resize(
            (crop_width, crop_height), Image.BICUBIC
        )
        overlay = Image.new("RGBA", (image.width, image.height), (0, 0, 0, 0))
        overlay.paste(heat, (x1, y1))
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def classify(
        self, image: Image.Image, want_cam: bool = True, threshold: float | None = None, tta: bool = True
    ) -> ClassificationResult:
        import time

        effective_threshold = self.threshold if threshold is None else float(threshold)
        grayscale = image.convert("L")
        tensor = self.transform(grayscale).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        probability = self._forward_probability(tensor, tta=tta)
        cam_png: bytes | None = None
        if want_cam:
            with self._cam_lock:
                cam = self._grad_cam(tensor)
            cam_png = self._cam_to_png(cam, image)
        inference_ms = (time.perf_counter() - started) * 1000.0

        is_loose = probability >= effective_threshold
        return ClassificationResult(
            label=self.class_names[1] if is_loose else self.class_names[0],
            probability=round(probability, 4),
            threshold=round(effective_threshold, 4),
            confidence=round(probability if is_loose else 1.0 - probability, 4),
            cam_png=cam_png,
            cam_box=self.crop_box(image.width, image.height) if want_cam else None,
            inference_ms=round(inference_ms, 1),
        )

    def info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.path),
            "model_name": self.path.name,
            "arch": self.arch,
            "img_size": self.img_size,
            "classes": self.class_names,
            "threshold": round(self.threshold, 4),
            "device": str(self.device),
            "val_metrics": self.val_metrics,
        }


def png_to_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
