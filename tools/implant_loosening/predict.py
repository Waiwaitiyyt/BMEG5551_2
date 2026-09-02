"""Classify implant X-rays with a trained checkpoint, optionally with Grad-CAM.

    python predict.py --checkpoint runs/baseline --image scan.png --cam out/
    python predict.py --checkpoint runs/baseline/fold1/best.pt --image folder/

Passing a run directory ensembles every ``fold*/best.pt`` inside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data import CLASS_NAMES, IMAGE_EXTENSIONS, build_transforms
from model import build_model, last_conv_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt file or a run directory.")
    parser.add_argument("--image", type=Path, required=True, help="Image file or a directory of images.")
    parser.add_argument("--cam", type=Path, default=None, help="Directory to write Grad-CAM overlays into.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold stored in the checkpoint.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_models(path: Path, device: torch.device) -> tuple[list[torch.nn.Module], dict]:
    files = sorted(path.glob("fold*/best.pt")) or ([path] if path.is_file() else sorted(path.glob("*/best.pt")))
    if not files:
        raise SystemExit(f"No checkpoint found at {path}")
    models, meta = [], {}
    thresholds = []
    for file in files:
        checkpoint = torch.load(file, map_location=device, weights_only=False)
        model = build_model(checkpoint["arch"], pretrained=False, dropout=checkpoint.get("dropout", 0.3))
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model.to(device).eval())
        thresholds.append(checkpoint.get("threshold", 0.5))
        meta = checkpoint
    meta["threshold"] = sum(thresholds) / len(thresholds)
    print(f"[model] {len(models)} checkpoint(s) from {path} (arch={meta['arch']}, thr={meta['threshold']:.3f})")
    return models, meta


def gather_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {path}")
    return files


def grad_cam(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Class activation map for the loose logit, from the deepest conv layer."""
    activations: dict[str, torch.Tensor] = {}
    layer = last_conv_layer(model)
    handles = [
        layer.register_forward_hook(lambda _m, _i, output: activations.__setitem__("value", output)),
        layer.register_full_backward_hook(lambda _m, _gi, grad_output: activations.__setitem__("grad", grad_output[0])),
    ]
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            model(tensor).squeeze().backward()
        weights = activations["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach()
    finally:
        for handle in handles:
            handle.remove()


def save_overlay(image: Image.Image, cam: torch.Tensor, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[cam] matplotlib is not installed, skipping overlay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    size = cam.shape[-1]
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image.resize((size, size)), cmap="gray")
    axis.imshow(cam.cpu().numpy(), cmap="jet", alpha=0.4)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models, meta = load_models(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    transform = build_transforms(meta.get("img_size", 320), train=False)

    for file in gather_images(args.image):
        image = Image.open(file).convert("L")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = [torch.sigmoid(model(tensor).squeeze()).item() for model in models]
            flipped = [torch.sigmoid(model(torch.flip(tensor, dims=[3])).squeeze()).item() for model in models]
        probability = (sum(probs) + sum(flipped)) / (2 * len(models))
        label = CLASS_NAMES[int(probability >= threshold)]
        print(f"{file.name}: {label}  p(loose)={probability:.3f}  (thr={threshold:.3f})")
        if args.cam is not None:
            save_overlay(image, grad_cam(models[0], tensor), args.cam / f"{file.stem}_cam.png")


if __name__ == "__main__":
    main()
