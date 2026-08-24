"""Train an Ultralytics YOLO detector on the prepared implant dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "dataset" / "data.yaml", help="Ultralytics dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model checkpoint or model YAML (default: yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1", help="Batch size; -1 lets Ultralytics auto-select it.")
    parser.add_argument("--device", default=None, help="CUDA device (0), multiple devices (0,1), or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=root / "runs" / "detect")
    parser.add_argument("--name", default="implant")
    parser.add_argument("--resume", action="store_true", help="Resume training from --model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}\nRun tools/prepare_dataset.py first.")
    try:
        batch: int | str = int(args.batch)
    except ValueError:
        batch = args.batch
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
    )
    run_dir = args.project / args.name
    print(f"Training complete. Results: {run_dir}")
    print(f"Best checkpoint: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
