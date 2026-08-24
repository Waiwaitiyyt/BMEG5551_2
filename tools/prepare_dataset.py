"""Prepare an Ultralytics YOLO dataset from Label Studio YOLO exports.

The current Label Studio export names labels like ``<task-id>-loose_105.txt``
while the source image is ``loose (105).png``.  The matcher below handles that
case as well as ordinary same-stem image/label pairs.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_NUMBER = re.compile(r"(?i)(?P<prefix>loose)[ _-]?\(?\s*(?P<number>\d+)\s*\)?$")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=root / "datasets", help="Original image directory (searched recursively).")
    parser.add_argument("--labels-dir", type=Path, default=root / "label" / "labels", help="Directory containing Label Studio YOLO .txt files.")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset", help="Output directory to create.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction assigned to validation (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the train/val split.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before writing it.")
    return parser.parse_args()


def image_key(path: Path) -> tuple[str, str] | None:
    """Return a key for the project's ``loose (N)`` naming convention."""
    match = IMAGE_NUMBER.fullmatch(path.stem)
    if match:
        return match.group("prefix").lower(), match.group("number")
    return None


def find_images(images_dir: Path) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    by_stem: dict[str, Path] = {}
    by_key: dict[tuple[str, str], Path] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        by_stem.setdefault(path.stem.lower(), path)
        key = image_key(path)
        if key:
            by_key.setdefault(key, path)
    return by_stem, by_key


def match_image(label: Path, by_stem: dict[str, Path], by_key: dict[tuple[str, str], Path]) -> Path | None:
    direct = by_stem.get(label.stem.lower())
    if direct:
        return direct
    # Label Studio's exported name contains a suffix such as ``loose_105``.
    match = re.search(r"(?i)(loose)[ _-](\d+)$", label.stem)
    if match:
        return by_key.get((match.group(1).lower(), match.group(2)))
    return None


def validate_label(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric YOLO value") from exc
        if class_id != 0 or any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{path}:{line_number}: this dataset only accepts class id 0 and box values in [0, 1]")


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in the range [0, 1)")
    if not args.labels_dir.is_dir():
        raise SystemExit(f"Labels directory does not exist: {args.labels_dir}")
    if not args.images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {args.images_dir}")

    labels = sorted(p for p in args.labels_dir.glob("*.txt") if not p.name.endswith(":Zone.Identifier"))
    if not labels:
        raise SystemExit(f"No .txt labels found in {args.labels_dir}")
    by_stem, by_key = find_images(args.images_dir)

    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for label in labels:
        image = match_image(label, by_stem, by_key)
        if image is None:
            missing.append(label)
            continue
        validate_label(label)
        pairs.append((image, label))
    if missing:
        names = ", ".join(path.name for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise SystemExit(f"Could not find images for {len(missing)} label(s): {names}{suffix}")
    if not pairs:
        raise SystemExit("No image/label pairs found")

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {args.output_dir} (use --overwrite)")
        shutil.rmtree(args.output_dir)
    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.Random(args.seed).shuffle(pairs)
    val_count = round(len(pairs) * args.val_ratio)
    if len(pairs) > 1 and args.val_ratio > 0:
        val_count = max(1, min(len(pairs) - 1, val_count))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image, label in split_pairs:
            shutil.copy2(image, args.output_dir / "images" / split / image.name)
            shutil.copy2(label, args.output_dir / "labels" / split / f"{image.stem}.txt")

    data_yaml = args.output_dir / "data.yaml"
    data_yaml.write_text(
        "# Generated by tools/prepare_dataset.py\n"
        f"path: {args.output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Implant\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(pairs)} image/label pairs: {len(train_pairs)} train, {len(val_pairs)} val")
    print(f"Dataset YAML: {data_yaml}")


if __name__ == "__main__":
    main()
