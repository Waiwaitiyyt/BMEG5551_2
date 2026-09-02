"""Train a CNN to tell an aseptically loose hip implant from a well-fixed one.

Designed for the small (206 image) Kaggle database, so the defaults favour
regularisation and honest evaluation over raw throughput:

* stratified splits with a held-out test set that never touches training,
* 5-fold cross-validation (``--folds 5``) because a single 20-image validation
  split on this dataset swings by ~10 accuracy points between seeds,
* a frozen-backbone warm-up followed by full fine-tuning at a lower LR,
* model selection on validation AUC, with the decision threshold tuned on
  validation and then applied unchanged to the test set.

Examples
--------
    python train.py --data-root ../../datasets --folds 5 --epochs 40
    python train.py --data-root /content/data --arch efficientnet_b0 --img-size 320
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import CLASS_NAMES, Sample, XrayDataset, build_transforms, find_data_root, holdout_split, list_samples, stratified_folds
from engine import predict, train_one_epoch
from metrics import best_threshold, compute_metrics
from model import build_model, head_parameters, set_backbone_trainable


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=here.parent.parent / "datasets",
                        help="Directory containing Control/ and Loose/ (searched recursively).")
    parser.add_argument("--out-dir", type=Path, default=here / "runs" / "baseline",
                        help="Where checkpoints, metrics and curves are written.")
    parser.add_argument("--arch", default="resnet50",
                        help="resnet18 | resnet50 | densenet121 | efficientnet_b0 | efficientnet_b3 | convnext_tiny")
    parser.add_argument("--img-size", type=int, default=320, help="Source scans are 331x331.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Epochs training the head alone before unfreezing the backbone.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="Backbone LR after unfreezing.")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val AUC.")
    parser.add_argument("--folds", type=int, default=5,
                        help="K-fold cross-validation over the train+val pool; 0 uses a single holdout split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Only used when --folds 0.")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Held out before any training or fold splitting; 0 disables the test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable the class-balanced sampler (dataset is 94 control / 112 loose).")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip test-time augmentation.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available).")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(samples: list[Sample], args: argparse.Namespace, train: bool) -> DataLoader:
    dataset = XrayDataset(samples, build_transforms(args.img_size, train=train))
    sampler = None
    if train and not args.no_balance:
        counts = [sum(1 for s in samples if s.label == label) for label in (0, 1)]
        weights = [1.0 / counts[s.label] for s in samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(samples) > args.batch_size,
    )


class SmoothedBCE(nn.Module):
    """BCE-with-logits plus label smoothing; radiographic labels are not perfect."""

    def __init__(self, smoothing: float, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(logits, targets)


def train_fold(
    fold_name: str,
    train_samples: list[Sample],
    val_samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_samples, args, train=True)
    val_loader = make_loader(val_samples, args, train=False)

    model = build_model(args.arch, pretrained=True, dropout=args.dropout).to(device)
    criterion = SmoothedBCE(args.label_smoothing)
    head_ids = {id(p) for p in head_parameters(model)}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
            {"params": head_parameters(model), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" and not args.no_amp else None
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.freeze_epochs, 1) * steps_per_epoch, eta_min=1e-6
    )

    set_backbone_trainable(model, trainable=args.freeze_epochs <= 0)
    history: list[dict] = []
    best = {"auc": -1.0, "epoch": -1, "state": None}
    since_improved = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            print(f"  [{fold_name}] backbone unfrozen at epoch {epoch}")
        use_scheduler = scheduler if epoch > args.freeze_epochs else None
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_scheduler)
        probs, labels, val_loss = predict(model, val_loader, device, criterion, tta=not args.no_tta)
        metrics = compute_metrics(labels, probs, threshold=0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(metrics.accuracy, 4),
                "val_auc": round(metrics.auc, 4),
                "val_sens": round(metrics.sensitivity, 4),
                "val_spec": round(metrics.specificity, 4),
            }
        )
        print(f"  [{fold_name}] epoch {epoch:>3}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} {metrics.summary()}")

        # AUC is threshold free, so it is the fairest selection signal on a
        # validation split this small; accuracy breaks ties.
        score = metrics.auc if metrics.auc == metrics.auc else metrics.accuracy
        if score > best["auc"] + 1e-6:
            best = {"auc": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= args.patience:
                print(f"  [{fold_name}] early stop at epoch {epoch} (best epoch {best['epoch']})")
                break

    model.load_state_dict(best["state"])
    val_probs, val_labels, _ = predict(model, val_loader, device, tta=not args.no_tta)
    threshold = best_threshold(val_labels, val_probs)
    val_metrics = compute_metrics(val_labels, val_probs, threshold)

    torch.save(
        {
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "img_size": args.img_size,
            "dropout": args.dropout,
            "threshold": threshold,
            "class_names": list(CLASS_NAMES),
            "val_metrics": val_metrics.as_dict(),
            "best_epoch": best["epoch"],
        },
        out_dir / "best.pt",
    )
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, out_dir / "curves.png", fold_name)
    print(f"  [{fold_name}] best epoch {best['epoch']} -> val {val_metrics.summary()}")
    return {"fold": fold_name, "threshold": threshold, "best_epoch": best["epoch"],
            "val_metrics": val_metrics.as_dict(), "checkpoint": str(out_dir / "best.pt")}


def plot_history(history: list[dict], path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is a convenience, never a hard dependency
        return
    epochs = [row["epoch"] for row in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(epochs, [row["train_loss"] for row in history], label="train")
    left.plot(epochs, [row["val_loss"] for row in history], label="val")
    left.set_xlabel("epoch"); left.set_ylabel("loss"); left.legend(); left.set_title(f"{title} loss")
    right.plot(epochs, [row["val_acc"] for row in history], label="val acc")
    right.plot(epochs, [row["val_auc"] for row in history], label="val auc")
    right.set_ylim(0, 1); right.set_xlabel("epoch"); right.legend(); right.set_title(f"{title} metrics")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def evaluate_ensemble(
    fold_results: list[dict], test_samples: list[Sample], args: argparse.Namespace, device: torch.device
) -> dict:
    """Average the fold models' probabilities on the untouched test split."""
    loader = make_loader(test_samples, args, train=False)
    summed: list[float] | None = None
    labels: list[int] = []
    per_fold: list[dict] = []
    for result in fold_results:
        checkpoint = torch.load(result["checkpoint"], map_location=device, weights_only=False)
        model = build_model(args.arch, pretrained=False, dropout=args.dropout).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        probs, labels, _ = predict(model, loader, device, tta=not args.no_tta)
        per_fold.append(compute_metrics(labels, probs, checkpoint["threshold"]).as_dict())
        summed = probs if summed is None else [a + b for a, b in zip(summed, probs)]
    mean_probs = [value / len(fold_results) for value in summed]
    mean_threshold = sum(result["threshold"] for result in fold_results) / len(fold_results)
    return {
        "ensemble": compute_metrics(labels, mean_probs, mean_threshold).as_dict(),
        "ensemble_at_0.5": compute_metrics(labels, mean_probs, 0.5).as_dict(),
        "per_fold": per_fold,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    root = find_data_root(args.data_root)
    samples = list_samples(root)
    labels = [sample.label for sample in samples]
    counts = {name: labels.count(index) for index, name in enumerate(CLASS_NAMES)}
    print(f"[data] root={root}\n[data] {len(samples)} images {counts}\n[setup] device={device} arch={args.arch}")

    pool_index, holdout_val_index, test_index = holdout_split(labels, args.val_ratio, args.test_ratio, args.seed)
    if args.folds == 0:
        pool_folds = [holdout_val_index]
        pool_all = pool_index + holdout_val_index
    else:
        # With cross-validation the holdout val slice rejoins the training pool.
        pool_all = sorted(pool_index + holdout_val_index)
        pool_labels = [labels[i] for i in pool_all]
        pool_folds = [[pool_all[i] for i in fold] for fold in stratified_folds(pool_labels, args.folds, args.seed)]
    test_samples = [samples[i] for i in test_index]
    print(f"[split] pool={len(pool_all)} test={len(test_samples)} folds={max(args.folds, 1)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict] = []
    for number, val_index in enumerate(pool_folds, 1):
        val_set = set(val_index)
        train_samples = [samples[i] for i in pool_all if i not in val_set]
        val_samples = [samples[i] for i in val_index]
        name = f"fold{number}" if args.folds else "holdout"
        print(f"[{name}] train={len(train_samples)} val={len(val_samples)}")
        fold_results.append(train_fold(name, train_samples, val_samples, args, device, args.out_dir / name))

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_root": str(root),
        "class_counts": counts,
        "folds": fold_results,
    }
    for key in ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity", "f1"):
        values = [result["val_metrics"][key] for result in fold_results]
        values = [v for v in values if v == v]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            summary.setdefault("val_mean", {})[key] = {"mean": round(mean, 4), "std": round(std, 4)}

    if test_samples:
        print("\n[test] evaluating on the held-out split")
        summary["test"] = evaluate_ensemble(fold_results, test_samples, args, device)
        ensemble = summary["test"]["ensemble"]
        print(f"[test] ensemble acc={ensemble['accuracy']:.3f} auc={ensemble['auc']:.3f} "
              f"sens={ensemble['sensitivity']:.3f} spec={ensemble['specificity']:.3f}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {args.out_dir / 'summary.json'}")
    if "val_mean" in summary:
        stats = summary["val_mean"]
        print("[val] " + "  ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
