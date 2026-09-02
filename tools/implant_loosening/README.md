# Implant loosening classifier

Binary image classification: does this hip X-ray show an **aseptically loose**
implant, or a **well-fixed** one? Built for
`tawsifurrahman/aseptic-loose-hip-implant-xray-database` — 206 grayscale 331×331
PNGs, 94 `Control` / 112 `Loose`.

This is the classification counterpart to the YOLO detector in `tools/train.py`
(which localises the implant); it is a rewrite of `tools/abdallah-project.ipynb`
in PyTorch, with the evaluation problems of that notebook fixed.

## What changed versus the notebook

| | `abdallah-project.ipynb` | here |
|---|---|---|
| Test set | `validation_split=0.2` over the **same** directory used for training — every "test" image was also a training image, so the reported 0.775 is inflated | 20% held out before anything else, never seen by any fold |
| Validation | one 20-image split | 5-fold cross-validation, mean ± std reported |
| Backbone | frozen ResNet50V2, 4 epochs | frozen warm-up → full fine-tune, cosine LR, early stopping |
| Head | 2-way softmax on a binary problem | single logit + BCE, so AUC and a tunable threshold are available |
| Metrics | accuracy only | AUC, sensitivity, specificity, balanced accuracy, F1, confusion matrix |
| Threshold | fixed 0.5 | tuned on validation (Youden's J), then applied unchanged to test |
| Class imbalance | ignored | class-balanced sampler |

Sensitivity matters more than accuracy here: a missed loosening sends a patient
home with a failing prosthesis, so read `sensitivity` and `specificity`
separately rather than trusting a single accuracy number.

## Files

- `data.py` — dataset discovery, stratified splits/folds, augmentation pipeline
- `model.py` — torchvision backbones (resnet18/50, densenet121, efficientnet_b0/b3, convnext_tiny) with a 1-logit head
- `engine.py` — train/eval loops, AMP, flip TTA
- `metrics.py` — AUC, sensitivity/specificity, threshold search (no sklearn dependency)
- `train.py` — the training entry point
- `predict.py` — inference on one image or a folder, with Grad-CAM overlays
- `colab_train.ipynb` — **run this on Colab**: GPU check → data download → training → results → Grad-CAM

## Colab

Open `colab_train.ipynb` in Colab, set the runtime to a **T4 GPU**, and run the
cells top to bottom. The defaults (`resnet50`, 320px, 40 epochs, 5 folds) take
roughly 15 minutes.

## Local usage

```bash
pip install -r requirements.txt

# 5-fold CV + held-out test set, dataset auto-located under ../../datasets
python train.py --folds 5 --epochs 40

# faster single split, smaller backbone
python train.py --folds 0 --arch resnet18 --img-size 224 --epochs 25

# predict (a run directory ensembles all fold checkpoints)
python predict.py --checkpoint runs/baseline --image scan.png --cam cam_out/
```

Key flags: `--arch`, `--img-size`, `--epochs`, `--freeze-epochs`, `--batch-size`,
`--lr` / `--head-lr`, `--folds`, `--test-ratio`, `--patience`, `--seed`,
`--no-balance`, `--no-tta`, `--no-amp`. Run `python train.py --help` for the rest.

## Outputs

```
runs/<name>/
  fold1/best.pt        # weights + arch + img_size + tuned threshold + val metrics
  fold1/history.csv    # per-epoch losses and validation metrics
  fold1/curves.png     # loss and metric curves
  ...
  summary.json         # config, per-fold results, val mean±std, ensemble test metrics
```

## Notes on the dataset

206 images is small for deep learning, so treat single-run numbers with
suspicion — the fold-to-fold spread in `summary.json` is the honest error bar.
Two further caveats worth stating in any write-up: the database gives no
patient identifiers, so images from the same patient may land on both sides of a
split (patient-level grouping would be stricter), and the Control and Loose sets
may differ in acquisition protocol, which a CNN can exploit without ever looking
at the implant. Check the Grad-CAM overlays: if attention sits on borders or
image text rather than on the bone–implant interface, the model is reading the
scanner, not the pathology.
