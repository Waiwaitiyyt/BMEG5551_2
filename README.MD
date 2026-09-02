# BMEG5552 — Implant Loosening Detection (Demo Prototype)

This is the group project repository for **BMEG5552** at the **University of Western Australia**. The project demonstrates a software pipeline for detecting **aseptic loosening** of orthopaedic implants from X-ray images.

> **Important:** This is a **demo prototype for educational purposes, not for clinical use.** The classifier is trained on 206 images from a single public database, which is far too small to support a clinical claim.

---

## What does this project do?

1. **Locate the implant** in an X-ray image using an object detection model (YOLO).
2. **Predict loosening** — a ResNet50 CNN classifies the X-ray as *Loose* or *Well fixed* and reports the probability.
3. **Explain the prediction** — a Grad-CAM heat map shows which region of the radiograph drove the classifier's decision.
4. **Display everything** on a web page — the original X-ray with the bounding box, the verdict, and the heat map overlay.

One click runs all three: the two model calls go out concurrently and settle
independently, so a failure in one half still leaves the other's result on screen.

---

## How it works (high-level)

```
                                                       ┌─ YOLO ────────→ bounding box
X-ray  →  Frontend  →  Express gateway  →  FastAPI  ───┤
            :3000          :3000            :8000      └─ ResNet50 ────→ p(loose) + Grad-CAM
                       serves the page,                                            │
                       proxies /api/* ──────────────────────────────────────────── ↓
                                                                        JSON back to the frontend
```

1. **Get data** — Download X-ray images from Kaggle.
2. **Label data** — Use Label Studio to draw boxes around the implants in each X-ray.
3. **Train the detector** — Ultralytics (YOLO) trains an object detection model. The result is a `.pt` file.
4. **Train the classifier** — `tools/implant_loosening/` trains a ResNet50 on the Control/Loose folders, on Colab. The result is another `.pt` file.
5. **Serve both models** — A FastAPI server loads them once at startup and exposes `/predict` and `/classify`.
6. **Show results** — A vanilla HTML/CSS/JS web page lets users upload an X-ray and see the box, the verdict, and the heat map.

The Express (Node.js) server is the single origin the browser talks to: it serves
the static frontend and proxies `/api/*` to FastAPI, so no CORS setup is needed.
The pipeline is stateless — uploads are held in memory for inference only and
nothing is written to disk.

---

## Project Structure

```
5552/
├── public/                  # Frontend (vanilla HTML, CSS, JavaScript)
│   ├── index.html           #   Main web page
│   ├── app.js               #   Frontend logic
│   └── style.css            #   Page styles
│
├── server/
│   ├── py/                  # Python backend — FastAPI, YOLO + ResNet50 inference
│   │   ├── pyproject.toml   #   Python dependencies (FastAPI, Ultralytics, torch)
│   │   ├── .python-version  #   Python version (3.11+)
│   │   ├── server.py        #   FastAPI app entrypoint (/predict, /classify, /health)
│   │   ├── classifier.py    #   ResNet50 loosening classifier + Grad-CAM
│   │   └── weights/
│   │       ├── best.pt      #   Trained YOLO checkpoint (localisation)
│   │       └── classifier.pt #  Trained ResNet50 checkpoint (prediction + heat map)
│   │
│   └── ts/                  # TypeScript backend — Express gateway
│       ├── package.json     #   Node dependencies (Express, http-proxy-middleware)
│       ├── tsconfig.json
│       └── src/
│           └── server.ts    #   Static hosting + /api proxy to FastAPI
│
├── tools/                   # Utility scripts (dataset download, labelling, training)
│   ├── pyproject.toml       #   Python dependencies (kagglehub, label-studio, ultralytics)
│   ├── .python-version
│   ├── fetch_dataset.py     #   Download X-ray datasets from Kaggle
│   ├── prepare_dataset.py   #   Match Label Studio exports and split YOLO data
│   ├── train.py             #   Train the YOLO detector
│   └── implant_loosening/   #   Loosening classifier — see its own README
│       ├── colab_train.ipynb #    Run this on Colab to train
│       ├── train.py         #    5-fold CV training entry point
│       ├── data.py          #    Splits, transforms, dataset discovery
│       ├── model.py         #    torchvision backbones with a 1-logit head
│       ├── engine.py        #    Train/eval loops, AMP, flip TTA
│       ├── metrics.py       #    AUC, sensitivity/specificity, threshold search
│       └── predict.py       #    CLI inference + Grad-CAM overlays
│
├── resnet50_models/         # Trained 5-fold classifier run (fold1 is deployed)
│
├── .gitignore
└── README.MD                # This file
```

---

## Tech Stack

| Component               | Technology                                           |
| ----------------------- | ---------------------------------------------------- |
| Frontend                | Vanilla HTML, CSS, JavaScript (no frameworks)        |
| ML Model                | YOLO (via[Ultralytics](https://docs.ultralytics.com)) |
| Python Backend          | [FastAPI](https://fastapi.tiangolo.com/)              |
| Node.js Backend         | [Express](https://expressjs.com/) (TypeScript)        |
| Data Labelling          | [Label Studio](https://labelstud.io/)                 |
| Dataset Source          | [Kaggle](https://www.kaggle.com/)                     |
| Python Package Manager  | [uv](https://docs.astral.sh/uv/)                      |
| Node.js Package Manager | npm                                                  |

---

## Setting Up the Environment

You will need **Python 3.11+**, **Node.js 20+**, and **uv** installed on your machine.

### 1. Install uv (Python package manager)

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or follow the on-screen instructions to add `uv` to your PATH.

### 2. Install Node.js

Download and install from [nodejs.org](https://nodejs.org/) (v20 or later recommended). Verify installation:

```bash
node --version   # should be v20.x.x or higher
npm --version
```

### 3. Set up Python components

Each Python component (`tools` and `server/py`) has its own isolated environment managed by `uv`. Run the following from the project root:

```bash
# Set up the tools environmen (datasett download, labelling)
cd tools
uv sync

# Set up the Python server environment (FastAPI + YOLO)
cd ../server/py
uv sync

# Return to project root
cd ../..
```

`uv sync` reads the `.python-version` and `pyproject.toml` files in each folder. It automatically downloads the correct Python version and installs all dependencies into a local `.venv` folder.

### 4. Set up the TypeScript (Node.js) server

```bash
cd server/ts
npm install
cd ../..
```

---

## Workflow — Step by Step

### Step 1: Download the Dataset

The dataset is hosted on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/aseptic-loose-hip-implant-xray-database). You need a Kaggle account to download it.

```bash
cd tools
uv run python fetch_dataset.py
```

The script will download the X-ray images and print the path where they are stored. Dataset files are saved in the project root directory.

### Step 2: Label the Images

Label Studio is a graphical tool that lets you draw bounding boxes around the implants in each X-ray image. These labels teach the YOLO model what to look for.

Start Label Studio:

**Linux / macOS:**

```bash
cd tools
NLTK_DISABLE_IMPORT_SECURITY=1 uv run label-studio start
```

**Windows (PowerShell):**

```powershell
cd tools
$env:NLTK_DISABLE_IMPORT_SECURITY = "1"
uv run label-studio start
```

> If you omit `NLTK_DISABLE_IMPORT_SECURITY=1`, Label Studio may fail to start with an `ImportError: Blocked import of regex from current working directory` error — see [Troubleshooting](#troubleshooting) below.

Once Label Studio opens in your browser:

1. Create a new project.
2. Import the downloaded X-ray images.
3. Set up a **bounding box** labelling task (object detection with rectangles).
4. Label each image by drawing a box around the implant.
5. Export the labels in **YOLO format** when finished.

### Step 3: Prepare the YOLO Dataset

The repository contains the downloaded source images under `datasets/` and the
current Label Studio export under `label/labels/`. `tools/prepare_dataset.py`
matches each label to its image, creates a reproducible train/validation split,
and writes the directory structure expected by Ultralytics:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Run it from the project root (the default paths are already configured):

```bash
cd tools
uv run python prepare_dataset.py
```

The default split is 80% training and 20% validation, with a fixed seed of
`42`. To rebuild an existing output directory, add `--overwrite`. The input
images are searched recursively, so the Kaggle directory layout does not need
to be flattened. You can also provide custom locations, for example:

```bash
uv run python prepare_dataset.py \
  --images-dir ../datasets \
  --labels-dir ../label/labels \
  --output-dir ../dataset \
  --val-ratio 0.2 \
  --seed 42
```

#### Required annotation format

Export annotations from Label Studio in **YOLO format**. For every labelled
image there must be one UTF-8 text file containing one object per line:

```text
class_id x_center y_center width height
```

The four box values are normalised to `[0, 1]` relative to the image (not pixel
coordinates). This project currently has one class, `Implant`, so each row
starts with class id `0`, for example:

```text
0 0.4949 0.5140 0.6497 0.9213
```

The label file name normally matches the image stem. The current Label Studio
export uses names such as `724c...-loose_105.txt`; the preparation script maps
that suffix to the source image `loose (105).png`. Images may be PNG, JPEG, BMP,
TIFF, or WebP. A label without a matching image, malformed row, or box value
outside `[0, 1]` stops the preparation with an actionable error.

### Step 4: Train the YOLO Model

After preparing the dataset, train using the Ultralytics `yolo11n.pt`
pretrained checkpoint:

```bash
cd tools
uv run python train.py
```

The default run uses 100 epochs, 640-pixel images, and automatic batch-size
selection. Results and checkpoints are written to `runs/detect/implant/`; the
best checkpoint is `runs/detect/implant/weights/best.pt`. Common options:

```bash
uv run python train.py --epochs 150 --imgsz 640 --batch 16 --device 0
uv run python train.py --model yolo11s.pt --name implant-large
uv run python train.py --data ../my_dataset/data.yaml
```

Use `--resume --model path/to/last.pt` to resume an interrupted run. Copy the
resulting checkpoint to where the FastAPI server loads it from:

```bash
cp runs/detect/implant/weights/best.pt server/py/weights/best.pt
```

### Step 4b: Train the Loosening Classifier

The classifier is a separate model with its own pipeline in
`tools/implant_loosening/` — it learns from the Kaggle database's `Control/`
and `Loose/` folders directly and needs no bounding-box labels. It is meant to
be trained on Colab (see `tools/implant_loosening/README.md` for the full
story):

1. Open `tools/implant_loosening/colab_train.ipynb` in Colab.
2. Set the runtime to a **T4 GPU**.
3. Run the cells top to bottom — roughly 15 minutes for the defaults
   (ResNet50, 320 px, 40 epochs, 5-fold cross-validation).

The run writes `runs/<name>/fold<k>/best.pt` plus a `summary.json` holding
per-fold validation metrics and held-out test results. Each checkpoint carries
its own architecture, input size, and tuned decision threshold, so the server
needs no configuration to load one.

**The deployed model is `fold1`** — it had the highest *validation* AUC (0.993)
of the five, which is why it was picked; on the held-out test set it reaches
AUC 0.988, accuracy 0.976, sensitivity 0.955, specificity 1.000. The 5-model
ensemble scores marginally lower here (AUC 0.986, accuracy 0.951) while costing
5× the memory and latency, so a single fold is the better deployment.

```bash
cp resnet50_models/fold1/best.pt server/py/weights/classifier.pt
```

> Selecting a model by test-set score would be peeking. Fold 1 was chosen on
> validation AUC, which is independent of the test split; the test numbers above
> are reported, not optimised against.

### Step 5: Run the Servers

Both servers run at the same time, in two separate terminals.

**Terminal 1 — Python (FastAPI) server**, runs YOLO inference on port 8000:

```bash
cd server/py
uv run python server.py
```

Startup takes a few seconds while the checkpoint loads. The server is ready when
it prints `Application startup complete.`; confirm with:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_path":".../weights/best.pt","classes":["Implant"], ...}
```

**Terminal 2 — TypeScript (Express) server**, serves the frontend on port 3000:

```bash
cd server/ts
npm run dev
```

### Step 6: Open the Frontend

Open **<http://localhost:3000>** in a browser. The page is a three-pane
workstation:

| Pane                   | What it does                                                                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Study queue** (left) | Drop one or many X-rays anywhere on the window to build a session queue. Each row shows its thumbnail, dimensions, and either its verdict (`loose 97%` / `fixed 100%`) or its state (`queued`, `running`, `none`, `error`). Filter by file name. Nothing is persisted server-side. |
| **Viewer** (centre)    | The active study with the detector's box and the classifier's Grad-CAM drawn over it. Tools: **window/level**, **zoom**, **pan** — pick one, then drag on the image. The scroll wheel zooms about the pointer, and once the image is larger than its frame a plain drag pans it whatever tool is armed, so you can zoom in and go looking (hold **Shift** to keep adjusting window/level there, or drag with the middle button to pan at any zoom). The pan stops at the image edges; click the zoom readout to reset. Overlay modes: **Box**, **Heatmap**, **Both**, **Off** — a display switch, not a re-run. **Compare** opens a picker for choosing any two queued studies, then splits the viewer between them (A on the left, B on the right; B is the active study). |
| **Findings** (right)   | The loosening verdict with p(loose) and the decision threshold, then the confidence bar and `xyxy` per detection, the **inference controls**, the run metadata, both raw JSON responses, and **Export JSON**. |

**Run analysis** fires both models at once. The verdict card is tinted by its
own answer — red for loosening, green for well fixed — and reports the
confidence in the label it gave, with p(loose) and the tuned threshold spelled
out underneath so a marginal call is visible as one.

Two of the inference controls are sent to the detector on the next run — the
**confidence threshold** and the **IoU (NMS)** threshold. They do not affect the
classifier, which uses its own tuned threshold. **Window/level** is a
display-only adjustment: the bytes posted to either model are always the
untouched original file.

> The **Heatmap** overlay is a real Grad-CAM saliency map from the ResNet50
> classifier, not a decoration derived from the box. On confidently well-fixed
> cases read it with care — Grad-CAM rescales per image, so a hot spot appears
> even when the activation behind it is negligible.

> Opening `public/index.html` directly from disk also works — `app.js` then falls
> back to `http://localhost:8000` — but going through <http://localhost:3000> is
> the intended path, and the one that needs no CORS configuration.

---

## API Reference

The frontend calls the FastAPI server through the Express proxy: a request to
`/api/predict` on port 3000 is forwarded to `/predict` on port 8000. All routes
below therefore work at `http://localhost:3000/api/...` and at
`http://localhost:8000/...` directly.

| Route            | Feature       | Model            |
| ---------------- | ------------- | ---------------- |
| `POST /predict`  | localisation  | YOLO             |
| `POST /classify` | prediction + heat map | ResNet50 + Grad-CAM |
| `GET /health`    | status of both | —               |

### `POST /predict`

Runs implant detection on one uploaded image.

**Request** — `multipart/form-data`:

| Field   | Required | Range              | Purpose                                                       |
| ------- | -------- | ------------------ | ------------------------------------------------------------- |
| `file`  | yes      | 25 MB max          | PNG, JPEG, WEBP, BMP, or TIFF                                 |
| `conf`  | no       | `0 < conf <= 1`    | Confidence threshold for this request only                    |
| `iou`   | no       | `0 < iou <= 1`     | NMS IoU threshold for this request only                       |
| `imgsz` | no       | `64`–`1536`        | Inference size for this request only; rounded up to a /32 step |

Each optional field overrides the process-wide default for that one call — the
server never mutates its own configuration, so concurrent requests cannot see
each other's settings. Omit them all and the server's `CONF_THRESHOLD`,
`IOU_THRESHOLD`, and `IMGSZ` apply. These three back the workstation's
**Inference controls**.

```bash
curl -X POST http://localhost:3000/api/predict \
  -F "file=@dataset/images/val/loose (103).png" \
  -F conf=0.35 -F iou=0.6
```

**Response** — `200 OK`

```json
{
  "detections": [
    { "label": "Implant", "confidence": 0.9592, "box": [55.32, 13.63, 312.75, 325.63] }
  ],
  "image_width": 331,
  "image_height": 331,
  "conf_threshold": 0.35,
  "iou_threshold": 0.6,
  "imgsz": 640,
  "model": "best.pt",
  "classes": ["Implant"],
  "inference_ms": 41.7
}
```

`box` is `[x1, y1, x2, y2]` — the top-left and bottom-right corners in pixels of
the **original uploaded image**, so the frontend can scale them to whatever size
the preview is displayed at. `detections` is sorted by confidence, highest
first, and is an empty array when nothing is found above the threshold.

Everything after `image_height` echoes the settings this run actually used, so
the viewer's corner annotation and the findings pane always describe the run
that produced the boxes on screen rather than assuming the sliders and the
server agree.

**Error responses** carry a `detail` string:

| Status | Meaning                                                       |
| ------ | ------------------------------------------------------------- |
| 400    | File was empty, or could not be decoded as an image           |
| 413    | Image exceeded `MAX_UPLOAD_BYTES`                             |
| 422    | The `file` field was missing, or `conf`/`iou`/`imgsz` was out of range |
| 502    | Express could not reach FastAPI (inference server not running) |
| 503    | FastAPI is up but the model has not finished loading           |

### `POST /classify`

Classifies one uploaded X-ray as loose or well fixed, and returns the Grad-CAM
heat map that explains the call.

**Request** — `multipart/form-data`:

| Field       | Required | Range           | Purpose                                            |
| ----------- | -------- | --------------- | -------------------------------------------------- |
| `file`      | yes      | 25 MB max       | PNG, JPEG, WEBP, BMP, or TIFF                      |
| `threshold` | no       | `0 < t < 1`     | Decision threshold for this request only           |
| `heatmap`   | no       | `true`/`false`  | Compute the Grad-CAM overlay (default `true`)      |

Omit `threshold` and the value tuned on validation data (0.6132) applies. Set
`heatmap=false` to skip the backward pass — roughly 0.19 s instead of 0.50 s on
CPU.

```bash
curl -X POST http://localhost:3000/api/classify \
  -F "file=@dataset/images/val/loose (103).png"
```

**Response** — `200 OK`

```json
{
  "label": "Loose",
  "probability": 0.9743,
  "confidence": 0.9743,
  "threshold": 0.6132,
  "classes": ["Control", "Loose"],
  "heatmap": "data:image/png;base64,iVBORw0KGgo…",
  "heatmap_box": [20.01, 20.01, 310.99, 310.99],
  "image_width": 331,
  "image_height": 331,
  "model": "classifier.pt",
  "arch": "resnet50",
  "imgsz": 320,
  "inference_ms": 498.2
}
```

- `probability` is **p(Loose)** and is threshold free — report it rather than
  the label alone, because 0.62 and 0.99 both read "Loose" and only one of them
  deserves a second look.
- `confidence` is the probability of whichever label was reported, so it is
  `probability` for Loose and `1 - probability` for Control.
- `heatmap` is an RGBA PNG data URI, the same pixel size as the upload and fully
  transparent outside the region the classifier saw, so the frontend stretches
  it over the image with no coordinate maths. It is `null` when `heatmap=false`.
- `heatmap_box` is `[x1, y1, x2, y2]` of that region in source pixels: the model
  resizes the short side by 1.14× and centre-crops to 320², so roughly the
  central 88% of a square image is all it looks at.

> **Reading the heat map.** Grad-CAM is renormalised per image, so a hot spot
> appears even when the underlying activation is negligible. On a case scored
> `Control` at p(loose) 0.002, the red region is the strongest evidence the model
> could find *for* loosening — which is nearly none. It is a candidate, not a
> finding. The UI says so on the verdict card.

**Error responses** carry a `detail` string:

| Status | Meaning                                                          |
| ------ | ---------------------------------------------------------------- |
| 400    | File was empty, or could not be decoded as an image              |
| 413    | Image exceeded `MAX_UPLOAD_BYTES`                                |
| 422    | The `file` field was missing, or `threshold` was outside `(0, 1)` |
| 503    | The classifier checkpoint is not loaded — see `CLASSIFIER_PATH`   |

A 503 here does not affect `/predict`: the classifier is optional, and
localisation keeps working without it.

### `GET /health`

Reports model status and the defaults the frontend seeds its controls from.

```json
{
  "status": "ok",
  "model_path": "/…/server/py/weights/best.pt",
  "model_name": "best.pt",
  "classes": ["Implant"],
  "conf_threshold": 0.5,
  "iou_threshold": 0.7,
  "imgsz": 640,
  "device": "auto",
  "classifier": {
    "available": true,
    "model_path": "/…/server/py/weights/classifier.pt",
    "model_name": "classifier.pt",
    "arch": "resnet50",
    "img_size": 320,
    "classes": ["Control", "Loose"],
    "threshold": 0.6132,
    "device": "cpu",
    "val_metrics": { "accuracy": 0.9697, "auc": 0.9926, "sensitivity": 1.0, "specificity": 0.9333, "…": "…" }
  }
}
```

`classifier.available` is `false` when the checkpoint is missing; the object
then carries only `available` and `model_path`. `status` is `loading` until the
detector checkpoint has finished loading. The page polls
this every 15 s to drive the service chips in its top bar, and seeds the two
threshold sliders from it — until you move one, after which the sliders win.

The Express gateway has its own `GET /healthz` that reports only whether the
gateway itself is up; it drives the second chip.

### Configuration

Both servers read environment variables, so nothing needs editing to change a
port or point at a different checkpoint.

**FastAPI** (`server/py`):

| Variable           | Default              | Purpose                                     |
| ------------------ | -------------------- | ------------------------------------------- |
| `MODEL_PATH`       | `weights/best.pt`       | YOLO detector checkpoint to load            |
| `CLASSIFIER_PATH`  | `weights/classifier.pt` | ResNet50 loosening checkpoint; `/classify` returns 503 if absent |
| `CONF_THRESHOLD`   | `0.5`                | Default confidence, when `conf` is not sent |
| `IOU_THRESHOLD`    | `0.7`                | Default NMS IoU, when `iou` is not sent     |
| `IMGSZ`            | `640`                | Default inference size, when `imgsz` is not sent — match training |
| `DEVICE`           | auto                 | `0`, `0,1`, or `cpu`                        |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MB)   | Upload size limit                           |
| `HOST` / `PORT`    | `127.0.0.1` / `8000` | Bind address                                |
| `ALLOWED_ORIGINS`  | `*`                  | Comma-separated CORS origins                |
| `RELOAD`           | off                  | `1` to auto-reload on code changes          |

**Express** (`server/ts`):

| Variable        | Default                 | Purpose                              |
| --------------- | ----------------------- | ------------------------------------ |
| `INFERENCE_URL` | `http://127.0.0.1:8000` | Where to proxy `/api/*`              |
| `PUBLIC_DIR`    | `../../public`          | Static directory to serve            |
| `HOST` / `PORT` | `127.0.0.1` / `3000`    | Bind address                         |

For example, to run inference on the first GPU with a stricter threshold:

```bash
cd server/py
DEVICE=0 CONF_THRESHOLD=0.5 uv run python server.py
```

---

## Important Notes

- **Python versions** are pinned in `.python-version` files inside each component folder. `uv sync` respects these automatically — you do not need to manually manage Python versions.
- **Always use `uv run`** when executing Python scripts inside a component. This ensures the correct virtual environment is active. For example:
  ```bash
  cd tools && uv run python fetch_dataset.py
  ```
- **Datasets are git-ignored.** The `.gitignore` file excludes the `datasets/` and `dataset/` folders, so downloaded images are not committed to the repository.
- **The trained checkpoints live in `server/py/weights/`.** `best.pt` is the YOLO detector (default `MODEL_PATH`) and `classifier.pt` is the ResNet50 (default `CLASSIFIER_PATH`), so the FastAPI server finds both with no configuration. After retraining, copy `runs/detect/implant/weights/best.pt` and `tools/implant_loosening/runs/<name>/fold1/best.pt` over them respectively.
- **The classifier is optional at runtime.** Without `classifier.pt` the server still starts, localisation still works, and `/classify` answers 503 — the frontend then shows the box and an explanatory card in place of the verdict.
- **Both servers must be running** for the web page to work. Express alone serves the page but every detection returns `502`; FastAPI alone answers on port 8000 but nothing serves the frontend.
- This project is a **prototype for educational purposes**. It is not intended for clinical use.

---

## Troubleshooting

### Label Studio: `ImportError: Blocked import of regex from current working directory`

**Symptom:** Running `label-studio` without `NLTK_DISABLE_IMPORT_SECURITY=1` set (e.g. `uv run label-studio --version`) fails with a traceback like:

```
File ".../nltk/inisec.py", line 128, in find_spec
    raise ImportError(
ImportError: Blocked import of regex from current working directory for security reasons. Use '-P' or set PYTHONSAFEPATH to prevent Python from searching the current working directory.
```

**Cause:** `label-studio` depends on `label-studio-sdk`, which depends on `nltk`. Recent `nltk` versions ship a built-in security guard (`nltk/inisec.py`) that mitigates CWE-427 (uncontrolled search path) by blocking any import that resolves to a path *inside the current working directory* whenever `nltk` itself triggers the import.

This project's `tools/.venv` lives inside `tools/`, the directory you run `label-studio` from. So when `nltk` imports its `regex` dependency (which resolves to `tools/.venv/lib/.../site-packages/regex`), the guard sees "resolves under cwd" and blocks it — even though it's just your own project's virtual environment, not a maliciously planted file. It's a false positive.

Note: setting `PYTHONSAFEPATH=1` (as the error message suggests) does **not** fix this — that flag only affects freshly spawned child interpreters, not the main-process check that raises this error.

**Fix:** Set `NLTK_DISABLE_IMPORT_SECURITY=1` before running `label-studio`:

```bash
NLTK_DISABLE_IMPORT_SECURITY=1 uv run label-studio start
```

```powershell
$env:NLTK_DISABLE_IMPORT_SECURITY = "1"
uv run label-studio start
```

---

### FastAPI server: `ImportError: libGL.so.1: cannot open shared object file`

**Symptom:** `uv run python server.py` in `server/py` fails during startup while
importing Ultralytics:

```
File ".../cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Cause:** Ultralytics depends on `opencv-python`, whose wheels link against
`libGL` — a desktop OpenGL library. Headless machines (servers, WSL without a GUI
stack, slim Docker images) do not ship it, so the `cv2` native module fails to
load even though the package installed cleanly.

**Fix:** already applied — `server/py/pyproject.toml` pulls in
`opencv-python-headless` and overrides `opencv-python` out of the dependency
tree:

```toml
[tool.uv]
override-dependencies = ["opencv-python; sys_platform == 'never'"]
```

The headless build exposes the identical `cv2` API without the GUI dependency.
If you still hit this, run `uv sync` in `server/py` to pick up the override, and
check that `uv pip list | grep opencv` shows only `opencv-python-headless`.

Note that `tools/` deliberately keeps the regular `opencv-python`, since
Ultralytics uses its display functions when plotting training results.
