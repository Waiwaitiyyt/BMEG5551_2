# BMEG5552 — Implant Loosening Detection (Demo Prototype)

This is the group project repository for **BMEG5552** (Medical Device Development) at the **University of Western Australia**. The project demonstrates the first step of a software pipeline for detecting **aseptic loosening** of orthopaedic implants from X-ray images.

> **Important:** This is a **demo prototype**. It focuses on locating the implant inside an X-ray image. The actual diagnosis of loosening is not implemented — this project shows what the full workflow *would* look like.

---

## What does this project do?

1. **Locate the implant** in an X-ray image using a computer vision model (YOLO).
2. **Display the result** on a web page — the original X-ray with a bounding box drawn around the implant.

In a real clinical tool, the next step would be to analyse the region around the implant to detect loosening. This project builds the foundation for that pipeline.

---

## How it works (high-level)

```
X-ray Image  →  Frontend (web page)  →  FastAPI server (Python)  →  YOLO model  →  Image with bounding box  →  Back to Frontend
                                                                         ↑
                                                              trained on labelled data
```

1. **Get data** — Download X-ray images from Kaggle.
2. **Label data** — Use Label Studio to draw boxes around the implants in each X-ray.
3. **Train model** — Use Ultralytics (YOLO) to train an object detection model. The result is a `.pt` file.
4. **Serve the model** — A FastAPI server loads the trained model, receives uploaded X-ray images, runs inference, and returns the image with the implant highlighted.
5. **Show results** — A vanilla HTML/CSS/JS web page lets users upload an X-ray and see the result.

An additional Express (Node.js) server handles data storage and other supporting tasks.

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
│   ├── py/                  # Python backend — FastAPI + YOLO inference
│   │   ├── pyproject.toml   #   Python dependencies (FastAPI, Ultralytics)
│   │   ├── .python-version  #   Python version (3.11+)
│   │   └── src/
│   │       └── py/
│   │           └── __init__.py
│   │
│   └── ts/                  # TypeScript backend — Express (data storage, etc.)
│       ├── package.json     #   Node dependencies (Express)
│       ├── tsconfig.json
│       └── src/
│           └── server.ts
│
├── tools/                   # Utility scripts (dataset download, labelling)
│   ├── pyproject.toml       #   Python dependencies (kagglehub, label-studio)
│   ├── .python-version
│   └── src/
│       ├── fetch_dataset.py #   Download X-ray datasets from Kaggle
│       ├── label.sh         #   Start Label Studio (Linux/macOS)
│       └── label.ps1        #   Start Label Studio (Windows PowerShell)
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
uv run python src/fetch_dataset.py
```

The script will download the X-ray images and print the path where they are stored. Dataset files are saved in the project root directory.

### Step 2: Label the Images

Label Studio is a graphical tool that lets you draw bounding boxes around the implants in each X-ray image. These labels teach the YOLO model what to look for.

Start Label Studio:

**Linux / macOS:**

```bash
cd tools
bash src/label.sh
```

**Windows (PowerShell):**

```powershell
cd tools
.\src\label.ps1
```

> If Label Studio fails to start with an `ImportError: Blocked import of regex from current working directory` error, see [Troubleshooting](#troubleshooting) below. The start scripts already work around this, but it can still occur if you run `label-studio` manually.

Once Label Studio opens in your browser:

1. Create a new project.
2. Import the downloaded X-ray images.
3. Set up a **bounding box** labelling task (object detection with rectangles).
4. Label each image by drawing a box around the implant.
5. Export the labels in **YOLO format** when finished.

### Step 3: Train the YOLO Model

Using the Ultralytics library (already installed in the `server/py` environment), train a YOLO object detection model on your labelled dataset. The training script will be added to `server/py/src/`.

After training, you will get a `.pt` file (the trained model). Place this file where the FastAPI server can find it.

### Step 4: Run the Servers

**Python (FastAPI) server** — handles YOLO inference:

```bash
cd server/py
uv run <startup-command>   # to be configured
```

**TypeScript (Express) server** — handles data storage:

```bash
cd server/ts
npm run dev
```

### Step 5: Open the Frontend

Open `public/index.html` in a browser, or serve it through the Express server. Upload an X-ray image and see the implant highlighted with a bounding box.

---

## Important Notes

- **Python versions** are pinned in `.python-version` files inside each component folder. `uv sync` respects these automatically — you do not need to manually manage Python versions.
- **Always use `uv run`** when executing Python scripts inside a component. This ensures the correct virtual environment is active. For example:
  ```bash
  cd tools && uv run python src/fetch_dataset.py
  ```
- **Datasets are git-ignored.** The `.gitignore` file excludes the `datasets/` and `dataset/` folders, so downloaded images are not committed to the repository.
- This project is a **prototype for educational purposes**. It is not intended for clinical use.

---

## Troubleshooting

### Label Studio: `ImportError: Blocked import of regex from current working directory`

**Symptom:** Running `label-studio` (directly, e.g. `uv run label-studio --version`, rather than via `src/label.sh` / `src/label.ps1`) fails with a traceback like:

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

`tools/src/label.sh` and `tools/src/label.ps1` already set this variable, so use those scripts instead of calling `label-studio` directly whenever possible.
