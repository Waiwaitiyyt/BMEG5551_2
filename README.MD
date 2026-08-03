# BMEG5552/1 Project - Implant Loosening Detection

This is a demo repository for loosening detection on implant X-ray images.

## Setting-up

This project has two TypeScript/Python components: `server/ts` (Node.js/TypeScript API server), and `server/py` + `tools` (Python, managed with [uv](https://docs.astral.sh/uv/)).

### Node.js (server/ts)

1. Install [Node.js](https://nodejs.org/) (v20+ recommended).
2. Install dependencies:
   ```sh
   cd server/ts
   npm install
   ```
3. Run the dev server:
   ```sh
   npm run dev
   ```

### Python (server/py, tools)

Python dependencies are managed with [uv](https://docs.astral.sh/uv/), which handles both the interpreter version and virtualenv for you.

1. Install `uv`:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. For each Python component (`server/py` and `tools`), sync its environment:
   ```sh
   cd server/py   # or tools
   uv sync
   ```
   This creates a `.venv` using the Python version pinned in `.python-version` and installs dependencies from `pyproject.toml`.
3. Run scripts inside the environment with `uv run`, e.g.:
   ```sh
   uv run python src/fetch_dataset.py
   ```


## Datasets

The datasets are currently fetched from **kaggle** website. Download the dataset by running `./tools/src/fetch_dataset.py`.
