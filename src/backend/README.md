# Backend

This directory contains the backend service for the University Administrative Copilot.

## Project structure

```text
backend/
├── api/
│   ├── __init__.py
│   └── ...                 # API routers and endpoint definitions
├── logic/
│   ├── __init__.py
│   └── ...                 # Business logic only
├── main.py                 # Creates the app and connects it to the frontend
├── pyproject.toml          # Project metadata and uv dependencies
├── requirements.txt        # Pip dependency lock list
├── uv.lock                 # uv lock file
└── README.md
```

### Responsibilities

- `api/` contains the routers and endpoints. Request handling, validation, and response definitions belong here.
- `logic/` contains business logic only. Keep framework-specific routing concerns out of this package.
- `main.py` contains only the application setup and the connection between the backend app and the frontend.

## Setup with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already installed, then run:

```bash
cd src/backend
uv sync
source .venv/bin/activate
```

`uv sync` creates or updates the virtual environment from `pyproject.toml` and `uv.lock`.

## Setup with pip

Create and activate a virtual environment, then install the pinned dependencies:

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the backend

From `src/backend`, with the virtual environment activated:

```bash
uvicorn main:app --reload
```

The API is then available at `http://127.0.0.1:8000`.

## Updating dependencies before pushing

Using uv:

```bash
cd src/backend
uv sync
uv pip freeze > requirements.txt
```

Using pip:

```bash
cd src/backend
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

Review dependency changes before committing. If `pyproject.toml` changes, regenerate `uv.lock` with `uv lock` and ensure both dependency files remain consistent.

## Add, commit, and push changes

From the repository root:

```bash
git status
git add src/backend/
git commit -m "Describe the backend change"
git push
```

Always update dependencies and review the resulting changes before pushing.