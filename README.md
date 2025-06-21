[![Code Quality](https://github.com/remla25-team20/model-training/actions/workflows/CodeQuality.yml/badge.svg)](https://github.com/remla25-team20/model-training/actions/workflows/CodeQuality.yml)
[![PyLint](/badges/pylint_badge.svg)](https://github.com/remla25-team20/model-training/actions/workflows/CodeQuality.yml)
[![Flake8](/badges/flake_badge.svg)](https://github.com/remla25-team20/model-training/actions/workflows/CodeQuality.yml)
[![Pipeline](/badges/pipe_test_badge.svg)](https://github.com/remla25-team20/model-training/actions/workflows/pipeline.yml)
# model-training

This repository contains the training pipeline for a sentiment analysis model built for restaurant reviews, used as part of the Release Engineering course.

---

## 📦 Project Structure

```
model-training/
├── data/
│   ├── external/         # Downloaded raw data
│   └── interim/          # Processed train/test data
├── models/               # Trained model artifacts (.joblib)
├── notebooks/            # Jupyter notebooks (for exploration only)
├── src/
│   ├── data/             # Data download and preprocessing scripts
│   └── modeling/         # Training and evaluation
├── .dvcignore
├── .env
├── .gitignore
├── params.yaml           # Parameters for pipeline stages (managed via DVC)
├── poetry.lock
├── pyproject.toml        # Poetry configuration
└── README.md
```
---

## ⚙️ Pipeline Overview

The machine learning pipeline is implemented in modular Python scripts within the `src/` directory:

- `src/data/`
  - `data_download.py`: Downloads raw data from Google Drive
  - `preprocess.py`: Cleans and vectorizes the text
- `src/modeling/`
  - `train.py`: Trains the model and performs the train/test split
  - `evaluation.py`: Evaluates the trained model

All parameters (like `test_size`, `random_state`, and download IDs) are stored in `params.yaml` and tracked with **DVC**.

---

## 📁 Data Flow

1. **Raw data** is downloaded into `data/external/`.
2. **Intermediate train/test data** is saved in `data/interim/`.
3. The **trained model** is stored in the `models/` directory.

---

## 📦 Using Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

### Install Poetry
```
pip install poetry
```

### Install Project Dependencies

```
poetry install
```

This sets up a virtual environment and installs dependencies from `pyproject.toml`.

### Running Python scripts using poetry

Run any script like so:

```
poetry run python src/data/data_download.py
```
---

## ✅ Linting: Code Quality Checks

We use both **pylint** and **flake8** to enforce code quality in the `src/` directory. These are automatically checked in CI (see badge above), but can also be run locally:

### 🔧 Run locally using poetry

```bash
# Important: PYTHONPATH=. is required so that custom pylint plugins are discoverable
PYTHONPATH=. poetry run pylint src/
poetry run flake8 src/
```

> ℹ️ We intentionally run linters **only on the `src/` folder** to exclude files like `__init__.py` for plugin registration and DVC-related files from lint scope.

> ⚠️ If you run `flake8` without specifying `src/`, you will see **intentional violations** such as:
> ```
> ./pylint_custom_checks/__init__.py:1:1: F401 '.hardcoded_params.register' imported but unused
> ```
> This is expected. The `register()` function must be imported to activate the custom plugin, even if not directly used.

### 🧠 Lint goals

- `pylint` includes a **custom checker** for detecting hardcoded ML hyperparameters (e.g. `learning_rate=0.01`).
- `flake8` enforces general formatting rules, with relaxed spacing and complexity rules defined in `.flake8`.

```bash
# .flake8 file is preconfigured to allow:
# - One blank line between functions
# - Line length up to 150
# - Ignored W503 (line break before binary op)
```
---

## 📌 Parameters with DVC

All pipeline parameters are stored in `params.yaml` and versioned using DVC. These are accessed in Python scripts using:

```
import dvc.api
params = dvc.api.params_show()
```

## 🔐 DVC Remote Access Setup

This project uses a shared Google Drive folder as a DVC remote. To push or pull data, you need to authenticate with your own Google account via the Drive API.

We strongly recommend setting up your own Google Cloud OAuth credentials to avoid "This app is blocked" errors. 

👉 Follow these instructions from the official DVC docs up untill step 6:
[Using a Custom Google Cloud project](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

Finally, use the `dvc remote modify` command to set the credentials:

```bash
dvc remote modify group20remote gdrive_client_id 'YOUR_CLIENT_ID' --local
dvc remote modify group20remote gdrive_client_secret 'YOUR_CLIENT_SECRET' --local
```

Then you can run

```bash
dvc pull
```

## 🔁 Reproducing Results with DVC

To reproduce the entire pipeline (download data, preprocess, train, evaluate) using the current parameters and dependencies:
```
dvc repro
```
This command will automatically detect changes in:
- Parameters (params.yaml)
- Code (src/)
- Data dependencies

...and re-run only the necessary stages. It's useful when:
- You've changed model parameters
- You've updated preprocessing or training scripts
- You need to reproduce results from scratch or on another machine

## External Model Usage

```python
import requests
from joblib import load
import os

os.makedirs("models", exist_ok=True)

model_url = "https://github.com/remla25-team20/model-training/releases/download/v0.1.4/Sentiment_Analysis_Model.joblib"
preprocessor_url = "https://github.com/remla25-team20/model-training/releases/download/v0.1.4/Sentiment_Analysis_Preprocessor.joblib"

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)

download_file(model_url, "models/Sentiment_Analysis_Model.joblib")
download_file(preprocessor_url, "models/Sentiment_Analysis_Preprocessor.joblib")

model = load("models/Sentiment_Analysis_Model.joblib")
preprocessor = load("models/Sentiment_Analysis_Preprocessor.joblib")
```
