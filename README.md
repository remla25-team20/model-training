[![Pylint](https://github.com/remla25-team20/model-training/actions/workflows/pylint.yml/badge.svg)](https://github.com/remla25-team20/model-training/actions/workflows/pylint.yml)

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

## 📌 Parameters with DVC

All pipeline parameters are stored in `params.yaml` and versioned using DVC. These are accessed in Python scripts using:

```
import dvc.api
params = dvc.api.params_show()
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
