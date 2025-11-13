# Credit Fraud Detection — XGBoost pipeline

This repository contains a small end-to-end credit card fraud detection pipeline using PySpark, Delta Lake, and XGBoost. It demonstrates ingestion, preprocessing, feature engineering (Spark ML pipeline), training, and model export.

Quick summary:

- End-to-end Spark-based ML pipeline (ingestion → preprocessing → feature engineering → training)
- Saved Spark feature pipeline: `models/pipeline/feature_pipeline`
- Trained XGBoost model and metrics: `models/production/xgboost_model/` (includes `metrics.json`)
- Optional evaluation artifacts (predictions CSV, ROC/PR plots) under `models/production/xgboost_model/evaluation/` when generated
- Example notebooks in `notebooks/` and sample raw data in `data/raw/`
- Docker/Docker Compose support for reproducible runs

## Key features

- Batch ETL using PySpark and Delta Lake (bronze -> silver -> gold tables)
- Reusable Spark feature pipeline serialized under `models/pipeline/feature_pipeline`
- XGBoost model trained and exported to `models/production/xgboost_model`
- Example notebooks for EDA and prototyping in `notebooks/`
- Docker / docker-compose support for containerized runs

## Repository layout

Top-level files and important folders:

- `main.py` — orchestrates the full pipeline (ingestion -> preprocess -> features -> train)
- `requirements.txt` — Python dependencies
- `docker-compose.yml`, `Dockerfile` — containerized environment
- `configs/prototype_config.json` — pipeline configuration used by feature engineering and training
- `data/` — sample data and Delta Lake folders (bronze / silver / raw)
- `models/` — pipeline and model artifacts (feature pipeline + trained XGBoost)
- `notebooks/` — exploratory notebooks (`EDA.ipynb`, `prototype.ipynb`)
- `src/` — implementation modules (01_ingestion.py, 02_preprocessing.py, 03_feature_engineering.py, 04_training.py)

## Prerequisites

- Python 3.9+ recommended
- Docker & docker-compose (optional, only if you want containerized execution)

This project uses PySpark and Delta Lake. Installation of the dependencies listed in `requirements.txt` is required.

## Setup (local / virtualenv)

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

3. Verify the `data/raw/creditcard_raw.csv` exists (the repo contains a sample under `data/raw`).

## Run the full pipeline (local)

From the repo root you can run the pipeline entrypoint which will:

- create/read Delta tables in `data/bronze` and `data/silver`
- run feature engineering and save the feature pipeline under `models/pipeline/feature_pipeline`
- train an XGBoost model and save artifacts under `models/production/xgboost_model`

```powershell
python main.py
```

If you need to change paths or parameters, edit `configs/prototype_config.json` or the variables at the top of `main.py`.

## Run with Docker / docker-compose

The repository includes a `Dockerfile` and `docker-compose.yml` for a reproducible environment. To build and run:

```powershell
docker-compose up --build
```

This will create a container with the same pipeline entrypoint. Adjust the compose file if you want to mount external data.

## Notebooks

- `notebooks/EDA.ipynb` — exploratory data analysis and visual checks
- `notebooks/prototype.ipynb` — prototyping and quick experiments

Open them with JupyterLab (installed in `requirements.txt`):

```powershell
jupyter lab
```

## Artifacts and outputs

- Bronze / Silver Delta tables: `data/bronze/bronze_creditcard`, `data/silver/silver_creditcard`
- Feature pipeline: `models/pipeline/feature_pipeline/`
- Trained model + metrics: `models/production/xgboost_model/` (JSON + metrics files)

### Model evaluation artifacts

When the pipeline runs it saves an evaluation summary under the model folder. New outputs produced by the training step include:

- `models/production/xgboost_model/metrics.json` — summary metrics (accuracy, precision, recall, f1) plus extended metrics `auc_roc` and `auc_pr` when available.
- `models/production/xgboost_model/evaluation/predictions.csv` — optional CSV of test set predictions (columns: true label, prediction, probability_one). This is saved only when the test set is reasonably small to avoid driver OOMs.
- `models/production/xgboost_model/evaluation/roc_curve.png` and `pr_curve.png` — optional ROC and Precision-Recall plots generated for small test sets.

If the test set is large, the training script will skip saving the full predictions and plots to avoid collecting a large DataFrame on the driver; the metrics JSON will still contain the numeric evaluation results.

## Model performance summary (from `models/production/xgboost_model/metrics.json`)

The repository includes a saved metrics file produced by the last training run. Below is a concise snapshot taken from `models/production/xgboost_model/metrics.json`.

- Trained date: 2025-11-13T07:05:48.740307
- Model: XGBoost (xgboost.spark.SparkXGBClassifier)

Primary evaluation (test set):

- Accuracy: 0.998098
- Precision: 0.451220
- Recall: 0.804348
- F1 score: 0.578125

Confusion matrix (test set):

- True Positive (TP): 74
- False Positive (FP): 90
- True Negative (TN): 56,593
- False Negative (FN): 18

Cross-validated prototype performance (from `prototype_config`):

- CV ROC AUC: 0.958398
- CV Precision: 0.660476
- CV Recall: 0.707143
- CV F1: 0.679048
- CV folds: 5
- Number of samples used in prototype CV: 22,698
- Fraud ratio (prototype): 0.001674

Notes:

- These numbers are a snapshot and will change when you retrain the model. Run `python main.py` to produce a fresh `metrics.json` in `models/production/xgboost_model/`.
- If you need ROC AUC / PR AUC values for the current trained model, re-run the pipeline with the extended evaluation enabled (the training script will add `auc_roc` and `auc_pr` to the metrics file when available).

## Development notes and assumptions

- The project uses a local Spark session configured in `spark_config.py`. It’s designed for development and small datasets. For production-scale runs, point Spark to a cluster and adjust Delta storage accordingly.
- The pipeline is implemented as sequential stages in `main.py` and the `src/` modules. You can call individual modules directly for development or testing (for example, import and run functions from `src/feature_engineering.py`).

## Troubleshooting

- If PySpark or Delta initialization fails, check Java is installed and the environment variables are correct for Spark. On Windows, ensure compatible Java version (OpenJDK 11+) is available.
- If you see dependency errors, recreate the virtual environment and run `pip install -r requirements.txt`.

## Next steps / ideas

- Add inference script and sample API wrapper for model serving.
- Add a Quick API using ONNX (FastAPI + onnxruntime): include a conversion script (`scripts/convert_to_onnx.py`) and a small FastAPI app (`api/app.py`) to serve `models/production/xgboost_model/xgb_model.onnx`.

### Quick API using ONNX

You can expose the trained model via a lightweight FastAPI service that serves an ONNX model using ONNX Runtime.

## License

This project is licensed under the MIT License — see the top-level `LICENSE` file for details. (SPDX: MIT)