# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A scalable, production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost**. This project demonstrates an **MLOps-focused architecture** (Bronze/Silver/Gold layers) to effectively detect fraudulent transactions in highly imbalanced datasets.

---

## 🎯 Executive Summary & Business Impact

The core challenge in credit card fraud detection is the class imbalance (≈600:1). The strategy prioritizes **High Recall** to minimize financial loss, while managing **Precision** to maintain an acceptable operational cost.

### Key Results (Production Model Performance)

📅 **Evaluation Date:** **2025-11-21**

* **Recall (Fraud Capture Rate):** **83.5%**
    * *Impact:* Successfully identified **76 out of 91** fraud cases in the test set.
* **Precision:** **75.2%**
    * *Efficiency:* For every **1.33 alerts**, 1 is confirmed fraud—delivering excellent operational balance.
* **F1 Score:** **0.7917**
* **Accuracy:** **99.93%**

### 💰 Business Value Analysis

**Cost Assumptions:**
* Average fraud transaction value: **$500** per case
* Investigation cost per alert: **$10** per review

| Scenario | Action | Economic Impact |
| :--- | :--- | :--- |
| Without Model | 91 frauds undetected (91 × $500) | **-$45,500** (Total Loss) |
| With This Model | Catch 76 frauds, Miss 15 (15 × $500) | **-$7,500** (Residual Loss) |
| Operational Cost | Review 101 alerts: 76 TP + 25 FP (101 × $10) | **-$1,010** (Investigation Cost) |
| **Net Savings** | Fraud prevention benefit | **+$36,990 Saved** per batch |

> **Business Impact:** This model delivers **$37,000 in net savings** per evaluation batch, preventing **81% of potential fraud losses** (76/91 cases) while maintaining a lean review workload of just 101 alerts. The **75% precision rate** ensures the fraud investigation team receives high-quality alerts, with only 1 false alarm per 3 real fraud cases.
>
> **ROI Calculation:** For every $1 spent on investigation ($1,010), the model prevents $38 in fraud losses ($38,000), delivering a **37.6x return on investment**.

---

## 💾 Data Source

The project utilizes a publicly available dataset of credit card transactions for European cardholders.

* **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Provider:** Worldline and ULB (Université Libre de Bruxelles)
* **Characteristics:** Contains transactions that occurred over two days, highly imbalanced (≈0.172% fraud rate), and features (V1-V28) are Principal Component Analysis (PCA) transformed for privacy.
* **Storage Location:** `data/raw/creditcard_raw.csv`

---

## ⚙️ Model Configuration

**XGBoost Hyperparameters:**
* `scale_pos_weight: 596.32` — Handles extreme class imbalance
* `max_depth: 5`, `n_estimators: 100` — Prevents overfitting
* `subsample: 0.8`, `colsample_bytree: 0.8` — Improves generalization

**Feature Engineering:**
* RobustScaler on `Amount` (outlier-resistant)
* 28 PCA-transformed features (V1-V28) + Time
* Total: 30 features

---

## 🛠️ Technical Architecture

* **Data Lakehouse:** Medallion Architecture (Bronze → Silver → Gold) using **Delta Lake**
* **Processing:** **PySpark** for scalable ETL and feature engineering
* **ML Framework:** `xgboost.spark.SparkXGBClassifier` for distributed training
* **Deployment:** **Docker** containers for reproducibility
* **Pipeline:** Serialized feature transformations (`models/pipeline/feature_pipeline`)

---

## 📊 Model Performance Details

### Confusion Matrix (Test Set: 56,784 transactions)

| | Predicted Fraud (1) | Predicted Normal (0) |
| :--- | :---: | :---: |
| **Actual Fraud (1)** | **76 (TP)** ✅ | 15 (FN) ❌ |
| **Actual Normal (0)** | 25 (FP) ⚠️ | 56,668 (TN) |

**Key Insights:**
* **True Positives (76):** Strong fraud detection capability preventing major losses ($38,000 saved)
* **False Negatives (15):** Only 16.5% of frauds missed—acceptable for high-stakes scenarios
* **False Positives (25):** Minimal false alarm rate requiring only 25 unnecessary reviews
* **True Negatives (56,668):** Legitimate transactions correctly classified with 99.96% specificity

### Performance Comparison: Prototype vs. Production

| Phase | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Prototype (5-Fold CV)** | 0.860 | 0.757 | 0.788 | Stratified cross-validation on 22,698 samples |
| **Production (Test Set)** | 0.752 | 0.835 | 0.792 | Final evaluation on 56,784 samples |

> **Validation Success:** The production model maintains consistent performance with the prototype, demonstrating excellent generalization. The slight precision trade-off for higher recall aligns with the business priority of minimizing fraud losses.

### Cross-Validation Metrics (Prototype Phase)

The model demonstrated robust performance during the prototype phase with 5-fold stratified cross-validation:

* **ROC-AUC:** 0.952 — Excellent discrimination between classes
* **CV Precision:** 0.860 — High confidence in fraud predictions
* **CV Recall:** 0.757 — Strong fraud capture rate
* **Training Set:** 22,698 samples with 0.167% fraud ratio

---

## 📂 Repository Structure
```text
.
├── configs/                # Pipeline configurations (JSON)
├── data/                   # Delta Lake tables & raw data
│   ├── raw/               # Bronze layer: creditcard_raw.csv
│   ├── silver/            # Silver layer: cleaned data
│   └── gold/              # Gold layer: feature-engineered data
├── models/                 # Trained models & feature pipelines
│   ├── xgboost_model/     # Serialized XGBoost model
│   └── pipeline/          # Feature transformation pipeline
├── notebooks/              # EDA & prototyping (Jupyter)
├── src/                    # Core implementation modules
│   ├── ingestion.py       # Data ingestion
│   ├── preprocessing.py   # Data cleaning
│   ├── features_engineering.py  # Feature engineering
│   └── training.py        # Model training
└── main.py                # Pipeline orchestrator
```