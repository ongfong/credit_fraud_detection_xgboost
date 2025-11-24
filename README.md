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

📅 **Evaluation Date:** **2025-11-24**

* **Recall (Fraud Capture Rate):** **86.8%**
    * *Impact:* Successfully identified **79 out of 91** fraud cases in the test set.
* **Precision:** **91.9%**
    * *Efficiency:* For every **1.09 alerts**, 1 is confirmed fraud—delivering exceptional operational balance.
* **F1 Score:** **0.8927**
* **Accuracy:** **99.97%**

### 💰 Business Value Analysis

**Cost Assumptions:**
* Average fraud transaction value: **$500** per case
* Investigation cost per alert: **$10** per review

| Scenario | Action | Economic Impact |
| :--- | :--- | :--- |
| Without Model | 91 frauds undetected (91 × $500) | **-$45,500** (Total Loss) |
| With This Model | Catch 79 frauds, Miss 12 (12 × $500) | **-$6,000** (Residual Loss) |
| Operational Cost | Review 86 alerts: 79 TP + 7 FP (86 × $10) | **-$860** (Investigation Cost) |
| **Net Savings** | Fraud prevention benefit | **+$38,640 Saved** per batch |

> **Business Impact:** This model delivers **$38,640 in net savings** per evaluation batch, preventing **87% of potential fraud losses** (79/91 cases) while maintaining an ultra-lean review workload of just 86 alerts. The **92% precision rate** ensures the fraud investigation team receives exceptionally high-quality alerts, with only 1 false alarm per 11 real fraud cases.
>
> **ROI Calculation:** For every $1 spent on investigation ($860), the model prevents $46 in fraud losses ($39,500), delivering a **45.9x return on investment**.

---

## 💾 Data Source

The project utilizes a publicly available dataset of credit card transactions for European cardholders.

* **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Provider:** Worldline and ULB (Université Libre de Bruxelles)
* **Characteristics:** Contains transactions that occurred over two days, highly imbalanced (≈0.172% fraud rate), and features (V1-V28) are Principal Component Analysis (PCA) transformed for privacy.
* **Storage Location:** `data/raw/creditcard_raw.csv`

---

## 🔍 Exploratory Data Analysis & Prototyping

**Dataset Characteristics:**
* **Total Transactions:** 284,807 (2 days of European cardholder data)
* **Class Distribution:** 492 frauds vs. 284,315 legitimate
* **Class Imbalance Ratio:** **598.84:1** (severe imbalance)

**Feature Analysis:**

| Analysis | Finding | Impact |
| :--- | :--- | :--- |
| **Correlation (Top 6)** | V17, V14, V12, V10, V16, V3 | Strongest fraud indicators |
| **Amount Distribution** | Right-skewed (mean > median), frauds typically involve smaller amounts | Requires scaling |
| **Amount Outliers** | **11.17%** of transactions | → **RobustScaler** selected for outlier resistance |
| **Time Patterns** | No significant difference between fraud/legitimate | Retained but lower priority |

---

### 🧪 Prototype Development Process

**Data Splitting Strategy:**
```
Full Dataset (284,807)
    ├─ Train Set (227,845) ──→ Sample for CV (22,698)
    └─ Test Holder (56,779)   [Hold out for final evaluation]
```

**Phase 1: Model Selection (5-Fold Cross-Validation)**

Evaluated 5 candidate algorithms on sampled training data:

| Model | ROC-AUC | Precision | Recall | F1 Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9519** | 0.9314 | 0.7571 | 0.8314 | 🥇 |
| **Logistic Regression** | 0.9164 | 0.0256 | **0.8357** | 0.0496 | 🥈 |
| Random Forest | 0.9022 | 0.9214 | 0.6250 | 0.7398 | - |
| LightGBM | 0.8610 | **0.9600** | 0.7571 | **0.8429** | - |
| Decision Tree | 0.7748 | 0.6628 | 0.5500 | 0.5923 | - |

> **Implementation Note:**  
> XGBoost uses `scale_pos_weight=596.32` while other models use `class_weight='balanced'` for handling the 598.84:1 class imbalance.

**Model Selection Rationale:**
* **XGBoost chosen** despite Logistic Regression's higher Recall (0.836 vs 0.757) because:
  1. **Higher ROC-AUC (0.9519 vs 0.9164)** — 3.9% better class discrimination capability
  2. **Significantly better Precision (0.9314 vs 0.0256)** — 36x fewer false alarms
  3. **Balanced metrics** — Strong F1 Score (0.831 vs 0.050) indicates better overall performance
  4. **Threshold tuning potential** — Can increase Recall in production while maintaining reasonable Precision

**Phase 2: Finalist Evaluation**

Top 2 models (XGBoost, Logistic Regression) further analyzed with:
* **Confusion Matrix** — Error pattern analysis
* **ROC Curve** — Discrimination capability visualization

**Winner: XGBoost** confirmed for superior balance between Precision and Recall with highest ROC-AUC.

**Phase 3: Hyperparameter Tuning**

Applied **GridSearchCV** on XGBoost using sampled training data (22,698 samples):

| Parameter | Search Range | Best Value |
| :--- | :--- | :---: |
| `n_estimators` | [100, 200] | **100** |
| `max_depth` | [3, 5, 7] | **7** |
| `learning_rate` | [0.01, 0.1, 0.2] | **0.2** |
| `subsample` | [0.8, 1.0] | **0.8** |
| `colsample_bytree` | [0.8, 1.0] | **0.8** |

**Best Configuration Saved:** `configs/prototype_config.json`

---

**Prototype Summary (5-Fold CV on 22,698 samples):**

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9519** | Highest among all candidates (3.9% better than Logistic Regression) |
| **Precision** | **0.9314** | 93% of alerts are true fraud (36x better than Logistic Regression) |
| **Recall** | **0.7571** | Catches 76% of fraud cases |
| **F1 Score** | **0.8314** | Strong balance between Precision and Recall |

**Key Modeling Decisions:**
1. **XGBoost selected** — Best ROC-AUC (0.9519) with superior Precision-Recall balance
2. **`scale_pos_weight=596.32`** — Directly addresses the 598.84:1 class imbalance
3. **RobustScaler for Amount** — Handles 11.17% outliers without distortion
4. **Hyperparameters optimized** — GridSearchCV tuned: `max_depth=7`, `learning_rate=0.2`, `n_estimators=100`
5. **Balanced approach** — High Precision (93%) with acceptable Recall (76%), allowing threshold tuning for production

---

## ⚙️ Model Configuration

**XGBoost Hyperparameters (Production):**
* `scale_pos_weight: 596.32` — Handles extreme class imbalance
* `max_depth: 7` — Allows deeper trees for complex patterns
* `n_estimators: 100` — Number of boosting rounds
* `learning_rate: 0.2` — Faster convergence with higher learning rate
* `subsample: 0.8` — Row sampling ratio per tree
* `colsample_bytree: 0.8` — Feature sampling ratio per tree
* `random_state: 42` — Ensures reproducibility

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

### Confusion Matrix (Test Set: 56,779 transactions)

| | Predicted Fraud (1) | Predicted Normal (0) |
| :--- | :---: | :---: |
| **Actual Fraud (1)** | **79 (TP)** ✅ | 12 (FN) ❌ |
| **Actual Normal (0)** | 7 (FP) ⚠️ | 56,681 (TN) |

**Key Insights:**
* **True Positives (79):** Strong fraud detection capability preventing major losses ($39,500 saved)
* **False Negatives (12):** Only 13.2% of frauds missed—excellent fraud capture rate
* **False Positives (7):** Exceptional false alarm rate requiring only 7 unnecessary reviews
* **True Negatives (56,681):** Legitimate transactions correctly classified with 99.99% specificity

### Performance Comparison: Prototype vs. Production

| Phase | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Prototype (5-Fold CV)** | 0.9314 | 0.7571 | 0.8314 | Stratified cross-validation on 22,698 samples |
| **Production (Test Set)** | 0.9186 | 0.8681 | 0.8927 | Final evaluation on 56,779 samples |

> **Validation Success:** The production model demonstrates exceptional performance improvement over the prototype. Recall increased significantly (75.7% → 86.8%), while Precision remained high (93.1% → 91.9%), resulting in a superior F1 Score (0.831 → 0.893). This validates the model's excellent generalization and its ability to achieve the optimal balance between catching fraud and minimizing false alarms.

### Cross-Validation Metrics (Prototype Phase)

The model demonstrated robust performance during the prototype phase with 5-fold stratified cross-validation:

* **ROC-AUC:** 0.9519 — Excellent discrimination between classes
* **CV Precision:** 0.9314 — 93% of alerts are true fraud
* **CV Recall:** 0.7571 — Catches 76% of fraud cases
* **CV F1 Score:** 0.8314 — Strong balance between Precision and Recall
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