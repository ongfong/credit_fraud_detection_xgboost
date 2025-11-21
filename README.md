# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A scalable, production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost**. This project demonstrates an **MLOps-focused architecture** (Bronze/Silver/Gold layers) to effectively detect fraudulent transactions in highly imbalanced datasets.

---

## 🎯 Executive Summary & Business Impact

The core challenge in credit card fraud detection is the extreme class imbalance (≈600:1 ratio). This pipeline addresses the challenge by optimizing for **High Recall** to capture maximum fraud cases while maintaining strong **Precision** to keep operational costs manageable.

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

This project uses a real-world dataset of European credit card transactions.

* **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Provider:** Worldline and ULB (Université Libre de Bruxelles)
* **Dataset Profile:** Two-day transaction window with severe class imbalance (≈0.172% fraud rate). Features V1-V28 are PCA-transformed for cardholder privacy protection.
* **Storage Location:** `data/raw/creditcard_raw.csv`

---

## ⚙️ Model Architecture & Configuration

### XGBoost Hyperparameters (Production Model)

The model uses carefully tuned parameters to balance performance and generalization:

| Parameter | Value | Purpose |
| :--- | :---: | :--- |
| `scale_pos_weight` | **596.32** | Addresses class imbalance by penalizing missed fraud |
| `learning_rate` | **0.1** | Controls training step size |
| `max_depth` | **5** | Limits tree complexity to prevent overfitting |
| `n_estimators` | **100** | Number of boosting rounds |
| `subsample` | **0.8** | Row sampling ratio per tree |
| `colsample_bytree` | **0.8** | Feature sampling ratio per tree |
| `random_state` | **42** | Ensures reproducibility |

### Feature Engineering Pipeline

**Preprocessing Strategy:**
* **RobustScaler** applied to `Amount` feature (resistant to outliers)
* **Passthrough columns:** Time, V1-V28 (PCA-transformed features used as-is)
* **Total features:** 30 (Time + V1-V28 + Amount)

**Pipeline Architecture:**
```
Input → ColumnTransformer → XGBoost Classifier → Predictions
         (RobustScaler)      (Spark-enabled)
```

---

## 📊 Model Performance Analysis

### Confusion Matrix (Test Set)

| | Predicted Fraud (1) | Predicted Normal (0) |
| :--- | :---: | :---: |
| **Actual Fraud (1)** | **76 (TP)** ✅ | 15 (FN) ❌ |
| **Actual Normal (0)** | 25 (FP) ⚠️ | 56,668 (TN) |

**Key Insights:**
* **True Positives (76):** Strong fraud detection capability preventing major losses
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

## 🛠️ Technical Architecture & MLOps Features

This production-grade system implements modern MLOps best practices:

### Data Platform
* **Medallion Architecture:** Bronze (raw) → Silver (cleaned) → Gold (feature-engineered) layers
* **Storage:** Delta Lake for ACID transactions, versioning, and time-travel capabilities
* **Processing:** PySpark for distributed data transformation at scale

### ML Pipeline
* **Framework:** `xgboost.spark.SparkXGBClassifier` for distributed training
* **Serialization:** Feature pipeline saved to `models/pipeline/feature_pipeline` for inference consistency
* **Versioning:** Model metadata and configurations tracked with timestamps

### Infrastructure
* **Containerization:** Docker + Docker Compose for environment reproducibility
* **Orchestration:** `main.py` serves as the central pipeline coordinator
* **Scalability:** Designed for horizontal scaling with Spark cluster deployment

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
│   ├── ingest.py          # Data ingestion
│   ├── preprocess.py      # Data cleaning
│   ├── features.py        # Feature engineering
│   └── train.py           # Model training
└── main.py                 # Pipeline orchestrator
```

---

## 🔧 Technology Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| Language | Python | 3.10.18 |
| ML Framework | XGBoost | Spark-enabled |
| ML Library | Scikit-learn | 1.3.2 |
| Data Processing | PySpark | 3.0+ |
| Data Storage | Delta Lake | Latest |
| Containerization | Docker | Latest |

---

## 📈 Model Metadata

* **Model Type:** XGBoost Binary Classifier
* **Training Timestamp:** 2025-11-21 07:44:11 UTC
* **Configuration Created:** 2025-11-21 07:18:11 UTC
* **Pipeline Creator:** prototype_pipeline
* **Training Dataset:** 22,698 transactions (stratified split)
* **Test Dataset:** 56,784 transactions
* **Class Distribution:** ≈0.167% fraud rate (1:596 ratio)

---

## 🎯 Production Readiness

This model is production-ready with the following characteristics:

✅ **High Recall (83.5%)** — Captures most fraud cases  
✅ **Strong Precision (75.2%)** — Minimizes false alarms  
✅ **Balanced F1 (0.79)** — Optimal trade-off for operations  
✅ **Reproducible Pipeline** — Containerized and version-controlled  
✅ **Scalable Architecture** — Built on distributed computing frameworks  
✅ **Cost-Effective** — Delivers $37K+ savings per batch

---

## 📝 Future Enhancements

* **Threshold Optimization:** Fine-tune decision threshold for specific business requirements
* **Real-time Inference:** Deploy as streaming service for live transaction scoring
* **Model Monitoring:** Implement drift detection and performance tracking
* **Feature Store:** Centralize feature definitions for consistency across models
* **A/B Testing:** Framework for comparing model versions in production