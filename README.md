# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A scalable, production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost**. This project demonstrates an **MLOps-focused architecture** (Bronze/Silver/Gold layers) to effectively detect fraudulent transactions in highly imbalanced datasets.

---

## 🎯 Executive Summary & Business Impact

The core challenge in credit card fraud detection is the class imbalance (≈600:1). The strategy prioritizes **High Recall** to minimize financial loss, while managing **Precision** to maintain an acceptable operational cost.

### Key Results (Test Set Performance after `scale_pos_weight` Tuning)

* **Evaluation Date:** **2025-11-19** 📅
* **Recall (Fraud Capture Rate):** **83.9%**
    * *Impact:* The base model successfully identified **73 out of 87** total fraud cases in the unseen test set.
* **Precision:** **29.2%**
    * *Trade-off:* For every **~3.4 alerts**, 1 is actual fraud. This result represents the optimal base model balance achieved before final operational tuning.
* **F1 Score:** **0.4332**
* **Accuracy:** **0.9966**

### 💰 Business Value Analysis

| Scenario | Action | Economic Impact |
| :--- | :--- | :--- |
| Without Model | 87 frauds go unnoticed | **-$43,500** (Loss) |
| With This Model | Catch 73 frauds, Miss 14 | **-$7,000** (Loss from missed fraud) |
| Operational Cost | Review 250 alerts (73 TP + 177 FP) | **-$2,500** (Labor cost) |
| **Net Savings** | Compared to no model | **+$34,000 Saved** per batch |

> **Verdict:** This pipeline provides a net saving of **~$34,000** per batch, confirming that the cost of manual review is significantly lower than the cost of missed fraud. The base model saves **~80% of fraud losses**.

---

## 💾 Data Source

The project utilizes a publicly available dataset of credit card transactions for European cardholders.

* **Source:** Kaggle: Credit Card Fraud Detection
* **Provider:** Worldline and ULB (Université Libre de Bruxelles)
* **Characteristics:** Contains transactions that occurred over two days, highly imbalanced (≈0.172% fraud rate), and features (V1-V28) are Principal Component Analysis (PCA) transformed for privacy.
* **Storage Location:** `data/raw/creditcard_raw.csv`

## ⚙️ Model Tuning Strategy: Achieving Balance

This section details the **two-stage optimization strategy** implemented to overcome the severe class imbalance (Normal:Fraud ratio ≈ 600:1) and build a model suitable for operational use.

### Stage 1: Iterative `scale_pos_weight` Tuning

We focused on tuning the `scale_pos_weight` parameter in XGBoost to artificially increase the cost of **False Negatives** (missing a fraud).

| Phase | `scale_pos_weight` | Precision | Recall | False Positive (FP) |
| :--- | :---: | :---: | :---: | :---: |
| Initial (Ratio) | 596.31 | 0.087 | 0.900 | 847 |
| **Optimal Base** | **200.00** | **0.292** | **0.839** | **177** |

> **Process Note:** Reducing the weight from the calculated ratio (596.31) to 200.00 provided the highest F1 Score (0.4332) while stabilizing the False Positive rate (from 847 to 177), establishing a strong base model.

### Stage 2: Final Threshold Optimization (The Next Step)

The high-Recall base model is now ready for final tuning to meet the operational Precision target (e.g., ≈50%) required by the business.

| Action | Goal | Rationale |
| :--- | :--- | :--- |
| **Adjust Threshold** | Reduce False Positives from **177** to **~85-90** | Meet the operational target (Precision ≈ 50%) for the fraud investigation team. |
| **Method** | Adjust prediction threshold from default 0.50 to a higher value (e.g., 0.75-0.85). | Leverage the strong probability outputs of the base model (scale\_pos\_weight = 200.00). |

---

## 🛠️ Technical Architecture & Key Features

This project emphasizes scalability and reproducibility, key aspects of MLOps.

* **Data Lakehouse:** Implementation of a **Medallion Architecture** using **Delta Lake** (Bronze -> Silver -> Gold tables) for reliable, versioned data storage.
* **Scalable ETL:** Batch data transformation and feature engineering handled efficiently using **PySpark**.
* **ML Pipeline:** The Spark feature pipeline is **serialized** (`models/pipeline/feature_pipeline`) to ensure consistent, production-ready feature calculation during both training and inference.
* **Containerization:** Full environment packaging using **Docker** and **Docker Compose** for reproducible execution.

---

## 📊 Model Performance Details

**Confusion Matrix (Snapshot post-Tuning):**

| | Predicted Fraud (1) | Predicted Normal (0) |
| :--- | :---: | :---: |
| **Actual Fraud (1)** | **73 (TP)** ✅ | 14 (FN) ❌ |
| **Actual Normal (0)** | 177 (FP) ⚠️ | 56,514 (TN) |

* **False Negatives (14):** The model only missed 14 cases, highlighting its strength in minimizing loss.
* **False Positives (177):** These are the focus for the next step (Threshold Tuning) to reduce the operational workload.

---

## 📂 Repository Structure

```text
├── configs/                # Pipeline configurations (JSON)
├── data/                   # Delta Lake storage (Bronze/Silver) & Raw data
├── models/                 # Saved models & Feature Pipelines
├── notebooks/              # EDA & Prototyping (Jupyter)
├── src/                    # Implementation modules (Ingest, Preprocess, Feature, Train)
└── main.py                 # Pipeline Orchestrator Entrypoint
