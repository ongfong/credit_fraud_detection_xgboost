# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost**. This project demonstrates a scalable architecture (Bronze/Silver/Gold layers) to detect fraudulent transactions in highly imbalanced datasets.

---

## 🎯 Executive Summary & Business Impact

In credit card fraud detection, the primary goal is to minimize financial loss by catching as many fraudulent transactions as possible (**High Recall**) while maintaining a reasonable customer experience (**Manageable Precision**).

### Key Results (Test Set Performance)
* **Recall (Fraud Capture Rate):** **83.3%**
    * *Impact:* The model successfully identified **90 out of 108** fraud cases in the unseen test set.
* **Precision:** **52.3%**
    * *Trade-off:* For every ~2 alerts, 1 is actual fraud. This is an acceptable operational cost compared to the high cost of missed fraud.
* **ROC-AUC:** **0.958**

### 💰 Business Value Analysis
*Assumptions: Average loss per fraud = \$500 | Cost of manual review per alert = \$10*

| Scenario | Action | Economic Impact |
| :--- | :--- | :--- |
| **Without Model** | 108 frauds go unnoticed | **-\$54,000** (Loss) |
| **With This Model** | Catch 90 frauds, Miss 18 | **-\$9,000** (Loss from missed fraud) |
| **Operational Cost** | Review 172 alerts (90 TP + 82 FP) | **-\$1,720** (Labor cost) |
| **Net Savings** | Compared to no model | **+\$43,280 Saved** per batch |

> **Verdict:** This pipeline potentially saves the company **~80% of fraud losses** with a manageable workload for the fraud investigation team.

---

## 🛠️ Tech Stack & Architecture

This project implements a **Medallion Architecture** using a data lakehouse approach:

1.  **Ingestion (Bronze):** Raw data ingestion into Delta Lake.
2.  **Preprocessing (Silver):** Cleaning and schema validation.
3.  **Feature Engineering:** Spark ML Pipelines (serialized for inference).
4.  **Modeling:** Distributed training with **XGBoost on Spark**.
5.  **Deployment:** Model artifacts exported to `models/production/`.
6.  **Reproducibility:** Fully containerized with **Docker** & **Docker Compose**.

---

## 📊 Model Performance Details

**Confusion Matrix (Test Set Snapshot):**

| | Predicted Fraud (1) | Predicted Normal (0) |
| :--- | :---: | :---: |
| **Actual Fraud (1)** | **90 (TP)** ✅ | 18 (FN) ❌ |
| **Actual Normal (0)** | 82 (FP) ⚠️ | 56,588 (TN) |

* **False Negatives (18):** The most critical metric. Future improvements will aim to reduce this further via threshold tuning.
* **False Positives (82):** Represents <0.15% of legitimate customers being flagged, minimizing friction.

---

## 📂 Repository Structure

```text
├── .github/                # CI/CD configurations
├── configs/                # Pipeline configurations (JSON)
├── data/                   # Delta Lake storage (Bronze/Silver) & Raw data
├── models/                 # Saved models & Feature Pipelines
├── notebooks/              # EDA & Prototyping (Jupyter)
├── src/                    # Source code (Ingest, Preprocess, Feature, Train)
├── main.py                 # Pipeline Orchestrator
├── Dockerfile              # Docker image definition
└── docker-compose.yml      # Container orchestration
