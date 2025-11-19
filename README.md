# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost**. This project demonstrates a scalable architecture (Bronze/Silver/Gold layers) to detect fraudulent transactions in highly imbalanced datasets.

---

## 🎯 Executive Summary & Business Impact

In credit card fraud detection, the primary goal is to minimize financial loss by catching as many fraudulent transactions as possible (**High Recall**) while maintaining a reasonable customer experience (**Manageable Precision**).

### Key Results (Test Set Performance after **scale_pos_weight** Tuning)
* **Recall (Fraud Capture Rate):** **83.9%**
    * *Impact:* The model successfully identified **73 out of 87** fraud cases in the unseen test set.
* **Precision:** **29.2%**
    * *Trade-off:* For every ~3.4 alerts, 1 is actual fraud. This result represents the optimal balance achieved during the **scale_pos_weight** tuning phase.
* **F1 Score:** **0.4332**
* **Accuracy:** **0.9966**

### 💰 Business Value Analysis
*Assumptions: Average loss per fraud = \$500 | Cost of manual review per alert = \$10*

| Scenario | Action | Economic Impact |
| :--- | :--- | :--- |
| **Without Model** | 87 frauds go unnoticed | **-\$43,500** (Loss) |
| **With This Model** | Catch 73 frauds, Miss 14 | **-\$7,000** (Loss from missed fraud) |
| **Operational Cost** | Review 250 alerts (73 TP + 177 FP) | **-\$2,500** (Labor cost) |
| **Net Savings** | Compared to no model | **+\$34,000 Saved** per batch |

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
| **Actual Fraud (1)** | **73 (TP)** ✅ | 14 (FN) ❌ |
| **Actual Normal (0)** | 177 (FP) ⚠️ | 56,521 (TN) |

* **False Negatives (14):** Low FN count validates the high effectiveness of the model at detecting actual fraud.
* **False Positives (177):** The focus of final optimization.

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
