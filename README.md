# 💳 End-to-End Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.0%2B-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Storage-cyan)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

A production-ready Machine Learning pipeline built with **PySpark**, **Delta Lake**, and **XGBoost** to detect fraudulent credit card transactions in highly imbalanced datasets (≈600:1 ratio).

---

## 🎯 Business Impact

**The Challenge:** Detect fraud in 284,807 transactions where only 0.17% are fraudulent.

**The Solution:** ML model that balances catching frauds (Recall) with minimizing false alarms (Precision).

### Key Results (Production Model - Nov 2025)

| Metric | Value | What It Means |
| :--- | :---: | :--- |
| **Recall** | 83.5% | Caught **76 out of 91** fraud cases |
| **Precision** | 75.2% | Only **1 false alarm per 3 real frauds** |
| **F1 Score** | 0.79 | Strong balance between metrics |

### 💰 Financial Impact

**Cost Assumptions:**
* Average fraud loss: **$500** per case
* Investigation cost: **$10** per alert

| Scenario | Outcome | Impact |
| :--- | :--- | ---: |
| **Without Model** | 91 frauds undetected | **-$45,500** |
| **With Model** | Miss only 15 frauds | **-$7,500** |
| **Operation Cost** | Review 101 alerts | **-$1,010** |
| **Net Savings** | — | **+$36,990** ✅ |

> **ROI: 37.6x** — For every $1 spent on investigation, prevent $38 in fraud losses.

---

## 🛠️ Technical Implementation

### Model Configuration

**XGBoost Classifier** with key parameters:
* `scale_pos_weight: 596.32` — Handles extreme class imbalance
* `max_depth: 5`, `n_estimators: 100` — Prevents overfitting
* `subsample: 0.8`, `colsample_bytree: 0.8` — Improves generalization

**Feature Engineering:**
* RobustScaler on `Amount` (handles outliers)
* 28 PCA-transformed features (V1-V28) + Time
* Total: 30 features

### Data Pipeline Architecture
```
Raw Data (Bronze) → Cleaned Data (Silver) → Features (Gold) → Model Training
     CSV              Delta Lake              Delta Lake         XGBoost
```

**Tech Stack:**
* **Processing:** PySpark for scalability
* **Storage:** Delta Lake (ACID compliance, versioning)
* **Deployment:** Docker containers
* **Model:** XGBoost with Spark integration

---

## 📊 Model Performance

### Confusion Matrix (Test Set: 56,784 transactions)

|  | Predicted Fraud | Predicted Normal |
| :--- | :---: | :---: |
| **Actual Fraud** | 76 ✅ | 15 ❌ |
| **Actual Normal** | 25 ⚠️ | 56,668 ✅ |

**Insights:**
* **76 True Positives** — Prevented $38,000 in losses
* **15 False Negatives** — Only 16.5% of frauds missed
* **25 False Positives** — Minimal wasted investigation effort

### Development vs. Production

| Phase | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| Prototype (5-Fold CV) | 0.860 | 0.757 | 0.788 |
| **Production** | **0.752** | **0.835** | **0.792** |

✅ Consistent performance shows good generalization

---

## 📂 Project Structure
```text
.
├── data/
│   ├── raw/           # Original CSV data
│   ├── silver/        # Cleaned data (Delta Lake)
│   └── gold/          # Feature-engineered data
├── models/
│   ├── xgboost_model/ # Trained model
│   └── pipeline/      # Feature pipeline
├── src/
│   ├── ingestion.py      # Data loading
│   ├── preprocessing.py  # Data cleaning
│   ├── features_engineering.py    # Feature engineering
│   └── training.py       # Model training
├── configs/           # Pipeline configurations
└── main.py            # Orchestrator
```

---

## 💾 Data Source

* **Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Provider:** Worldline & ULB (Université Libre de Bruxelles)
* **Size:** 284,807 transactions over 2 days
* **Features:** PCA-transformed for privacy (V1-V28)

---

## 🎯 Key Achievements

✅ **81% fraud loss prevention** with minimal operational cost  
✅ **37.6x ROI** on investigation spend  
✅ **Production-ready pipeline** with Docker + Delta Lake  
✅ **Scalable architecture** using PySpark  
✅ **Reproducible results** with version control

---

## 🔧 Technologies Used

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.10 |
| **ML Framework** | XGBoost, Scikit-learn |
| **Big Data** | PySpark, Delta Lake |
| **DevOps** | Docker, Git |

---

## 📝 Future Improvements

* Deploy as **real-time scoring API**
* Add **model drift monitoring**
* Implement **A/B testing framework**
* Build **automated retraining pipeline**