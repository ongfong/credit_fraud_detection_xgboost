# Databricks notebook source
# MAGIC %pip install xgboost lightgbm

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import sys
import sklearn
import json
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CreditCardFraud") \
    .master("local[*]") \
    .getOrCreate()

# COMMAND ----------

df_silver = spark.table("silver.silver_creditcard")

# COMMAND ----------

df_full = df_silver.toPandas()

# COMMAND ----------

df_full.head()

# COMMAND ----------

df_full.drop(['create_silver_timestamp','ingestion_timestamp'], axis=1, inplace=True)

# COMMAND ----------

df_full.head()

# COMMAND ----------

X_full = df_full.drop('Class',axis=1)
y_full = df_full['Class']

# COMMAND ----------

X_train_full, X_test_holder, y_train_full, y_test_holder = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)

# COMMAND ----------

X_train_full.shape

# COMMAND ----------

X_train_sample, _, y_train_sample, _ = train_test_split(X_train_full, y_train_full, test_size=0.9, stratify=y_train_full, random_state=42)

# COMMAND ----------

X_train_sample.shape

# COMMAND ----------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# COMMAND ----------

preprocessor = ColumnTransformer(transformers=[('robust', RobustScaler(), ['Amount'])], remainder='passthrough')

# COMMAND ----------

scoring = {
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score, pos_label=1, zero_division=0),
    'recall': make_scorer(recall_score, pos_label=1, zero_division=0),
    'f1': make_scorer(f1_score, pos_label=1, zero_division=0)
}

# COMMAND ----------

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight= (len(y_train_sample) - y_train_sample.sum()) / y_train_sample.sum(), random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
}

# COMMAND ----------

results = []

for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv_results = cross_validate(pipeline, X_train_sample, y_train_sample, cv=cv, scoring=scoring)
    results.append({
        'model': name,
        'roc_auc': cv_results['test_roc_auc'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'pipe': pipeline
    })

# COMMAND ----------

results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
results_df_display = results_df.drop('pipe', axis=1)
print(results_df_display)

# COMMAND ----------

top2_models = results_df.head(2)

plt.figure(figsize=(12, 5), facecolor='white')
ax = plt.gca()

for idx, row in top2_models.iterrows():
    model_name = row['model']
    pipe = row['pipe']

    y_pred_sample = cross_val_predict(pipe, X_train_sample, y_train_sample, cv=cv, method='predict_proba')[:, 1]
    RocCurveDisplay.from_predictions(y_train_sample, y_pred_sample, name=model_name, ax=ax)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Top 2 Models (Cross-Validated)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

fig, axes = plt.subplots(1,2, figsize=(15, 4))

for ax, (_, row) in zip(axes, top2_models.iterrows()):
    
    model_name = row['model']
    pipe = row['pipe'] 
    
    y_pred_sample = cross_val_predict(pipe, X_train_sample, y_train_sample, cv=cv)
    
    # confusion matrix
    cm = confusion_matrix(y_train_sample, y_pred_sample)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'{model_name}\nCV_ROC_AUC: {row["roc_auc"]:.3f}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC hyperparameter tuning

# COMMAND ----------

scoring_tune = {
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1)
}

# COMMAND ----------

param_grid_xgb = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1],
    'model__colsample_bytree': [0.8, 1]
}

# COMMAND ----------

xgb_pipe = results_df.loc[results_df['model'] == 'XGBoost','pipe'].values[0]

# COMMAND ----------

gric_xgb = GridSearchCV(xgb_pipe, param_grid_xgb, cv=cv, scoring=scoring_tune, refit='roc_auc', verbose=1)
gric_xgb.fit(X_train_sample, y_train_sample)
print(gric_xgb.best_params_)

# COMMAND ----------

config = {
    'model': results_df.loc[results_df['model'] == 'XGBoost','model'].values[0],
    'tuned_params': {k.replace('model__', ''): v for k, v in gric_xgb.best_params_.items()},
    'fixed_params': {
        'scale_pos_weight': float((len(y_train_sample) - y_train_sample.sum()) / y_train_sample.sum()),
        'random_state': 42
    },
    'preprocessing': {
        'scaler_type': 'RobustScaler',
        'scaler_name': 'robust',
        'scaled_columns': ['Amount'],
        'passthrough_columns': [c for c in X_train_sample.columns if c != 'Amount']
    },
     'pipeline_steps': [
        {'step': 'preprocessor', 'type': 'ColumnTransformer'},
        {'step': 'model', 'type': results_df.loc[results_df['model'] == 'XGBoost','model'].values[0]}
    ],
    'prototype_performance': {
        'cv_roc_auc': float(results_df.loc[results_df['model'] == 'XGBoost','roc_auc'].values[0]),
        'cv_precision': float(results_df.loc[results_df['model'] == 'XGBoost','precision'].values[0]),
        'cv_recall': float(results_df.loc[results_df['model'] == 'XGBoost','recall'].values[0]),
        'cv_f1': float(results_df.loc[results_df['model'] == 'XGBoost','f1'].values[0]),
        'cv_folds': cv.n_splits,
        'n_samples': len(X_train_sample),
        'fraud_ratio': float(y_train_sample.sum() / len(y_train_sample))
    },
    'feature_columns': list(X_train_sample.columns),
    'target_column': 'Class',
    
    'created_date': datetime.now().isoformat(),
    'created_by': 'prototype_pipeline',
    'sklearn_version': sklearn.__version__,
    'python_version': sys.version.split()[0]
}

# COMMAND ----------

print(json.dumps(config, indent=4))

# COMMAND ----------

config_path = "../configs/prototype_config.json"  
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print("✅ Prototype config saved to configs/prototype_config.json")