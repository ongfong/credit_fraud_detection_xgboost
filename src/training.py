import json
import os
from datetime import datetime
from pyspark.sql.functions import sum, when, col

try:
    from xgboost.spark import SparkXGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  Warning: xgboost.spark not available")

from .config_adapter import ConfigAdapter

def evaluate_model(predictions, label_col="Class"):

    print("\n📊 Evaluating model (Optimized)...")
    # 1. Calculate confusion matrix components
    
    metrics_row = predictions.agg(
        sum(when((col(label_col) == 1) & (col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        sum(when((col(label_col) == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        sum(when((col(label_col) == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        sum(when((col(label_col) == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("fn"),
        sum(when(col(label_col) == col("prediction"), 1).otherwise(0)).alias("correct")
    ).collect()[0]

    tp = metrics_row['tp']
    fp = metrics_row['fp']
    tn = metrics_row['tn']
    fn = metrics_row['fn']
    correct = metrics_row['correct']
    
    total = tp + fp + tn + fn
    
    # 2. Check for edge cases
    if total == 0:
        print("⚠️  No predictions to evaluate")
        return {
            "accuracy": None,
            "confusion_matrix": {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        }
    
    # 3. Calculate metrics
    accuracy = float(correct) / float(total)
    
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn)
        }
    }
    
    print(f"\n   Results (on {total:,} predictions):")
    print(f"   ├─ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   ├─ Precision: {metrics['precision']:.4f}")
    print(f"   ├─ Recall:    {metrics['recall']:.4f}")
    print(f"   └─ F1 Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n   Confusion Matrix:")
    print(f"   ├─ TP: {tp:>6,}  FP: {fp:>6,}")
    print(f"   └─ FN: {fn:>6,}  TN: {tn:>6,}")

    return metrics

def run_training(config_path, train_df, test_df, model_path, spark):

    print("\n" + "="*70)
    print("🤖 MODEL TRAINING: XGBoost on Gold Data")
    print("="*70)
    
    # Check XGBoost availability
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost.spark not available. Install with: pip install xgboost[spark]==2.0.3"
        )
    
    # ========================================
    # 1. Load Config
    # ========================================
    print("\n📄 Loading config...")
    config_adapter = ConfigAdapter(config_path)
    xgb_params = config_adapter.get_xgboost_params()
    target_col = config_adapter.get_target_column()
    
    train_df.cache() 
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()

    fraud_count = train_df.filter(f"{target_col} = 1").count()
    normal_count = train_df.filter(f"{target_col} = 0").count()

    total = train_count + test_count

    print(f"Total: {total:,}")
    print(f"normal_count: {normal_count:,}")
    print(f"fraud_count: {fraud_count:,}")
    print(f"Train dataset: {train_count:,}")
    print(f"Test dataset:  {test_count:,}")

    # ========================================
    # 4. Create XGBoost Classifier
    # ========================================
    print(f"\n🔧 Creating XGBoost classifier...")
    
    # Extract num_round
    num_round = xgb_params.pop("num_round", 100)
    
    # Remove unsupported params
    unsupported = ["objective", "eval_metric", "tree_method"]
    for key in unsupported:
        xgb_params.pop(key, None)
    
    try:
        xgb_classifier = SparkXGBClassifier(
            features_col="features",
            label_col=target_col,
            prediction_col="prediction",
            num_workers=1,
            use_gpu=False,
            num_boost_round=num_round,
            **xgb_params,
            # scale_pos_weight=200, #tuning for imbalanced data
            random_state=42
        )
    except Exception as e:
        raise ValueError(f"Failed to create XGBoost classifier: {e}")
    
    print(f"   num_boost_round: {num_round}")
    print(f"   num_workers: 1 (local mode)")
    
    # ========================================
    # 5. Train Model
    # ========================================
    print(f"\n⏳ Training XGBoost model...")
    print(f"   (This may take a while with {train_count:,} records...)")
    
    try:
        trained_model = xgb_classifier.fit(train_df)
        print("✅ Training completed")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ========================================
    # 6. Evaluate
    # ========================================
    print(f"\n📊 Evaluating on test set...")
    
    try:
        predictions = trained_model.transform(test_df)
        metrics = evaluate_model(predictions, target_col)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
    # ========================================
    # 7. Save Model
    # ========================================
    print(f"\n💾 Saving model to: {model_path}")
    os.makedirs(model_path, exist_ok=True)
    
    # Save as JSON (safer than SparkML format)
    try:
        model_json_path = os.path.join(model_path, "xgb_model.json")
        trained_model.get_booster().save_model(model_json_path)
        print(f"✅ Model saved (JSON): {model_json_path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save model as JSON: {e}")
    
    # Save metrics
    metrics_data = {
        "model_info": {
            "type": "XGBoost",
            "framework": "xgboost.spark.SparkXGBClassifier",
            "trained_date": datetime.now().isoformat()
        },
        "config": config_adapter.get_all(),
        "xgboost_params": {
            **xgb_params,
            "num_boost_round": num_round,
        },
        "metrics": metrics
    }
    
    metrics_path = os.path.join(model_path, "metrics.json")  
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Metrics saved: {metrics_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    
    return trained_model, metrics