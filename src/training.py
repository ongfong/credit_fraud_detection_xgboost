import json
import os
from datetime import datetime

try:
    from xgboost.spark import SparkXGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  Warning: xgboost.spark not available")

from .config_adapter import ConfigAdapter

def evaluate_model(predictions, label_col="Class"):

    print("\n📊 Evaluating model...")
    
    try:
        # Count predictions
        total = predictions.count()
        
        if total == 0:
            print("⚠️  No predictions to evaluate")
            return {
                "accuracy": None,
                "confusion_matrix": {
                    "true_positive": 0,
                    "false_positive": 0,
                    "true_negative": 0,
                    "false_negative": 0
                }
            }
        
        # Accuracy
        correct = predictions.filter(f"{label_col} = prediction").count()
        accuracy = float(correct) / float(total)
        
        # Confusion matrix
        tp = predictions.filter(f"{label_col} = 1 AND prediction = 1").count()
        fp = predictions.filter(f"{label_col} = 0 AND prediction = 1").count()
        tn = predictions.filter(f"{label_col} = 0 AND prediction = 0").count()
        fn = predictions.filter(f"{label_col} = 1 AND prediction = 0").count()
        
        # Calculate additional metrics
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
        print(f"   ├─ TP: {tp:>6}  FP: {fp:>6}")
        print(f"   └─ FN: {fn:>6}  TN: {tn:>6}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "confusion_matrix": {
                "true_positive": 0,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0
            }
        }

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
    
    train_count = train_df.count()
    test_count = test_df.count()

    fraud_count = train_df.filter(f"{target_col} = 1").count()
    normal_count = train_df.filter(f"{target_col} = 0").count()

    total = train_count + test_count
    scale_pos_weight = normal_count / fraud_count

    print(f"Total: {total:,}")
    print(f"normal_count: {normal_count:,}")
    print(f"fraud_count: {fraud_count:,}")
    print(f"Train dataset: {train_count:,}")
    print(f"Test dataset:  {test_count:,}")
    print(f"Scale_pos_weight: {scale_pos_weight:,}")

    # ========================================
    # 4. Create XGBoost Classifier
    # ========================================
    print(f"\n🔧 Creating XGBoost classifier...")
    
    # Extract num_round
    num_round = xgb_params.pop("num_round", 100)
    
    # Remove unsupported params
    unsupported = ["objective", "eval_metric", "tree_method","scale_pos_weight"]
    for key in unsupported:
        xgb_params.pop(key, None)
    
    try:
        xgb_classifier = SparkXGBClassifier(
            features_col="features",
            label_col=target_col,
            prediction_col="prediction",
            num_workers=1,
            use_gpu=False,
            scale_pos_weight=scale_pos_weight,
            num_boost_round=num_round,
            **xgb_params
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
            "num_boost_round": num_round
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