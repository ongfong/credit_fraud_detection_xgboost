from pyspark.sql import SparkSession
from src import ingestion, create_silver, feature_engineering, training
from delta import configure_spark_with_delta_pip
import sys

def get_local_spark():
    
    builder = (
        SparkSession.builder
        .appName("creditcard-xgboost-pipeline")
        .master("local[*]")
        .config("spark.jars", "/app/jars/*")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Memory tuning
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.maxResultSize", "1g")
        # Performance tuning
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
    )
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    print("="*70)
    print("✅ Spark Session Created")
    print(f"   Spark UI: {spark.sparkContext.uiWebUrl}")
    print("="*70 + "\n")
    
    return spark

def main():
    
    spark = None
    
    # Paths
    config_path = "./configs/prototype_config.json"
    raw_csv = "data/raw/creditcard_raw.csv"
    bronze_table = "data/bronze/bronze_creditcard"
    silver_table = "data/silver/silver_creditcard"
    gold_table = "data/gold/gold_creditcard"
    pipeline_path = "models/pipeline/feature_pipeline"
    model_path = "models/production/xgboost_model"
    
    try:
        # ========================================
        # Initialize Spark
        # ========================================
        spark = get_local_spark()
        
        print("="*70)
        print("🚀 PRODUCTION ML PIPELINE - XGBoost")
        print("="*70 + "\n")
        
        # ========================================
        # STAGE 1: Ingestion
        # ========================================
        print("STAGE 1: INGESTION")
        print("-"*70)
        ingestion.run_ingestion(raw_csv, bronze_table, spark)
        print("✅ Stage 1 complete\n")
        
        # ========================================
        # STAGE 2: Silver
        # ========================================
        print("STAGE 2: CREATE SILVER")
        print("-"*70)
        create_silver.create_silver(bronze_table, silver_table, spark)
        print("✅ Stage 2 complete\n")
        
        # ========================================
        # STAGE 3: Feature Engineering
        # ========================================
        print("STAGE 3: FEATURE ENGINEERING")
        print("-"*70)
        train_gold, test_gold, pipeline_model = feature_engineering.run_feature_engineer(
            config_path, silver_table, pipeline_path, spark
        )
        print("✅ Stage 3 complete\n")
        
        # ========================================
        # STAGE 4: Training
        # ========================================
        print("STAGE 4: MODEL TRAINING")
        print("-"*70)
        trained_model, metrics = training.run_training(
            config_path, train_gold, test_gold, model_path, spark
        )
        print("✅ Stage 4 complete\n")
        
        # ========================================
        # Summary
        # ========================================
        print("="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\n📁 Outputs:")
        print(f"   Bronze:   {bronze_table}")
        print(f"   Silver:   {silver_table}")
        print(f"   Gold:     {gold_table}")
        print(f"   Pipeline: {pipeline_path}")
        print(f"   Model:    {model_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ PIPELINE FAILED")
        print(f"{'='*70}")
        print(f"\nError: {str(e)}")
        
        import traceback
        print(f"\nFull traceback:")
        print("-"*70)
        traceback.print_exc()
        print("-"*70)
        
        sys.exit(1)
    
    finally:
        # Safe Spark shutdown
        if spark is not None:
            try:
                print("\n🔌 Stopping Spark session...")
                spark.stop()
                print("✅ Spark session stopped successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error stopping Spark: {e}")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    main()