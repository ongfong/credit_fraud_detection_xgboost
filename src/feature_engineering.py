import os
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml import Pipeline
from delta.tables import DeltaTable
from .config_adapter import ConfigAdapter 


def run_feature_engineer(config_path, silver_table_name, pipeline_save_path, spark):
 
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING: Silver → Gold")
    print("="*70)
    
    # ========================================
    # 1. Load Config
    # ========================================
    print("\n📄 Loading config...")
    config_adapter = ConfigAdapter(config_path) 

    config = config_adapter.get_all()

    preprocessing = config.get("preprocessing", {})
    scaled_columns = preprocessing.get("scaled_columns", [])
    passthrough_columns = preprocessing.get("passthrough_columns", [])
    target_col = config_adapter.get_target_column()

    # ========================================
    # 2. Load Silver Data
    # ========================================
    print(f"\n📂 Loading Silver data from: {silver_table_name}")
    
    if DeltaTable.isDeltaTable(spark, silver_table_name):
        silver_df = spark.read.format("delta").load(silver_table_name)
    else:
        silver_df = spark.table(silver_table_name)
    
    record_count = silver_df.count()
    print(f"   Records: {record_count:,}")

    print(f"\n✂️  Splitting data (80/20) BEFORE transformation...")
    train_raw, test_raw = silver_df.randomSplit([0.8, 0.2], seed=42)

    print(f"Train total: {train_raw.count():,}")
    print(f"Test total: {test_raw.count():,}")
    print(train_raw.groupBy("Class").count().show())
    print(test_raw.groupBy("Class").count().show())

    # ========================================
    # 3. Build Pipeline
    # ========================================
    print("\n⚙️  Building preprocessing pipeline...")
    
    # Stage 1: VectorAssembler for Amount
    amount_assembler = VectorAssembler(
        inputCols=scaled_columns,
        outputCol="amount_vector",
        handleInvalid="skip"
    )
    
    # Stage 2: RobustScaler
    scaler = RobustScaler(
        inputCol="amount_vector",
        outputCol="amount_scaled",
        withCentering=True,
        withScaling=True
    )
    
    # Stage 3: Final VectorAssembler
    final_assembler = VectorAssembler(
        inputCols=passthrough_columns + ["amount_scaled"],
        outputCol="features",
        handleInvalid="skip"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[amount_assembler, scaler, final_assembler])
    
    print("   Pipeline stages:")
    print(f"   1. VectorAssembler: {scaled_columns} → amount_vector")
    print(f"   2. RobustScaler: amount_vector → amount_scaled")
    print(f"   3. VectorAssembler: {len(passthrough_columns)} + 1 → features")
    
    # ========================================
    # 4. Fit Pipeline
    # ========================================
    print("\n🔧 Fitting pipeline on Silver data...")
    pipeline_model = pipeline.fit(train_raw)
    print("✅ Pipeline fitted successfully")
    
    # ========================================
    # 5. Transform Data
    # ========================================
    print("\n⚡ Transforming data...")
    train_transformed  = pipeline_model.transform(train_raw)
    test_transformed  = pipeline_model.transform(test_raw)
    
    train_gold = train_transformed.select(
        target_col,
        *passthrough_columns,
        *scaled_columns,
        "features"
    )

    test_gold = test_transformed.select(
        target_col,
        *passthrough_columns,
        *scaled_columns,
        "features"
    )
    
    print(f"✅ Transformation complete:")
    print(f"   Train: {train_gold.count():,} records")
    print(f"   Test: {test_gold.count():,} records")
    
    # Show sample
    # print("\n📊 Sample transformed data (first 3 rows):")
    # train_gold.select(target_col, "Amount", "features").show(3, truncate=False)
    print("\n📊 Sample transformed data (first 3 rows):")
    df_sample = train_gold.select(target_col, "Amount", "features").limit(3).toPandas()
    print(df_sample)
    
    print(f"\n💾 Saving fitted pipeline to: {pipeline_save_path}")
    os.makedirs(os.path.dirname(pipeline_save_path), exist_ok=True)
    pipeline_model.write().overwrite().save(pipeline_save_path)
    print("✅ Pipeline model saved successfully")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("="*70)

    return train_gold, test_gold, pipeline_model
