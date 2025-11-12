from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

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