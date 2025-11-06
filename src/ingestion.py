from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

def run_ingestion(raw_table_name, bronze_table_name, spark):
    
    if raw_table_name.endswith(".csv"):
        bronze_df = spark.read.option("header", True).option("inferSchema", True).csv(raw_table_name)
    else:
        bronze_df = spark.table(raw_table_name)

    bronze_df = bronze_df.withColumn("ingestion_timestamp", current_timestamp())

    mode = "append" if DeltaTable.isDeltaTable(spark, bronze_table_name) else "overwrite"

    bronze_df.write.format("delta").mode(mode).save(bronze_table_name)

    bronze_count = bronze_df.count()
    print(f"✅ Bronze table updated: {bronze_table_name}")
    print(f"   Records in this batch: {bronze_count:,}")

    return bronze_df
