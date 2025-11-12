from pyspark.sql.functions import col, current_timestamp
from delta.tables import DeltaTable

def create_silver(bronze_table_name, silver_table_name, spark):

    if DeltaTable.isDeltaTable(spark, bronze_table_name):
        bronze_df = spark.read.format("delta").load(bronze_table_name)
    else:
        bronze_df = spark.table(bronze_table_name) #

    bronze_df = bronze_df.withColumn("create_silver_timestamp", current_timestamp())
    
    bronze_count = bronze_df.count()
    print(f"📦 Bronze Records ({bronze_table_name}): {bronze_count:,}")

    duplicates_removed = bronze_count - bronze_df.dropDuplicates().count()
    invalid_amount = bronze_df.filter(col('Amount') < 0).count()
    invalid_class = bronze_df.filter(~col('Class').isin([0, 1])).count()

    silver_batch_df = (
        bronze_df
        .filter(col('Amount')  >= 0)
        .filter(col('Class').isin([0,1]))
        .filter(col('Amount').isNotNull())
        .filter(col('Class').isNotNull())
        .dropDuplicates()
    )

    silver_count = silver_batch_df.count()
    total_removed = bronze_count - silver_count

    mode = "append" if DeltaTable.isDeltaTable(spark, silver_table_name) else "overwrite"
    silver_batch_df.write.format("delta").mode(mode).save(silver_table_name)

    print(f"\n✅ Silver layer updated: {silver_table_name}")
    print(f"   Records in this batch: {silver_count:,}")
    print(f"📊 Data Quality Summary:")
    print(f"   Duplicates removed: {duplicates_removed:,}")
    print(f"   Invalid Amount: {invalid_amount}")
    print(f"   Invalid Class: {invalid_class}")
    print(f"   Total removed: {total_removed:,} ({total_removed/bronze_count*100:.2f}%)")

    return silver_batch_df
