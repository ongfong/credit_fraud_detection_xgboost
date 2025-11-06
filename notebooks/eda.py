# Databricks notebook source
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# COMMAND ----------

df_spark = spark.table("workspace.bronze.bronze_creditcard")

# COMMAND ----------

df = df_spark.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

df.describe().T

# COMMAND ----------

print(df['Class'].value_counts(normalize=True) * 100)

# COMMAND ----------

duplicates = df.duplicated().sum()
print(duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC Class Distribution

# COMMAND ----------

class_counts = df['Class'].value_counts()
clss_pct = df['Class'].value_counts(normalize=True) * 100
class_ratio = class_counts[0] / class_counts[1]
print(f"Normal Transactions: {class_counts[0]:,} ({clss_pct[0]:.2f}%)")
print(f"Fraudulent Transactions: {class_counts[1]:,} ({clss_pct[1]:.2f}%)")
print(f"Class Ratio: {class_ratio:.2f} : 1")

# COMMAND ----------

fig = px.histogram(df,x="Class",color="Class",title="Class Distribution",text_auto=True)
fig.update_layout(bargap=0.4)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Importance & Correlation

# COMMAND ----------

df_corr = df.corr()['Class'].abs().sort_values(ascending=False)
top_features = df_corr.index[:7]

# COMMAND ----------

fig = px.imshow(df[top_features].corr(),color_continuous_scale='RdBu_r',text_auto=".3f",aspect="auto")
fig.update_layout(title="Correlation Matrix of Top 6 features ",width=800,height=800)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Relationships Analysis

# COMMAND ----------

sns.pairplot(df[top_features], diag_kind='kde',hue='Class',plot_kws={'alpha':0.5,'s':20})
plt.title("Pairplot of Top 6 features")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Transaction Amount Analysis

# COMMAND ----------

amounts_stat = df.groupby('Class')['Amount'].describe().T
print(amounts_stat)

# COMMAND ----------

fig = px.box(df,x='Class',y='Amount',color='Class',title="Boxplot of Amount by Class")
fig.update_layout(width=800,height=500)

# COMMAND ----------

fig, ax = plt.subplots(1,2, figsize=(12,5))

df[df['Class'] == 0]['Amount'].hist(bins=50, ax=ax[0], color='green', alpha=0.5, edgecolor = 'black')
ax[0].set_title('Normal Transactions')
ax[0].grid(alpha=0.3)

df[df['Class'] == 1]['Amount'].hist(bins=50, ax=ax[1], color='red', alpha=0.5, edgecolor = 'black')
ax[1].set_title('Fraudulent Transactions')
ax[1].grid(alpha=0.3)

plt.show()

# COMMAND ----------

print(f"Normal Range $0 - ${df[df['Class'] == 0]['Amount'].max():,.2f}")
print(f"Fraudulent Range $0 - ${df[df['Class'] == 1]['Amount'].max():,.2f}")

# COMMAND ----------

amount_bins = [0, 50, 100, 200 ,500 ,1000, df['Amount'].max()]
df['Amount_range'] = pd.cut(df['Amount'], bins = amount_bins)
fraud_by_range = df.groupby('Amount_range')['Class'].agg(['count','sum'])
fraud_by_range = fraud_by_range.rename(columns={'count':'Total','sum':'Fraud'})
fraud_by_range['fraud_rate'] = fraud_by_range['Fraud'] / fraud_by_range['Total'] * 100
print(fraud_by_range)

# COMMAND ----------

# MAGIC %md
# MAGIC Time Analysis

# COMMAND ----------

print(f"Time span: {df['Time'].max():.2f} - {df['Time'].min():.2f} seconds")
print(f"Approzimately: {df['Time'].max() / 60 / 60 / 24:.2f} days")

# COMMAND ----------

fig = px.box(df, x='Class', y='Time', color='Class')
fig.show()

# COMMAND ----------

time_stats = df.groupby('Class')['Time'].describe()[['mean','50%','std']]
print(time_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC Outlier

# COMMAND ----------

q1, q3 = df['Amount'].quantile([0.25,0.75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]
print(f"Outliers: {len(outliers) / len(df):.2%}")
print(f"Fraud in outliers: {outliers['Class'].mean()*100:.2f}%")