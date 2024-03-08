# Databricks notebook source
# MAGIC %md ## EDA Python
# MAGIC

# COMMAND ----------

# MAGIC %md <b>Imports
# MAGIC
# MAGIC Along with the imports required for the notebook to execute custom transformations, we have to import <b>MLCoreClient</b> from <b>MLCORE_SDK</b>, which provides helper methods to integrate the custom notebook with rest of the Data Prep or Data Prep Deployment flow.

# COMMAND ----------

# DBTITLE 1,Install MLCore SDK
# %pip install /dbfs/FileStore/sdk/Vamsi/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall
# %pip install databricks-feature-store

# COMMAND ----------

# MAGIC %pip install /dbfs/FileStore/mnt/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %pip install -U seaborn
# MAGIC %pip install pandas==1.0.5
# MAGIC %pip install numpy==1.20

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
import numpy
import seaborn as sns
from pyspark.sql import functions as F, types as T
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving images

# COMMAND ----------

# DBTITLE 1,Initialize MLCore Client
from MLCORE_SDK import mlclient

# COMMAND ----------

input_table_path = dbutils.widgets.get("input_table_path")

# COMMAND ----------

df = spark.read.load(input_table_path)

# COMMAND ----------

df.display()

# COMMAND ----------

# Converting PySpark data into pandas
pd_df = df.toPandas()
pd_df.head()

# COMMAND ----------

pd_df.drop(columns = ["id","date","timestamp"],inplace = True)

# COMMAND ----------

for column in pd_df.columns:
    plt.figure(figsize=(8, 6))  # Create a new figure for each column
    sns.histplot(pd_df[column], bins=10, kde=True)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'{column} Distribution')

    # mlclient.log(
    #     operation_type="register_plots",
    #     dbutils=dbutils,
    #     figure_to_save=plt,
    #     plot_name=f'{column}_Distribution',
    #     folder_name="custom_reports",
    #     request_type="push_plot"
    # )
    utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=plt,
						   plot_name=f'{column}_Distribution',
						   folder_name='custom_reports',
						   )


# COMMAND ----------

# Get numeric columns from the DataFrame
numeric_columns = pd_df.select_dtypes(include=['number']).columns
bi_fig = plt.figure(figsize=(8,6))
# Create a pairplot for all pairs of numeric columns
sns.set(style="ticks")
sns.pairplot(pd_df, vars=numeric_columns, diag_kind='kde')

# Show the plot
plt.show()

# mlclient.log(operation_type="register_plots",
#     dbutils = dbutils, 
#     figure_to_save=plt, 
#     plot_name="Bivariate_Plots",
#     folder_name="custom_reports", 
#     request_type="push_plot")
utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=tigermleda,
						   plot_name="Bivariate_Plots",
						   folder_name='custom_reports',
						   )

# COMMAND ----------

# Calculate the correlation matrix
corr_matrix = pd_df.corr()

# Create a heatmap
corr_fig = plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Using helper method push_plots_to_mlcore to save the plot
# mlclient.log(operation_type = "register_plots",
#     dbutils = dbutils, 
#     figure_to_save=corr_fig, 
#     plot_name="Correlation_Plot",
#     folder_name = "custom_reports", 
#     request_type="push_plot")
utils.push_plots_to_mlcore(dbutils=dbutils,
						   figure_to_save=corr_fig,
						   plot_name="Correlation_Plot",
						   folder_name='custom_reports',
						   )
