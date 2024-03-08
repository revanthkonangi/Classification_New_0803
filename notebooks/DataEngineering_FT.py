# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# %pip install /dbfs/FileStore/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall
# %pip install databricks-feature-store 

%pip install sparkmeasure

dbutils.library.restartPython() 

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# DE SPECIFIC PARAMETERS
primary_keys = solution_config["data_engineering"]["data_engineering_ft"]["primary_keys"]
features_table_name = solution_config["data_engineering"]["data_engineering_ft"]["features_table_name"]
features_dbfs_path = solution_config["data_engineering"]["data_engineering_ft"]["features_dbfs_path"]

# COMMAND ----------

from MLCORE_SDK import mlclient
from utils import utils

mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = "DE")

# COMMAND ----------

features_df = spark.read.load(features_dbfs_path)
features_df = features_df.drop('date','id','timestamp')

# COMMAND ----------

features_df.display()

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
features_df = features_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
features_df = features_df.withColumn("date", F.lit(date))
features_df = features_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in features_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  features_df = features_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

features_table_name = f"{features_table_name}_{env}_{sdk_session_id}"

utils.df_write(
    data_path=features_table_name,
    dataframe = features_df,
    mode = "overwrite",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = features_table_name,
    num_rows = features_df.count(),
    cols = features_df.columns,
    column_datatype = features_df.dtypes,
    table_schema = features_df.schema,
    primary_keys = primary_keys,
    table_path = f"{gcp_project_id}.{bq_database_name}.{features_table_name}",
    table_type="bigquery",
    table_sub_type="Source",
    env = "dev",
    verbose = True,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------


