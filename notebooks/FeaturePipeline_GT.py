# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# %pip install /dbfs/FileStore/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall
# %pip install databricks-feature-store 
%pip install sparkmeasure

# dbutils.library.restartPython() 

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
from utils import utils
from MLCORE_SDK import mlclient

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------


# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

if task.lower() == "fe":
    # JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
    ground_truth_dbfs_path = solution_config["feature_pipelines"]["feature_pipelines_gt"]["ground_truth_dbfs_path"]
    transformed_ground_truth_table_name = solution_config["feature_pipelines"]["feature_pipelines_gt"]["transformed_ground_truth_table_name"]
    is_scheduled = solution_config["feature_pipelines"]["feature_pipelines_gt"]["is_scheduled"]
    batch_size = int(solution_config["feature_pipelines"]["feature_pipelines_gt"].get("batch_size",500))
    cron_job_schedule = solution_config["feature_pipelines"]["feature_pipelines_gt"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["feature_pipelines"]["feature_pipelines_gt"]["primary_keys"]
else: 
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    ground_truth_dbfs_path = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["ground_truth_dbfs_path"]
    transformed_ground_truth_table_name = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["transformed_ground_truth_table_name"]
    is_scheduled = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["is_scheduled"]
    batch_size = int(solution_config["data_prep_deployments"]["data_prep_deployment_gt"].get("batch_size",500))
    cron_job_schedule = solution_config["data_prep_deployments"]["data_prep_deployment_gt"].get("cron_job_schedule","0 */10 * ? * *")
    primary_keys = solution_config["data_prep_deployments"]["data_prep_deployment_gt"]["primary_keys"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Ground Truth Data

# COMMAND ----------

# DBTITLE 1,Load the Data
_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

ground_truth_df = utils.df_read(
            spark = spark,
            data_path=f"{ground_truth_dbfs_path}_{env}_{sdk_session_id}",
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

# COMMAND ----------

ground_truth_df.display()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

ground_truth_df = ground_truth_df.toPandas()

# COMMAND ----------

ground_truth_df.info()

# COMMAND ----------

categ = ['Churn']

from sklearn.preprocessing import LabelEncoder
# Encode Categorical Columns
le = LabelEncoder()
ground_truth_df[categ] = ground_truth_df[categ].apply(le.fit_transform)

# COMMAND ----------

from pyspark.sql import functions as F
import pickle

# COMMAND ----------

if is_scheduled:
  # CREATE PICKLE FILE
  pickle_file_path = f"/mnt/FileStore/{db_name}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{transformed_ground_truth_table_name}.pickle"

  # LOAD CACHE IF AVAILABLE
  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1

  print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")
else :
  start_marker = 1
  end_marker = ground_truth_df.count()


# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step.
GT_DF = ground_truth_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

if not GT_DF.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

if task.lower() != "fe":
    mlclient.log(operation_type="job_run_add", session_id = sdk_session_id, dbutils = dbutils, request_type = task)

# COMMAND ----------

GT_DF = GT_DF.drop('date','timestamp')
GT_DF.display()

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
GT_DF = GT_DF.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
GT_DF = GT_DF.withColumn("date", F.lit(date))
GT_DF = GT_DF.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in GT_DF.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  GT_DF = GT_DF.withColumn("id", F.row_number().over(window))

# COMMAND ----------

transformed_ground_truth_table_name = f"{transformed_ground_truth_table_name}_{env}_{sdk_session_id}"

utils.df_write(
    data_path=transformed_ground_truth_table_name,
    dataframe = GT_DF,
    mode = "append",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

print(f"Big Query path : {gcp_project_id}.{bq_database_name}.{transformed_ground_truth_table_name}",)

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

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

# DBTITLE 1,Register Ground Truth Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = transformed_ground_truth_table_name,
    num_rows = GT_DF.count(),
    cols = GT_DF.columns,
    column_datatype = GT_DF.dtypes,
    table_schema = GT_DF.schema,
    primary_keys = primary_keys,
    table_path = f"{gcp_project_id}.{bq_database_name}.{transformed_ground_truth_table_name}",
    table_type="bigquery",
    table_sub_type="Ground_Truth",
    request_type = task,
    env = env,
    batch_size = batch_size,
    quartz_cron_expression = cron_job_schedule,
    verbose = True,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")

# COMMAND ----------

# try :
#     #define media artifacts path
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Tiger_ML_Python_EDA", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#             "input_table_path" : gt_hive_table_path,
#             "media_artifacts_path" : media_artifacts_path,
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")
