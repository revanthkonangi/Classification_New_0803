# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# %pip install /dbfs/FileStore/sdk/Vamsi/MLCoreSDK-0.5.96-py3-none-any.whl --force-reinstall
# %pip install databricks-feature-store
%pip install sparkmeasure

dbutils.library.restartPython() 

# COMMAND ----------

from sparkmeasure import StageMetrics
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
from utils import utils

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import *

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
# GENERAL PARAMETERS
sdk_session_id = solution_config[f'sdk_session_id_{env}']
env = solution_config['ds_environment']
db_name = solution_config['database_name']

# JOB SPECIFIC PARAMETERS
feature_table_path = solution_config['train']["feature_table_path"]
ground_truth_path = solution_config['train']["ground_truth_path"]
primary_keys = solution_config['train']["primary_keys"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
train_output_table_name = solution_config['train']["train_output_table_name"]
test_size = solution_config['train']["test_size"]
model_name = solution_config['train']["model_name"]
model_version = solution_config['train']["model_version"]
sensitive_columns = solution_config['train']["sensitive_variables"]

# COMMAND ----------

# DBTITLE 1,Update the table paths as needed.
_,vault_scope = utils.get_env_vault_scope(dbutils)

encrypted_sa_details = dbutils.secrets.get(vault_scope,"gcp-service-account-encypted")
encryption_key = dbutils.secrets.get(vault_scope,"gcp-service-account-private-key")
bq_database_name = dbutils.secrets.get(vault_scope,"gcp-bq-database-name")
gcp_project_id = dbutils.secrets.get(vault_scope,"gcp-api-quota-project-id")
az_container_name = str(dbutils.secrets.get(scope=vault_scope, key='az-container-name'))

feature_table_path = f"{feature_table_path}_{env}_{sdk_session_id}"
ground_truth_path = f"{ground_truth_path}_{env}_{sdk_session_id}"

ft_data = utils.df_read(
            spark = spark,
            data_path=feature_table_path,
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

gt_data = utils.df_read(
            spark = spark,
            data_path=ground_truth_path,
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type="bigquery")

# COMMAND ----------

features_data = ft_data.select(primary_keys + feature_columns)
gt_data = gt_data.select(primary_keys + target_columns)

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(gt_data, on = primary_keys)
final_df = final_df.drop('date','id','timestamp')

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
final_df_pandas.head()

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
# Split the Data to Train and Test
X_train, X_test, y_train, y_test = train_test_split(final_df_pandas[feature_columns], final_df_pandas[target_columns], test_size=test_size, random_state = 0)

# COMMAND ----------

# DBTITLE 1,Defining the Model Pipeline
# Build a Scikit learn pipeline
pipe = Pipeline([
    ('classifier',LogisticRegression())
])
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# COMMAND ----------

# DBTITLE 1,Fitting the pipeline on Train data 
# Fit the pipeline
lr = pipe.fit(X_train_np, y_train)

# COMMAND ----------

# DBTITLE 1,Calculating the test metrics from the model
# Predict it on Test and calculate metrics
y_pred = lr.predict(X_test_np)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# COMMAND ----------

# DBTITLE 1,Displaying the test metrics 
test_metrics = {"accuracy":accuracy, "f1":f1, "precision":precision, "recall":recall}
test_metrics

# COMMAND ----------

# Predict it on Train and calculate metrics
y_pred_train = lr.predict(X_train_np)
accuracy = accuracy_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)

# COMMAND ----------

train_metrics = {"accuracy":accuracy, "f1":f1, "precision":precision, "recall":recall}
train_metrics

# COMMAND ----------

# DBTITLE 1,Join the X and y to single df
pred_train = pd.concat([X_train, y_train], axis = 1)
pred_test = pd.concat([X_test, y_test], axis = 1)

# COMMAND ----------

# DBTITLE 1,Getting train and test predictions from the model
# Get prediction columns
y_pred_train = lr.predict(X_train_np)
y_pred = lr.predict(X_test_np)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

  pred_train["prediction"] = y_pred_train
  pred_train["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"
  pred_test["prediction"] = y_pred
  pred_test["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "test"
  pred_test["probability"] = lr.predict_proba(pred_test[feature_columns]).tolist()
  pred_train["probability"] = lr.predict_proba(pred_train[feature_columns]).tolist()

# COMMAND ----------

final_train_output_df = pd.concat([pred_train, pred_test])

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

train_output_df = spark.createDataFrame(final_train_output_df)
now = datetime.now()
date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
train_output_df = train_output_df.withColumn("date", F.lit(date))
train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))
w = Window.orderBy(F.monotonically_increasing_id())

train_output_df = train_output_df.withColumn("id", F.row_number().over(w))

# COMMAND ----------

train_output_table_name = f"{train_output_table_name}_{env}_{sdk_session_id}"

utils.df_write(
    data_path=train_output_table_name,
    dataframe = train_output_df,
    mode = "overwrite",
    bucket_name=f"{az_container_name}_{env}",
    bq_database_name=bq_database_name,
    bq_project_id=gcp_project_id,
    encrypted_service_account=encrypted_sa_details,
    encryption_key=encryption_key,
    resource_type="bigquery")

print(f"Big Query path : {gcp_project_id}.{bq_database_name}.{train_output_table_name}",)

# COMMAND ----------

stagemetrics.end()

# COMMAND ----------

stagemetrics.print_report()

# COMMAND ----------

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory").collect()[0].asDict()

# COMMAND ----------

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## REGISTER MODEL IN MLCORE

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

# DBTITLE 1,Registering the model in MLCore
mlclient.log(operation_type = "register_model",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    model = pipe,
    model_name = f"{model_name}",
    model_runtime_env = "python",
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    feature_table_path = f"{gcp_project_id}.{bq_database_name}.{feature_table_path}",
    ground_truth_table_path = f"{gcp_project_id}.{bq_database_name}.{ground_truth_path}",
    train_output_path = f"{gcp_project_id}.{bq_database_name}.{train_output_table_name}",
    train_output_rows = train_output_df.count(),
    train_output_cols = train_output_df.columns,
    feature_columns = feature_columns,
    target_columns = target_columns,
    column_datatype = train_output_df.dtypes,
    table_schema = train_output_df.schema,
    table_type = "bigquery",
    verbose = True,
    hp_tuning_result=None,
    compute_usage_metrics = compute_metrics)

# COMMAND ----------

# try :
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
#     dbutils.notebook.run(
#         "Model_Test", 
#         timeout_seconds = 5000, 
#         arguments = 
#         {
#             "feature_columns" : ",".join(feature_columns),
#             "target_columns" : ",".join(target_columns), #json dumps
#             "model_data_path":train_output_dbfs_path,
#             "model_name": model_name,
#             "media_artifacts_path" : media_artifacts_path
#             })
# except Exception as e:
#     print(f"Exception while triggering model testing notebook : {e}")

# COMMAND ----------

# from utils import utils

# COMMAND ----------

# try: 
#     #define media artifacts path
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
    
#     print(media_artifacts_path)

#     custom_notebook_result = dbutils.notebook.run(
#         "Model_eval",
#         0,
#         arguments = 
#         {
#         "feature_columns" : ",".join(feature_columns),
#         "target_columns" : ",".join(target_columns), #json dumps
#         "model_data_path":train_output_dbfs_path,
#         "model_name": model_name,
#         "media_artifacts_path" : media_artifacts_path,
#         },
#     )

# except Exception as e:
#     print(str(e))

# COMMAND ----------

# try :
#     #define media artifacts path
#     media_artifacts_path = utils.get_media_artifact_path(sdk_session_id = sdk_session_id,
#         dbutils = dbutils,
#         env = env)
    
#     print(media_artifacts_path)
#     custom_notebook_result = dbutils.notebook.run(
#         "Bias_Fairness",
#         0,
#         arguments = 
#         {
#         "feature_columns" : ",".join(feature_columns),
#         "target_columns" : ",".join(target_columns), #json dumps
#         "model_data_path":train_output_dbfs_path,
#         "media_artifacts_path" : media_artifacts_path,
#         },
#     )

# except Exception as e:
#     print(str(e))

# COMMAND ----------

# # try: 
# #     #define media artifacts path
# #     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
# #         sdk_session_id = sdk_session_id,
# #         dbutils = dbutils,
# #         env = env)
    
# #     print(media_artifacts_path)

# #     custom_notebook_result = dbutils.notebook.run(
# #         "Bias_Fairness_Evaluation",
# #         0,
# #         arguments = 
# #         {
# #         "feature_columns" : ",".join(feature_columns),
# #         "target_columns" : ",".join(target_columns), #json dumps
# #         "model_data_path":train_output_dbfs_path,
# #         "media_artifacts_path" : media_artifacts_path,
# #         },
# #     )

# # except Exception as e:
# #     print(str(e))


# from MLCORE_SDK.sdk.manage_sdk_session import get_session
# from MLCORE_SDK.helpers.auth_helper import get_container_name
# from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

# res = get_session(sdk_session_id,dbutils=dbutils)
# project = res.json()["data"]["project_id"]
# version = res.json()["data"]["version"]

# az_container_name = get_container_name(dbutils)
# job_id, run_id = get_job_id_run_id(dbutils)
# job_run_str = str(job_id) + "/" + str(run_id)

# if not env:
#     env = dbutils.widgets.get("env")

# media_artifacts_path = f"/mnt/{az_container_name}/{env}/media_artifacts/{project}/{version}/{job_run_str}"
# report_directory = f"{env}/media_artifacts/{project}/{version}/{job_run_str}"

# COMMAND ----------

# try :
#     #define media artifacts path
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Bias_Fairness", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#         "feature_columns" : ",".join(feature_columns),
#         "target_columns" : ",".join(target_columns), #json dumps
#         "model_data_path":train_output_dbfs_path,
#         "media_artifacts_path" : media_artifacts_path,
#         "report_directory" : report_directory,
#         "sensitive_columns" : ",".join(sensitive_columns),
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")

# COMMAND ----------

# try :
#     #define media artifacts path
#     print(media_artifacts_path)
#     dbutils.notebook.run(
#         "Model_Test", 
#         timeout_seconds = 0, 
#         arguments = 
#         {
#         "feature_columns" : ",".join(feature_columns),
#         "target_columns" : ",".join(target_columns), #json dumps
#         "model_data_path":train_output_dbfs_path,
#         "model_name": model_name,
#         "media_artifacts_path" : media_artifacts_path,
#         "report_directory" : report_directory,
#             })
# except Exception as e:
#     print(f"Exception while triggering EDA notebook : {e}")
