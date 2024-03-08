# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %pip install fairlearn
# MAGIC %pip install pandas==1.0.5
# MAGIC %pip install numpy==1.19.1

# COMMAND ----------

import pandas as pd
import numpy as np
from fairlearn.metrics import *
from MLCORE_SDK import mlclient
from sklearn.metrics import accuracy_score, precision_score

# COMMAND ----------

modelling_task_type = "Classification"
train_output_path = dbutils.widgets.get("model_data_path")
features = dbutils.widgets.get("feature_columns").split(",")
target = [dbutils.widgets.get("target_columns")]
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

data = spark.read.load(train_output_path)
y_true = data.select(target).toPandas()
y_pred = data.select('prediction').toPandas()

# COMMAND ----------

def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.

    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage

    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.

    """
    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols

# COMMAND ----------

print(f"Select the categorical columns from this list : {detect_categorical_cols(data[features])}")

# COMMAND ----------

columns_to_check_bias = ['Education','Marital_Status']

# COMMAND ----------

def save_plots_full(subplot_array, plot_name):
    for i, row in enumerate(subplot_array):
        for j, subplot in enumerate(row):
            # Generate a unique filename for each subplot based on its position
            #filename = f"/dbfs/FileStore/sdk/subplot_news.svg"  # You can adjust the filename format as needed

            # Get the figure associated with the subplot
            fig = subplot.get_figure()

            # Save the entire figure (including titles and legends) as an image
            #fig.savefig(filename)
            mlclient.log(operation_type = "register_plots",dbutils = dbutils, figure_to_save=fig, plot_name=plot_name,folder_name = "Model_Evaluation",request_type="push_plot", ext = "jpeg") 
             
def write_metrics_report(y_true, y_pred, sensitive_features, col_name):
    # Create a string to store HTML content
    html_content = f"<!DOCTYPE html>\n<html>\n<head>\n<title>Fairness Metrics for {col_name}</title>\n</head>\n<body>\n"
    html_content += "<h2>Fairness Metrics:</h2>\n"

    # Calculate fairness metrics and add explanations and recommendations to the HTML string
    def add_to_html(content):
        nonlocal html_content
        html_content += content + "<br>\n"

    try : 
        dp_difference = demographic_parity_difference(y_true = y_true,y_pred = y_pred, sensitive_features = sensitive_features)
    except Exception as e:
        print(f"Exception : {e}")
        dp_difference = "NotCalculated"
    add_to_html("<b>Demographic Parity Difference:</b> " + str(dp_difference))
    add_to_html("<i>Interpretation:</i> The demographic parity difference is a measure of the disparity in selection rates between different demographic groups.")
    add_to_html("<i>Recommendations:</i> To improve fairness, strive to reduce the difference in selection rates among different groups in the predictions.")
    
    try : 
        dp_ratio = demographic_parity_ratio(y_true = y_true,y_pred = y_pred, sensitive_features = sensitive_features)
    except Exception as e:
        print(f"Exception : {e}")
        dp_ratio = "NotCalculated"
    add_to_html("<b>Demographic Parity Ratio:</b> " + str(dp_ratio))
    add_to_html("<i>Interpretation:</i> The demographic parity ratio compares the favorable outcomes ratio between different demographic groups.")
    add_to_html("<i>Recommendations:</i> Aim for a ratio closer to 1 to ensure equitable outcomes across various demographic groups.")
    
    try : 
        eo_difference = equalized_odds_difference(y_true = y_true,y_pred = y_pred, sensitive_features = sensitive_features)
    except Exception as e:
        print(f"Exception : {e}")
        eo_difference = "NotCalculated"
    add_to_html("<b>Equalized Odds Difference:</b> " + str(eo_difference))
    add_to_html("<i>Interpretation:</i> The equalized odds difference evaluates the disparity in true positive rates among different demographic groups.")
    add_to_html("<i>Recommendations:</i> Work on minimizing the difference in true positive rates across various demographic groups to enhance fairness.")
    
    try : 
        eo_ratio = equalized_odds_ratio(y_true = y_true,y_pred = y_pred, sensitive_features = sensitive_features)
    except Exception as e:
        print(f"Exception : {e}")
        eo_ratio = "NotCalculated"
    add_to_html("<b>Equalized Odds Ratio:</b> " + str(eo_ratio))
    add_to_html("<i>Interpretation:</i> The equalized odds ratio compares the true positive rates ratio between different demographic groups.")
    add_to_html("<i>Recommendations:</i> Strive for a ratio closer to 1 to ensure fairness in predicting positive outcomes among various demographic groups.")

    html_content += "</body>\n</html>"

    mlclient.log(
        operation_type = "register_plots",
        dbutils = dbutils,
        figure_to_save=html_content,
        plot_name=f"FairnessMetricsReport_{col_name}",
        folder_name = "Model_Evaluation",
        request_type="push_plot",
        lib = "gx") 
    
# Analyze metrics using MetricFrame
if modelling_task_type.lower() == "classification":
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        'true_negative_rate' : true_negative_rate,
        'true_positive_rate' : true_positive_rate,
        "selection rate": selection_rate,
        "count": count,
    }
else :
    metrics = {
        'mean_prediction': mean_prediction,
        'mean_absolute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        'r_squared': r2_score,
        'explained_variance': explained_variance_score,
        'max_error': max_error,             
    }

# COMMAND ----------

for bias_check_col in columns_to_check_bias:
    sensitive_features = data.select(bias_check_col).toPandas()
    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )
    bias_plots = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[20, 12],
        title=f"Bias Report for {bias_check_col}",
    )

    # Save the plots to DBFS
    save_plots_full(bias_plots, f"BiasAndFairness_{bias_check_col}")

    # Save fairness metrics Report
    write_metrics_report(y_true, y_pred, sensitive_features, bias_check_col)
