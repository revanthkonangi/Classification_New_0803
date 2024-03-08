# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %pip install -U kaleido

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
train_output_path = dbutils.widgets.get("model_data_path")
features = dbutils.widgets.get("feature_columns").split(",")
target = dbutils.widgets.get("target_columns")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

import sklearn
import mlflow
import warnings
from MLCORE_SDK import mlclient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import *
from mlflow.tracking import MlflowClient
warnings.filterwarnings("ignore")

# COMMAND ----------

client = MlflowClient()
model_versions = client.get_latest_versions(model_name)
model_version = model_versions[0].version

# COMMAND ----------

train_output_df = spark.read.load(train_output_path).toPandas()
train_output_df.display()

# COMMAND ----------

train_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "train"]
test_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "test"]

# COMMAND ----------

x_train = train_df[features]
y_train = train_df[target]
y_pred_train = train_df['prediction']

x_test = test_df[features]
y_test = test_df[target]
y_pred_test = test_df['prediction']

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
model = loaded_model[-1]
type(model)

# COMMAND ----------

# Compute the confusion matrices for training and test data
cms = [confusion_matrix(y_train, y_pred_train), confusion_matrix(y_test, y_pred_test)]
titles = ['Training Data Confusion Matrix', 'Test Data Confusion Matrix']
cmaps = ['Blues', 'Greens']

# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, ax in enumerate(axes):
    sns.heatmap(
        cms[i],
        annot=True,
        fmt="d",
        cmap=cmaps[i],
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar=True,  # Add color bar
        cbar_kws={"orientation": "vertical"},  # Color bar orientation
        annot_kws={"size": 14},
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title(titles[i], fontsize=16)  # Improved titles
    ax.set_xticks(np.arange(len(cms[i])) + 0.5)
    ax.set_yticks(np.arange(len(cms[i])) + 0.5)
    ax.set_xticklabels(['0', '1'], rotation=0)
    ax.set_yticklabels(['0', '1'], rotation=0)

# Adjust spacing between subplots
plt.tight_layout()

# Show the subplots
plt.show()


# COMMAND ----------

# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig, 
    plot_name='Confusion_matrix_plot',
    folder_name = "Model_Evaluation",
    request_type="push_plot")   

# COMMAND ----------

# Predict probabilities on the training and test sets
y_scores = [model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]]
datasets = ['Training', 'Test']

# Initialize the figure with subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=[f'{dataset} ROC Curve' for dataset in datasets])

for i, dataset in enumerate(datasets):
    # Compute ROC curve and AUC score for the current dataset
    fpr, tpr, thresholds = roc_curve([y_train, y_test][i], y_scores[i])
    roc_auc = roc_auc_score([y_train, y_test][i], y_scores[i])

    # Add ROC curve trace to the subplot
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f'ROC Curve (AUC = {roc_auc:.2f})'
        ),
        row=1, col=i+1
    )

    # Set axis titles and show x/y axis lines
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=i+1, showline=True, linecolor='black', showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=i+1, showline=True, linecolor='black', showgrid=True, gridcolor='lightgray')

    # Show legend inside the subplot
    fig.update_traces(showlegend=True, selector=dict(type='scatter'))

# Update layout to set background color to white
fig.update_layout(plot_bgcolor='white')

# Show the interactive plot with legends inside each subplot
fig.show()


# COMMAND ----------

# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig, 
    plot_name="ROC_AUC_plot",
    lib="plotly",
    ext = "html",
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

import plotly.subplots as sp
import plotly.graph_objs as go
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_scores, dataset_name, precision_color, recall_color, legend_prefix):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    fig = go.Figure()

    # Add precision line (without markers)
    fig.add_trace(go.Scatter(x=thresholds, y=precision[:-1], mode='lines', name=f'{legend_prefix} Precision', line=dict(color=precision_color, width=4)))

    # Add recall line (without markers)
    fig.add_trace(go.Scatter(x=thresholds, y=recall[:-1], mode='lines', name=f'{legend_prefix} Recall', line=dict(color=recall_color, width=4)))

    return fig

def plot_precision_recall_subplots(x_train, y_train, x_test, y_test, model):
    # Predict probabilities on the training set
    y_scores_train = model.predict_proba(x_train)[:, 1]

    # Predict probabilities on the test set
    y_scores_test = model.predict_proba(x_test)[:, 1]

    # Create subplots with 1 row and 2 columns
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Training Precision-Recall Curve', 'Test Precision-Recall Curve'))

    # Loop to generate subplots
    for i, (y_true, y_scores, dataset_name, precision_color, recall_color, legend_prefix) in enumerate(
        [(y_train, y_scores_train, 'Training', 'green', 'red', 'Train'),
         (y_test, y_scores_test, 'Test', 'orange', 'blue', 'Test')]
    ):
        pr_curve = plot_precision_recall_curve(y_true, y_scores, dataset_name, precision_color, recall_color, legend_prefix)
        
        # Add traces to the subplots
        for trace in pr_curve.data:
            fig.add_trace(trace, row=1, col=i+1)

        # Add y-grid lines to the subplot
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=1, col=i+1)

    # Customize the layout
    for col in [1, 2]:
        fig.update_xaxes(title_text="Threshold", showline=True, linecolor='black', row=1, col=col)
        fig.update_yaxes(title_text="Probability Score", showline=True, linecolor='black', row=1, col=col)

    # Update layout to set background color to white
    fig.update_layout(plot_bgcolor='white')

    # Show the interactive plot with precision-recall curves side by side
    return fig

# Usage
fig = plot_precision_recall_subplots(x_train, y_train, x_test, y_test, model)


# COMMAND ----------

fig

# COMMAND ----------

# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig, 
    plot_name="PR_Curve",
    lib="plotly",
    ext = "html",
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

coefficients = model.coef_

avg_importance = np.mean(np.abs(coefficients), axis=0)

# Create a DataFrame
feature_importance = pd.DataFrame({'Feature': features, 'Importance': avg_importance})

# Sort the DataFrame by importance in ascending order
feature_importance = feature_importance.sort_values('Importance', ascending=True)

# Create an interactive horizontal bar plot using Plotly Express
fig = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',  # Horizontal orientation
    title='Feature Importance',
    labels={'Importance': 'Average Importance'},
    height=400  # Adjust the height as needed
)

# Customize the layout
fig.update_layout(
    xaxis_title='Average Importance',
    yaxis_title='Feature',
    showlegend=False,  # Hide legend
    plot_bgcolor='white',
)

# Show gridlines for both x-axis and y-axis
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Show the interactive plot
fig.show()


# COMMAND ----------

# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig, 
    plot_name="Feature_Importance",
    lib="plotly",
    ext = "html",
    folder_name = "Model_Evaluation",
    request_type="push_plot")
