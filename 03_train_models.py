# Databricks notebook source
# Import required libraries for ML model training and feature engineering
import pyspark.pandas as ps
from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
import databricks.automl_runtime
import time

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data from Databricks Feature Engineering

# COMMAND ----------

# Initialize Feature Engineering client to access the stored features
fe = FeatureEngineeringClient()

# COMMAND ----------

# Helper function to find a usable Unity Catalog with our data
def get_unity_catalog():
    try:
        # Get list of catalogs
        catalogs = spark.sql("SHOW CATALOGS").collect()

        # Look for a workspace-specific catalog (not system, hive_metastore, or samples)
        for catalog_row in catalogs:
            catalog_name = catalog_row.catalog
            if catalog_name not in ['system', 'hive_metastore', 'samples']:
                print(f"Checking catalog: {catalog_name}")
                # Try to use this catalog
                try:
                    # Try to access the tables directly to verify access
                    try:
                        # Try to query the catalog directly
                        test_df = spark.sql(f"SHOW SCHEMAS IN {catalog_name}")
                        schema_list = [row.databaseName for row in test_df.collect()]
                        print(f"Schemas in {catalog_name}: {schema_list}")

                        if "implied_volatility" in schema_list:
                            print(f"Found implied_volatility schema in catalog: {catalog_name}")
                            # Try to confirm table access
                            try:
                                table_test = spark.sql(f"SELECT COUNT(*) FROM {catalog_name}.implied_volatility.features").collect()
                                print(f"Successfully accessed features table in {catalog_name}")
                                return catalog_name
                            except Exception as e:
                                print(f"Cannot access tables in {catalog_name}.implied_volatility: {e}")
                    except Exception as e:
                        print(f"Error checking schemas in {catalog_name}: {e}")
                except Exception as e:
                    print(f"Cannot use catalog {catalog_name}: {e}")

        raise Exception("No Unity Catalog with implied_volatility schema found. Please run notebook 02 first.")
    except Exception as e:
        raise Exception(f"Error accessing Unity Catalog: {e}")

# Try to access a known catalog directly first, then fallback to discovery
try:
    print("Trying direct table access first...")
    # Try the known catalog directly
    direct_catalog = "legend_dbw_test_2357533909889557"
    test_query = spark.sql(f"SELECT COUNT(*) FROM {direct_catalog}.implied_volatility.features").collect()
    print(f"Successfully accessed {direct_catalog} directly")
    catalog_name = direct_catalog
except Exception as e:
    print(f"Direct access failed: {e}")
    print("Falling back to catalog discovery...")
    catalog_name = get_unity_catalog()

print(f"Using catalog: {catalog_name}")

# Read tables from Unity Catalog Feature Engineering
features_df = spark.table(f"{catalog_name}.implied_volatility.features").toPandas()
labels_df = spark.table(f"{catalog_name}.implied_volatility.labels").toPandas()

# COMMAND ----------

# For simplicity, we'll train a model for just one label (0.05_0.95)
# In a real scenario, we might train models for each strike/maturity combination
features_df = features_df.iloc[:, 1:]  # Remove the first column (likely an index)
features_df['target'] = labels_df['0.05_0.95']  # Add one label as our target

# COMMAND ----------

# Define target column and feature columns
target_col = "target"
training_col = list(features_df.columns)[:-1]  # All columns except target

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

# Create a ColumnSelector to ensure we only use supported columns
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = training_col
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# Initialize transformer list for preprocessing pipeline
transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

# Set up scikit-learn preprocessing pipeline for numerical features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Create imputer for handling missing values in numerical columns
num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), training_col))

# Create pipeline for preprocessing numerical data:
# 1. Convert to numeric (coercing errors)
# 2. Impute missing values
# 3. Standardize features (zero mean, unit variance)
numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

# Add numerical preprocessing to transformers list
transformers.append(("numerical", numerical_pipeline, training_col))

# COMMAND ----------

# Combine all preprocessing steps into a single ColumnTransformer
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

# Split data into train, validation, and test sets
from sklearn.model_selection import train_test_split

split_X = features_df.drop([target_col], axis=1)  # Features
split_y = features_df[target_col]  # Target

# Split out train data (60% of the dataset)
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=224145758)

# Split remaining data equally for validation and test (20% each)
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=224145758)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3624174420594729/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

# Import XGBoost regressor model
from xgboost import XGBRegressor

# COMMAND ----------

# Set up MLflow tracking and scikit-learn config
import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

# Set sklearn to display pipeline diagrams
set_config(display='diagram')

# Create XGBoost model with optimized hyperparameters
xgb_regressor = XGBRegressor(
  colsample_bytree=0.6385875217228281,  # Fraction of features to use per tree
  learning_rate=0.10603131742006,       # Step size shrinkage to prevent overfitting
  max_depth=6,                          # Maximum tree depth
  min_child_weight=8,                   # Minimum sum of instance weight needed in a child
  n_estimators=148,                     # Number of trees to build
  n_jobs=100,                           # Parallel jobs for computation
  subsample=0.5203076979604147,         # Fraction of samples used for fitting trees
  verbosity=0,                          # Silent mode
  random_state=224145758,               # For reproducibility
)

# Create full pipeline: column selection -> preprocessing -> model
model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("regressor", xgb_regressor),
])

# Create a separate preprocessing pipeline for validation data
# This is used for early stopping during training
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

# Disable MLflow autologging temporarily (we'll handle logging manually)
mlflow.sklearn.autolog(disable=True)

# Fit preprocessing pipeline to training data and transform validation data
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

# COMMAND ----------

# Get current username for MLflow experiment naming
# If not available, generate a unique ID
try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")

# COMMAND ----------

# Import numpy for calculating metrics
import numpy as np

# Enable MLflow autologging for scikit-learn models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

# Create or get MLflow experiment for tracking
experiment_name = f'/Users/{username}/implied_volatility'
try:
  experiment_id_ = mlflow.create_experiment(experiment_name)
except:
  experiment_id_ = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Function to evaluate model metrics and log them to MLflow
def evaluate_and_log_metrics(model, X, y, prefix=""):
    """
    Evaluate model performance on given data and log metrics to MLflow

    Args:
        model: Trained sklearn model
        X: Features
        y: Target values
        prefix: Prefix for metric names (e.g., 'val_' or 'test_')

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    metrics = {
        f"{prefix}r2_score": sklearn.metrics.r2_score(y, y_pred),
        f"{prefix}mean_absolute_error": sklearn.metrics.mean_absolute_error(y, y_pred),
        f"{prefix}mean_squared_error": sklearn.metrics.mean_squared_error(y, y_pred),
        f"{prefix}root_mean_squared_error": np.sqrt(sklearn.metrics.mean_squared_error(y, y_pred))
    }

    # Log metrics to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    return metrics

# Start MLflow run, train model, and log metrics
with mlflow.start_run(experiment_id=experiment_id_, run_name=f"implied_volatility_{time.time()}") as mlflow_run:
    # Train model with early stopping using validation data
    model.fit(X_train, y_train,
              regressor__early_stopping_rounds=5,  # Stop if no improvement after 5 rounds
              regressor__eval_set=[(X_val_processed, y_val)],
              regressor__verbose=False)

    # Training metrics are logged by MLflow autologging

    # Log metrics for the validation set
    xgb_val_metrics = evaluate_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set (held-out data)
    xgb_test_metrics = evaluate_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics in a table
    xgb_val_metrics_display = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics_display = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}
    display(pd.DataFrame([xgb_val_metrics_display, xgb_test_metrics_display], index=["validation", "test"]))

# COMMAND ----------

# Display all runs from our MLflow experiment
display(spark.read.format("mlflow-experiment").load(experiment_id_))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### MLFlow UI
# MAGIC
# MAGIC MLFlow has a UI component, making tracking of experiments extremely easy.
# MAGIC
# MAGIC <img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/Screenshot+2022-07-29+at+1.55.57+PM.png'  style="float: left" width="1150px" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
# SHAP plots help explain model predictions and feature importance
shap_enabled = False

# COMMAND ----------

# If SHAP is enabled, calculate and display feature importance
if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]))

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(10, X_val.shape[0]))

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)