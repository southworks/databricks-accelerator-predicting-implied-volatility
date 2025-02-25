# Databricks notebook source
# MAGIC %pip install tensorflow==2.9.0 tensorflow-probability==0.17.0 tf_quant_finance numpy==1.22.4

# COMMAND ----------

# Import required libraries for quantitative finance modeling
import numpy as np
import tensorflow as tf
import tf_quant_finance as tff
from tf_quant_finance.math import *
from tf_quant_finance.math.piecewise import *

from tf_quant_finance.models import *
from tf_quant_finance.models.generic_ito_process import *

import time

import scipy.optimize as optimize

import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1. Model Setup
# MAGIC Let's start by defining a *toy model* (generic Ito Process), which will be a function of specific model parameters.
# MAGIC
# MAGIC This model can be used to price call options with specific *maturities* and *strikes* and therefore implied vols (using blackscholes one to one mapping between price and implied vol).
# MAGIC
# MAGIC Our aim is to find the values of model parameters such that implied vols calculated from *this* model match with implied vols obtained from market (this process is called calibration).
# MAGIC
# MAGIC ## 1.1. Model Definition
# MAGIC Creating a toy model definition by following [lognormal](https://en.wikipedia.org/wiki/Log-normal_distributio) fx, [vasicek](https://en.wikipedia.org/wiki/Vasicek_model) ir & [local vol](https://en.wikipedia.org/wiki/Local_volatility) fx_vol.

# COMMAND ----------

class TimeSeries:
  """ Container that represent piecewise functions of time, compatible with XLA"""
  def __init__(self,jump_locations, values):
    self.jump_locations = jump_locations  # Time points where function values can change
    self.values = values                  # Function values at each time point

  def apply(self, input):
    # Return the appropriate value based on the input time
    res = self.values[-1]  # Default to last value
    for idx in range(len(self.jump_locations)):
      curr_jump_loc = self.jump_locations[idx]
      if input <= curr_jump_loc:
        res = self.values[idx]
    return res

class BlackScholesWithVasicelAndLocalVol(GenericItoProcess):
  """Toy Model for lognormal fx, vasicek ir & local vol fx"""

  def __init__(self,
               # rate 1 model parameters
               kappa_rate_1, theta_rate_1, vol_rate_1, fwd_rate_1,
               # rate 2 model parameters
               kappa_rate_2, theta_rate_2, vol_rate_2, fwd_rate_2,
               # fx vol model parameters
               jump_strikes, local_vol_fx,
               # fx model parameters
               fx_fwd,
               # correlation matrix
               corr_matrix,
               # discretization jump dt
               step_size,
               # numerical accuracy specifics
               dtype=None):

    # basic variables from parent class 'GenericItoProcess'
    self._name = 'BlackScholesWithVasicelAndLocalVol'
    self._dim = 4  # 4-dimensional process: rate1, rate2, fx_vol, log_fx
    self._dtype = dtype

    # rate 1 model parameters - mean-reversion rate, long-term mean, volatility, forward rate
    self.kappa_rate_1 = kappa_rate_1  # Speed of mean reversion
    self.theta_rate_1 = theta_rate_1  # Long-term mean
    self.vol_rate_1 = vol_rate_1      # Volatility of rate
    self.fwd_rate_1 = fwd_rate_1      # Forward rate curve

    # rate 2 model parameters - same structure as rate 1
    self.kappa_rate_2 = kappa_rate_2
    self.theta_rate_2 = theta_rate_2
    self.vol_rate_2 = vol_rate_2
    self.fwd_rate_2 = fwd_rate_2

    # fx vol model parameters - local volatility surface definition
    self.jump_strikes = jump_strikes
    self.log_jump_strikes = tf.math.log(jump_strikes)  # Log transformation for numerical stability
    self.local_vol_fx = local_vol_fx

    # fx model parameters - forward FX rates
    self.fx_fwd = fx_fwd

    # discretization jump dt - for numerical simulation
    self.step_size = step_size

    # correlation matrix - defines relationships between process components
    self.cholesky = tf.linalg.cholesky(corr_matrix)  # Cholesky decomposition for correlated random sampling

  def _volatility_fn(self, t, x):
    # Define the volatility function for the stochastic process
    # t: time
    # x: state vector [rate_factor_1, rate_factor_2, vol_fx, log_fx]

    vol_fx = x[..., 2]  # Current FX volatility
    zeros = tf.zeros_like(vol_fx)
    ones = tf.ones_like(vol_fx)

    # Get volatilities for each component at current time t
    vol_rate_1 = self.vol_rate_1.apply(t) * ones
    vol_rate_2 = self.vol_rate_2.apply(t) * ones
    vol_vol_fx = zeros  # Volatility of volatility is zero in this model

    vol_array = [vol_rate_1, vol_rate_2, vol_vol_fx, vol_fx]

    # Construct volatility matrix using Cholesky factor for correlation
    columns = []
    for col in range(self._dim):
      current_columns = []
      for row in range(self._dim):
        current_columns.append(self.cholesky[row][col] * vol_array[row])
      columns.append(tf.stack(current_columns, -1))

    result_matrix = tf.stack(columns, -1)
    return result_matrix

  def _drift_fn(self, t, x):
    # Define the drift function for the stochastic process
    # t: time
    # x: state vector [rate_factor_1, rate_factor_2, vol_fx, log_fx]

    rate_factor_1 = x[..., 0]
    rate_factor_2 = x[..., 1]
    vol_fx = x[..., 2]
    log_fx = x[..., 3]

    # Get forward rates at current time
    fwd_rate_1_t = self.fwd_rate_1.apply(t)
    fwd_rate_2_t = self.fwd_rate_2.apply(t)

    # Calculate actual rates by adding factors to forward rates
    rate_1 = fwd_rate_1_t + rate_factor_1
    rate_2 = fwd_rate_2_t + rate_factor_2

    # Get local volatility at current time and log spot price
    lv_for_current_t = self.local_vol_fx.apply(t)
    lv_func = PiecewiseConstantFunc(jump_locations=self.log_jump_strikes, values=lv_for_current_t, dtype=dtype)
    new_vol_fx = lv_func(log_fx)

    self.old_vol = new_vol_fx

    # Calculate drifts for each component using appropriate models
    # Rate 1 - Vasicek model drift
    drift_rate_1 = self.kappa_rate_1.apply(t) * (self.theta_rate_1.apply(t) - rate_factor_1)
    # Rate 2 - Vasicek model drift
    drift_rate_2 = self.kappa_rate_2.apply(t) * (self.theta_rate_2.apply(t) - rate_factor_2)
    # FX volatility drift - moving toward local volatility surface
    drift_vol_fx = (new_vol_fx - vol_fx)/self.step_size
    # Log FX drift - risk-neutral drift adjusted for volatility
    drift_fx = (rate_1 - rate_2) - 0.5 * vol_fx * vol_fx

    drift = tf.stack([drift_rate_1, drift_rate_2, drift_vol_fx, drift_fx], -1)
    return drift

  def implied_vol(self,
                  option_strikes,
                  option_maturities,
                  num_samples):
    """
    Calculates Black-Scholes implied volatilities for options with given strikes and maturities
    using Monte Carlo simulation of the model

    Args:
        option_strikes: Array of option strike prices
        option_maturities: Array of option maturities (in years)
        num_samples: Number of Monte Carlo paths to simulate

    Returns:
        Array of implied volatilities for each maturity and strike
    """

    # Simulate paths of the process
    paths = self.sample_paths(
          option_maturities,
          num_samples=num_samples,
          initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=self._dtype.name),
          time_step=self.step_size,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[42, 56])

    number_of_strikes = len(option_strikes)
    implied_vols = []

    # For each maturity, calculate implied volatilities
    for maturity_idx in range(len(option_maturities)):
      curr_maturity = option_maturities[maturity_idx]
      curr_paths = paths[:,maturity_idx]
      curr_fwd = self.fx_fwd.apply(curr_maturity)

      # Calculate discount factors from simulated rates
      df = tf.exp(-curr_paths[:,0]*curr_maturity)
      df_mean = tf.math.reduce_mean(df)

      # Calculate FX rates from log process
      fx = curr_fwd * tf.exp(curr_paths[:,3])

      prices = []
      # Calculate option prices for each strike by averaging payoffs
      for strike_idx in range(number_of_strikes):
        curr_strike = option_strikes[strike_idx]
        price = tf.math.reduce_mean(tf.maximum(tf.constant(0.0, dtype=self._dtype), (fx - curr_strike)))
        prices.append(price)

      # Calculate implied volatilities from option prices using Black-Scholes formula
      implied_vols_for_curr_expiry = tff.black_scholes.implied_vol(
          prices=prices,
          strikes=option_strikes,
          expiries= [curr_maturity] * number_of_strikes,
          forwards=[curr_fwd] * number_of_strikes,
          discount_factors= [df_mean] * number_of_strikes,
          is_call_options=True)

      # Clean up any NaN values in the implied vols
      implied_vols_for_curr_expiry_parsed = []
      for option_idx in range(0, len(implied_vols_for_curr_expiry)):
        curr_implied_vol = implied_vols_for_curr_expiry[option_idx]
        if np.isnan(curr_implied_vol):
          # Fall back to intrinsic value when implied vol calculation fails
          curr_implied_vol = min(0, prices[option_idx].numpy() - max(curr_fwd - option_strikes[option_idx], 0))
        implied_vols_for_curr_expiry_parsed.append(curr_implied_vol)

      implied_vols.append(implied_vols_for_curr_expiry_parsed)

    return implied_vols

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2. Model Initialization
# MAGIC
# MAGIC Initialize the model with dummy values of model parameters

# COMMAND ----------

# Let's instantiate a model with dummy model parameter values

dtype=tf.float64  # Use 64-bit precision for numerical stability

# Time points where model parameters can change
jump_locations = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1])

# Strike prices for local volatility surface
jump_strikes = np.array([0.95, 0.99, 1, 1.001, 1.05])

# Local volatility surface - one of the key parameters we'll be optimizing
# Matrix of volatilities: rows = maturities, columns = strike prices
lv_surface = [[0.6, 0.36, 0.246, 0.546, 0.7978],
              [0.68, 0.37, 0.112, 0.476, 0.8987],
              [0.65, 0.33, 0.224, 0.676, 0.764],
              [0.634, 0.336, 0.332, 0.566, 0.907],
              [0.76, 0.456, 0.152, 0.601, 0.67],
              [0.676, 0.3745, 0.1632, 0.623, 0.788],
              [0.687, 0.243, 0.2123, 0.622, 0.7576],
              [0.576, 0.473, 0.253, 0.556, 0.7123],
              [0.56, 0.346, 0.2252, 0.786, 0.867],
              [0.786, 0.354, 0.2691, 0.634, 0.7545]]

# Create model instance with initial parameters
model = BlackScholesWithVasicelAndLocalVol(
    # Rate 1 parameters - mean reversion rate, mean level, volatility, and forward curve
    kappa_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.05, 0.02, 0.07, 0.02, 0.04, 0.06, 0.07, 0.02, 0.08, 0.09],dtype=dtype.name)),
    theta_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([1.2, 2, 1.5, 1.7, 1, 1.3, 1.9, 3.0, 2.5, 1.0],dtype=dtype.name)),
    vol_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.11, 0.15, 0.9, 0.15,  0.15, 0.3, 0.15, 0.2, 0.17, 0.4],dtype=dtype.name)),
    fwd_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.02, 0.021, 0.022, 0.023, 0.019, 0.018, 0.23, 0.025, 0.015, 0.019],dtype=dtype.name)),

    # Rate 2 parameters - structure mirrors rate 1
    kappa_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.05, 0.02, 0.07, 0.02, 0.04, 0.06, 0.07, 0.02, 0.08, 0.09],dtype=dtype.name)),
    theta_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([1.2, 2, 1.5, 1.7, 1, 1.3, 1.9, 3.0, 2.5, 1.0],dtype=dtype.name)),
    vol_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.11, 0.15, 0.9, 0.15,  0.15, 0.3, 0.15, 0.2, 0.17, 0.4],dtype=dtype.name)),
    fwd_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.025, 0.051, 0.052, 0.053, 0.069, 0.068, 0.53, 0.055, 0.075, 0.049],dtype=dtype.name)),

    # FX volatility parameters
    jump_strikes = jump_strikes,
    local_vol_fx = TimeSeries(jump_locations=jump_locations, values=lv_surface),

    # FX forward rates
    fx_fwd = TimeSeries(jump_locations=jump_locations, values=np.array([1.0, 1.002, 0.998, 1.04, 1.035, 1.01, 0.999, 0.998, 1.003, 1.01],dtype=dtype.name)),

    # Monte Carlo simulation step size
    step_size=0.01,

    # Correlation matrix between the 4 dimensions
    corr_matrix = tf.constant([[1.0, 0.2, 0.0, 0.3],
                               [0.2, 1.0, 0.0, 0.3],
                               [0.0, 0.0, 1.0, 0.8],
                               [0.3, 0.3, 0.8, 1.0]], dtype=dtype),
    dtype=dtype
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3. Model Calibration
# MAGIC
# MAGIC To keep things simple let's just try to find the *lv_surface* that will map closest match to our target i.e implied_vol from market.
# MAGIC
# MAGIC Moment of truth: Given the input dimension of objective function is 50 ('local_vol_fx' surface of size 10x5 i.e 10 maturities by 5 strikes) and similarly output dimension is 50 as well, below cell execution will take forever. (if one really wants to test the execution, can reduce *option_maturities* to size of 1 array)

# COMMAND ----------

start_time_ = time.time()

# These are all the maturities we want to reprice our call options
option_maturities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

number_of_maturities = len(option_maturities)

# These are all the strikes (on each of above 'option_maturities') of those call options
option_strikes = np.array([0.95, 0.99, 1, 1.001, 1.05])
number_of_strikes = len(option_strikes)

# Number of Monte Carlo paths - more paths increase accuracy but also computation time
num_samples = 10

# Let's say this is the implied vol coming from market
# Market-observed implied volatilities that we're trying to match
implied_vol_target = np.array([ [ 0.36165047, 0.38006929, 0.38622322, 0.38688029, 0.42223257 ],
       [ 0.49930734, 0.52086326, 0.52665937, 0.52725076, 0.5561736 ],
       [ 0.53926329, 0.56007775, 0.56558462, 0.56613531, 0.59324591 ],
       [ 0.55870456, 0.57847004, 0.5836174 , 0.58413716, 0.60937169 ],
       [ 0.58499617, 0.60206912, 0.60646861, 0.60690938, 0.62796077 ],
       [ 0.61917847, 0.63211846, 0.63541694, 0.63574893, 0.65183933 ],
       [ 0.64890557, 0.65896691, 0.66159441, 0.66185734, 0.67469815 ],
       [ 0.68506316, 0.69196701, 0.69382324, 0.69401138, 0.70335816 ],
       [ 0.74270218, 0.74599681, 0.74702723, 0.74713323, 0.75264819 ],
       [ 0.8005369 , 0.79634181, 0.79554414, 0.79546886, 0.7927459 ] ])


implied_vol_target = implied_vol_target[:number_of_maturities]

# This is the initial guess for our LV surface (one of the model parameters of model)
lv_surface_init_guess = np.array([[0.6, 0.36, 0.246, 0.546, 0.7978],
                         [0.68, 0.37, 0.112, 0.476, 0.8987],
                         [0.65, 0.33, 0.224, 0.676, 0.764],
                         [0.634, 0.336, 0.332, 0.566, 0.907],
                         [0.76, 0.456, 0.152, 0.601, 0.67],
                         [0.676, 0.3745, 0.1632, 0.623, 0.788],
                         [0.687, 0.243, 0.2123, 0.622, 0.7576],
                         [0.576, 0.473, 0.253, 0.556, 0.7123],
                         [0.56, 0.346, 0.2252, 0.786, 0.867],
                         [0.786, 0.354, 0.2691, 0.634, 0.7545]])

lv_surface_init_guess = lv_surface_init_guess[:number_of_maturities]

# Let's define our objective function for calibration
def objective_fn(lv_surface_guess_flatened):
  """
  Objective function for calibration that measures the difference between
  model-implied vols and target market vols

  Args:
      lv_surface_guess_flatened: Flattened local volatility surface

  Returns:
      Vector of errors between model-implied and target vols (in basis points)
  """
  # Reshape flattened LV surface to 2D array
  lv_surface_guess = np.split(lv_surface_guess_flatened, number_of_maturities)

  # Update model with current guess for local volatility
  model.local_vol_fx = TimeSeries(jump_locations=option_maturities[:-1], values=lv_surface_guess)

  # Calculate implied vols from the model with current parameters
  implied_vols_from_model = model.implied_vol(option_maturities=option_maturities,
                                             option_strikes=option_strikes,
                                             num_samples=num_samples)

  # Calculate errors in basis points (scaled by 10000)
  errors = np.array((implied_vol_target - implied_vols_from_model)).flatten() * 1e4
  # print("errors:", errors)
  return errors

# Run calibration using least squares optimizer
# This attempts to find the local volatility surface that minimizes
# the difference between model and market implied vols
roots = optimize.least_squares(objective_fn,
                      x0=lv_surface_init_guess.flatten(),
                      ftol=0.05,  # Function tolerance for convergence
                      xtol=None,  # Parameter tolerance for convergence
                      gtol=None,) # Gradient tolerance for convergence

# roots.x contains the optimized LV surface values
# roots.x # this is best lv_surface that should be used in model, as it closely maps to implied_vols_from_model to 'implied_vol_target'

# Calculate total execution time
end_time_ = time.time()
durr_ = end_time_ - start_time_

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Use Machine learning to reduce calibration time
# MAGIC
# MAGIC Now we should try to reduce calibration time complexity using machine learning. The major bottleneck in above calibration is repeated calls to *model.implied_vol* function inside optimizer which is quite heavy due to Monte-Carlo Simulation.
# MAGIC
# MAGIC If somehow we can learn that function (which is mapping local_vol model parameter to implied vol), then we can use that function underneath the optimizer as replacement and it would be many fold faster!
# MAGIC
# MAGIC ## 2.1. Training Data Generation
# MAGIC First step is generate training dataset

# COMMAND ----------

# These are all the maturities we want to reprice our call options
option_maturities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
number_of_maturities = len(option_maturities)

# These are all the strikes (on each of above 'option_maturities') of those call options
option_strikes = np.array([0.95, 0.99, 1, 1.001, 1.05])
number_of_strikes = len(option_strikes)

# Number of Monte Carlo paths - increased for better accuracy in training data
num_samples = 4_000

# Number of training samples to generate
training_samples = 1_000
np.random.seed(0)  # For reproducibility

# COMMAND ----------

# Create schema for features and labels
# Format: maturity_strike for each combination
schema_ = []
for maturity in option_maturities:
  for strike in option_strikes:
    schema_.append(f'{maturity}_{strike}')

# COMMAND ----------

# Generate random local volatility surfaces as features
features = []
number_of_features = number_of_maturities * number_of_strikes
for i in range(number_of_features):
  features.append(np.random.uniform(0.0, 1.0, training_samples))
features = np.array(features).transpose()
#print("features", features.shape)


import time
start_time = time.time()

# Generate corresponding implied volatilities as labels
# This is the computationally expensive part we want to replace with ML
labels = []
for feature in features:
  # Reshape feature vector into LV surface
  lv_surface = np.split(feature, number_of_maturities)

  # Update model with current LV surface
  model.local_vol_fx = TimeSeries(jump_locations=option_maturities[:-1], values=lv_surface)

  # Calculate implied vols from the model
  implied_vols_from_model = model.implied_vol(option_maturities=option_maturities,
                                             option_strikes=option_strikes,
                                             num_samples=num_samples)

  # Store flattened implied vol surface as label
  labels.append(np.array(implied_vols_from_model).flatten())
labels = np.array(labels)
#print("labels", labels.shape)

# COMMAND ----------

# Validate that dimensions match
assert len(schema_) == features.shape[1] == labels.shape[1], 'There is a mismatch between the lengths of the schema and number of features/labels.'

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.2. Convert the features and labels into Koalas DataFrames
# MAGIC
# MAGIC Koalas provides a drop-in replacement for pandas. Commonly used by data scientists, pandas is a Python package that provides easy-to-use data structures and data analysis tools for the Python programming language. However, pandas does not scale out to big data. Koalas fills this gap by providing pandas equivalent APIs that work on Apache Spark. Koalas is useful not only for pandas users but also PySpark users, because Koalas supports many tasks that are difficult to do with PySpark, for example plotting data directly from a PySpark DataFrame.
# MAGIC
# MAGIC https://docs.databricks.com/languages/koalas.html

# COMMAND ----------

# Convert numpy arrays to Koalas (PySpark pandas) DataFrames
# This allows us to leverage Spark's distributed computing capabilities
features_ps = ps.DataFrame(features, columns=schema_).reset_index()
labels_ps = ps.DataFrame(labels, columns=schema_).reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Save the generated features and labels into Databricks Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Why use Databricks Feature Engineering?
# MAGIC Databricks Feature Engineering is fully integrated with Unity Catalog and other components of Databricks.
# MAGIC
# MAGIC - **Lineage**. When you create a feature table with Databricks Feature Engineering, the data sources used to create the feature table are saved and accessible. For each feature in a feature table, you can also access the models, notebooks, jobs, and endpoints that use the feature.
# MAGIC
# MAGIC - **Discoverability**. The Unity Catalog UI, accessible from the Databricks workspace, lets you browse and search for existing features.
# MAGIC
# MAGIC - **Integration** with model scoring and serving. When you use features from Databricks Feature Engineering to train a model, the model is packaged with feature metadata. When you use the model for batch scoring or online inference, it automatically retrieves features from Feature Engineering. The caller does not need to know about them or include logic to look up or join features to score new data. This makes model deployment and updates much easier.
# MAGIC
# MAGIC https://docs.databricks.com/machine-learning/feature-store/index.html

# COMMAND ----------

# Install the feature engineering client
%pip install databricks-feature-engineering

# COMMAND ----------

# Initialize the Feature Engineering client
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# Helper function to find a usable Unity Catalog
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
                    spark.sql(f"USE CATALOG {catalog_name}")
                    print(f"Successfully accessed Unity Catalog: {catalog_name}")
                    return catalog_name
                except Exception as e:
                    print(f"Cannot use catalog {catalog_name}: {e}")

        raise Exception("No usable Unity Catalog found. Feature Engineering requires Unity Catalog access.")
    except Exception as e:
        raise Exception(f"Error accessing Unity Catalog: {e}")

# Get Unity Catalog and create schema
catalog_name = get_unity_catalog()
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("CREATE SCHEMA IF NOT EXISTS implied_volatility")

# Set table names using the catalog
features_table_name = f"{catalog_name}.implied_volatility.features"
labels_table_name = f"{catalog_name}.implied_volatility.labels"

# COMMAND ----------

# Create the features table in Unity Catalog
# First convert to spark DataFrame
features_spark_df = features_ps.to_spark()

# Create feature table
try:
    fe.create_table(
        name=features_table_name,
        primary_keys=["index"],  # Use index as the primary key for lookups
        df=features_spark_df,
        description="Features set for Implied Volatility",
        timestamp_keys=None  # No time-based partitioning for this data
    )
    print(f"Successfully created feature table: {features_table_name}")
except Exception as e:
    print(f"Error creating feature table: {e}")
    raise e

# COMMAND ----------

# Create the labels table in Unity Catalog
# First convert to spark DataFrame
labels_spark_df = labels_ps.to_spark()

# Create feature table
try:
    fe.create_table(
        name=labels_table_name,
        primary_keys=["index"],  # Use index as the primary key for lookups
        df=labels_spark_df,
        description="Labels set for Implied Volatility",
        timestamp_keys=None  # No time-based partitioning for this data
    )
    print(f"Successfully created feature table: {labels_table_name}")
except Exception as e:
    print(f"Error creating feature table: {e}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3.1. Display the generated data and check the distribution

# COMMAND ----------

# Set the catalog and schema based on the Unity Catalog we found
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("USE SCHEMA implied_volatility")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.1.1. Visualize the generated data and explore
# MAGIC
# MAGIC Databricks Notebooks have built-in dashboarding capabilities (which we can observe below). We can very quickly visualize the features and labels that we just saved into the Databricks Feature Engineering tables. Below we have the scattered plot of various strike levels for the same maturity (of the generated data). The chart even offers a LOESS regression that can give us even more information about the distribution of the residuals of the generated data.

# COMMAND ----------
# Display the labels table using the appropriate table name
# This will automatically generate visualizations in the notebook
display(spark.sql(f'SELECT * FROM {catalog_name}.implied_volatility.labels'))

# COMMAND ----------
# MAGIC %md
# MAGIC
# MAGIC ### 3.1.2. Data Profiling
# MAGIC
# MAGIC Databricks Notebooks have built-in data profiling features. In the cell below, we can observe a lot of statistical information for the newly generated features and labels, without having to use third-party tools or write additional code.

# COMMAND ----------
# Display the labels table for profiling using the appropriate table name
# The Data Profile tab will show statistics for each column
display(spark.sql(f'SELECT * FROM {catalog_name}.implied_volatility.labels'))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Step 4: Check the generated data for statistical issues using R
# MAGIC
# MAGIC We will use R libraries to automatically test the newly generated data for heteroskedasticity.

# COMMAND ----------

# Create a view for R to use
# This makes the feature data accessible to R code
features_spark_df.createOrReplaceTempView('IVfeatures_view')

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Load SparkR library to interact with Spark data
# MAGIC library(SparkR)
# MAGIC
# MAGIC # Refresh the view to ensure latest data is available
# MAGIC sql("REFRESH TABLE IVfeatures_view")
# MAGIC
# MAGIC # Load the features data into R dataframe
# MAGIC features_df_r <- sql("SELECT * FROM IVfeatures_view")

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Display the first few rows of the data in R
# MAGIC head(features_df_r)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Step 5: Train ML model for each of the labels
# MAGIC
# MAGIC See **Implied Volatility Prediction - 2. ML**