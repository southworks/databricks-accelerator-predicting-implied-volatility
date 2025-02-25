# Databricks Notebooks for Implied Volatility Prediction

This accelerator includes three main notebooks that work together to demonstrate how to implement an academic paper on deep learning volatility using Databricks. Here's what each notebook does:

## 01_introduction.py

This notebook provides an overview of the accelerator and explains the problem we're trying to solve.

**Key points:**
- Introduces the concept of a quantitative researcher's workflow
- Explains the "Deep Learning Volatility" paper by Horvath et al. (2019)
- The paper's goal is to build neural networks that can approximate complex pricing functions for financial derivatives
- Details why Databricks Lakehouse is ideal for this type of work:
  - Scale: Handles computationally intensive simulations efficiently
  - DataOps: Stores generated features in Delta tables for reuse
  - Collaboration: Enables use of both Python and R in the same workflow
  - MLOps: Tracks model experiments with MLflow
  - Productionalization: Orchestrates workflows for reliable deployment
- Includes a high-level architecture diagram showing the entire solution

This notebook serves as an introduction and doesn't contain computational code.

## 02_create_features.py

This notebook implements the core quantitative finance model and generates the training data needed for machine learning.

**Key points:**
- Defines a stochastic model for financial derivatives pricing:
  - Implements a 4-dimensional Ito process model combining:
    - Vasicek interest rate models for two currencies
    - Local volatility model for FX
    - Lognormal FX rate process
  - Uses TensorFlow Quant Finance for financial mathematics
- Demonstrates traditional calibration of the model:
  - Shows how model parameters are normally fitted to market data
  - Highlights the computational bottleneck in this approach
- Generates synthetic training data:
  - Creates random local volatility surfaces as inputs
  - Calculates corresponding implied volatilities using Monte Carlo simulation
  - This represents the expensive computation we want to replace with machine learning
- Saves generated data to Databricks Feature Engineering:
  - Stores features (local volatility surfaces) and labels (implied volatilities)
  - Makes data available for model training with proper lineage tracking
- Visualizes and profiles the generated data
- Uses R to check for statistical issues like heteroskedasticity

This notebook is computationally intensive, especially in the Monte Carlo simulation step.

## 03_train_models.py

This notebook trains machine learning models to replace the expensive pricing function.

**Key points:**
- Loads the generated data from Databricks Feature Engineering
- Prepares data for ML model training:
  - Selects one implied volatility (0.05_0.95) as the target for simplicity
  - Creates preprocessing pipeline for numerical features
  - Splits data into train (60%), validation (20%), and test (20%) sets
- Trains an XGBoost regression model:
  - Uses carefully selected hyperparameters
  - Implements early stopping for best performance
- Logs model performance metrics to MLflow:
  - R² score, mean absolute error, mean squared error, RMSE
  - Tracks both validation and test performance
- Optional SHAP feature importance analysis:
  - Can be enabled to explain which inputs most affect the predictions
  - Helps understand the relationship between local volatility and implied volatility

The trained model can replace the slow Monte Carlo pricing function, making calibration significantly faster.

## How the Notebooks Work Together

1. **Notebook 01** explains the problem and approach
2. **Notebook 02** creates the financial model and generates training data
3. **Notebook 03** trains ML models to approximate the expensive pricing function

Together, these notebooks demonstrate how to use machine learning to accelerate quantitative finance workflows, which is a major real-world use case in financial institutions.

## Running the Accelerator

The notebooks can be run individually for exploration, or as a complete workflow using Databricks Workflows. The included RUNME.py script sets up the necessary compute and creates a workflow DAG that runs the notebooks in the correct sequence.