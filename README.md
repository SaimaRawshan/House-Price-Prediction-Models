# House Price Prediction

Predicting residential house sale prices using the Ames Housing dataset by training and comparing seven regression models across linear, tree-based, and neural network approaches.

## Overview

This project explores the [Ames Housing dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) to predict `SalePrice` based on 79 explanatory features describing residential properties (size, quality, location, condition, etc.). The workflow covers data cleaning, exploratory data analysis, feature transformation, and training/evaluating multiple regression models.

## Dataset

- **Target variable:** `SalePrice`
- **Features:** 79 variables covering lot characteristics, building type, quality ratings, square footage, garage/basement details, and sale conditions.

## Project Workflow

### 1. Data Cleaning
- Explored dataset structure, summary statistics, and target variable distribution (`SalePrice` skewness and kurtosis).
- Identified missing values and dropped columns with more than 30% missing data (`Alley`, `MasVnrType`, `FireplaceQu`, `PoolQC`, `Fence`, `MiscFeature`).
- Dropped remaining rows with null values and verified completeness using a null-value heatmap.
- Removed the `Utilities` column after finding it had zero correlation with `SalePrice` and only one unique value.
- Label-encoded all categorical (object/category) columns for model compatibility.

### 2. Exploratory Data Analysis
- Generated scatter plots of all numeric features against `SalePrice`.
- Plotted histograms to inspect feature distributions.
- Built correlation heatmaps (full and filtered to strong correlations ≥ 0.7) to identify influential features.
- Computed skewness for 18 numeric, square-footage-based features (`LotArea`, `GrLivArea`, `TotalBsmtSF`, etc.) and applied a `log1p` transformation to those with skewness > 1, reducing skew and improving downstream model performance.

### 3. Modeling
Trained and evaluated seven regression models on an 80/20 train-test split with standardized features (`StandardScaler`):

| Model | Library |
|---|---|
| Linear Regression | scikit-learn |
| K-Nearest Neighbors (KNN) | scikit-learn |
| Random Forest Regressor | scikit-learn |
| Neural Network (MLP) | scikit-learn |
| Deep Neural Network | TensorFlow / Keras |
| 1D Convolutional Neural Network | TensorFlow / Keras |
| XGBoost | xgboost |

### 4. Evaluation Metrics
Each model was assessed using:
- MSE, RMSE, MAE, Median AE, MAPE
- Mean Bias Error (MBE)
- R², Adjusted R², Explained Variance

### 5. Visualization
- Actual vs. Predicted scatter plots for each model.
- Dual-axis charts comparing error metrics (bar) against R² / Adjusted R² / Explained Variance (line) across all models.
- Bar chart of Mean Bias Error (MBE) per model.
- Final consolidated metrics table across all seven models.

## Tech Stack
- **Language:** Python
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn, TensorFlow/Keras, XGBoost
