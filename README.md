# NIFTY Intraday Direction Prediction – Machine Learning Project

This project implements a complete machine learning pipeline to predict whether the next intraday NIFTY candle will close higher or lower. The workflow includes data loading, feature engineering, model training, performance evaluation, signal generation, PnL calculation based on the project rule, and exporting processed datasets and charts.

---

## 1. Project Objectives

- Build a well-structured Python project and push it to a GitHub repository.  
- Train and compare at least two machine learning models for candle direction classification.  
- Select the best-performing model based on accuracy.  
- Generate model-based trading signals and compute cumulative PnL as per the assignment rule.  
- Export processed datasets and visualizations.  

---

## 2. Requirements and Installation

Ensure Python 3.8 or above is installed.

Install all dependencies manually:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost


Or use a `requirements.txt` (if provided):


---

## 3. How to Run the Project

1. Place the raw dataset (`nifty50_ticks.csv`) inside the `data/` folder.
2. Open a terminal in the project root directory.
3. Run the pipeline using:


The script will automatically:

- Load and preprocess the intraday data  
- Generate technical indicators and engineered features  
- Perform time-based train/test split  
- Train Random Forest and XGBoost  
- Evaluate using Accuracy, Precision, and Recall  
- Select the best model  
- Generate predictions and project-rule PnL  
- Save plots inside `outputs/`  
- Save `train_data.csv` and `test_data.csv` inside `data/`

---

## 4. Model Performance Summary

Both Random Forest and XGBoost were trained and evaluated.  
XGBoost performed better overall due to its ability to model non-linear relationships and reduce bias using sequential boosting. It achieved higher accuracy and more stable metrics on the time-based test set, and was therefore selected as the final model.

---

## 5. Output Files Generated

### Data Files
- `data/train_data.csv`  
- `data/test_data.csv`  
These files include all engineered columns and the target label.

### Visualization Files
- `outputs/confusion_matrix.png`  
- `outputs/feature_importance.png`  
- `outputs/pnl_curve.png`  
- `outputs/pnl_curve_prob.png`  

---

