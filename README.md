# Shipment Delay Predictor

A machine learning project that predicts whether a shipment will be delivered late based on order and shipping features.

## Overview

This project builds a binary classifier to predict `Late_delivery_risk` using an e-commerce shipment dataset. It follows a full ML pipeline: data cleaning, exploratory data analysis, feature engineering, and model training with ensemble methods.

## Dataset

The dataset contains **180,519 rows** and **53 columns** covering order details, shipping information, customer data, and product attributes. After feature selection, **18 key columns** are retained for modeling, including:

- **Order details**: Order Item Quantity, Sales, Discount, Profit Ratio, Total
- **Shipping info**: Shipping Mode, Days for shipping (real vs. scheduled)
- **Financials**: Benefit per order, Sales per customer, Product Price
- **Target**: `Late_delivery_risk` (0 = Late, 1 = Not Late)

## Pipeline

1. **Preprocessing** — Column selection, null/duplicate checks
2. **EDA** — Distribution plots for transaction types, shipping modes, and delivery risk
3. **Feature Engineering**
   - Date parsing (extracting year, month, day, day of week, hour from order/shipping dates)
   - One-hot encoding of categorical features (`Type`, `Shipping Mode`)
   - Outlier handling via Z-score capping (threshold = 3)
   - Min-max normalization of float features
4. **Modeling** — Stacking ensemble combining:
   - Random Forest (100 trees)
   - XGBoost (100 rounds, lr=0.1)
   - Logistic Regression as the meta-learner (5-fold CV)
5. **Evaluation** — Accuracy, Log Loss, and classification report

## Files

| File | Description |
|------|-------------|
| `secondTry.ipynb` | Main notebook with the full pipeline |
| `data.csv` | Raw dataset |
| `cleaned.csv` | Cleaned dataset |

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, scipy, joblib
