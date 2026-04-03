# Technical Report

## Objective

The goal of this project is to develop a modular machine learning analytics suite capable of solving three business prediction tasks:

1. Customer churn classification
2. Sales revenue forecasting
3. House price estimation

---

## Dataset Description

### Customer Churn Dataset

Target variable:
Churn

Features include:

- tenure
- contract type
- internet service
- billing information

Task type:
Binary classification

---

### Sales Dataset

Target variable:
Total_Sales

Features include:

- product
- quantity
- region
- weekday
- month

Task type:
Regression

---

### House Prices Dataset

Target variable:
Price

Features include:

- area
- bedrooms
- bathrooms
- location
- property type

Task type:
Regression

---

## Preprocessing Pipeline

Steps applied:

Missing value handling

Categorical encoding using one-hot encoding

Feature alignment for deployment compatibility

---

## Model Selection

Random Forest chosen due to:

robust performance on tabular data  
resistance to overfitting  
automatic feature importance extraction  
minimal scaling requirements  

---

## Evaluation Metrics

Classification:

Accuracy  
Precision  
Recall  
F1-score  

Regression:

MAE  
MSE  
R² score  

---

## Deployment Architecture

Pipeline structure:

data loader → preprocessing → encoding → training → evaluation → model persistence

Feature schemas saved separately for safe inference alignment.

---

## Result Summary

Random Forest models achieved strong baseline predictive performance across all three datasets with interpretable feature importance outputs supporting business explainability.
