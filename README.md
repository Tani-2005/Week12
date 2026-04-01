# Business Analytics & Machine Learning Suite

An end-to-end Business Analytics platform that combines predictive modeling, feature engineering, and interactive visualization dashboards to support decision-making across customer retention, sales forecasting, and property valuation.

## Project Overview

This project implements three machine learning pipelines:

| Module | Task | Model |
|-------|------|------|
| Customer Churn | Classification | RandomForestClassifier |
| Sales Forecasting | Regression | RandomForestRegressor |
| House Price Prediction | Regression | RandomForestRegressor |

The system includes:

- modular ML pipeline architecture
- automated training script
- feature alignment for deployment safety
- Streamlit analytics dashboard
- feature importance explainability
- reusable preprocessing layer

---

## Project Structure
```
Week-12/
│
├── data/
├── models/
├── notebooks/
├── src/
├── deployment/
├── reports/
├── presentation/
├── train_all_models.py
├── requirements.txt
└── README.md
```

---

## Key Features

### Modular ML Pipeline

Reusable architecture:

- preprocessing
- feature engineering
- evaluation
- training orchestration
- model factory abstraction

### Automated Model Training

Run: python -m src.train_all_models

Generates:
```
models/
├── churn_model.pkl
├── churn_columns.pkl
├── sales_model.pkl
├── sales_columns.pkl
├── house_model.pkl
└── house_columns.pkl
```

---

### Interactive Dashboard

Launch locally:
```
cd deployment
streamlit run app.py
```

Includes:

- churn risk gauge
- revenue forecasting charts
- property valuation visualizations
- feature importance explanations

---

## Machine Learning Workflow
data
↓
cleaning
↓
encoding
↓
feature alignment
↓
training
↓
evaluation
↓
deployment-ready models

---

## Technologies Used

Python  
Pandas  
Scikit-learn  
Plotly  
Streamlit  

---

## Example Use Cases

Customer retention optimization  
Revenue forecasting support  
Real estate pricing insights  
Business KPI monitoring dashboards  

---

## Future Improvements

Hyperparameter tuning with GridSearchCV  
Model version tracking  
Cloud deployment (Streamlit Cloud / AWS)  
REST API inference endpoint  

---
