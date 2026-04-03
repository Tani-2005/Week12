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
│   ├── customer_churn.csv
│   ├── sales_data.csv
│   └── house_prices.csv
├── models/
│  ├── churn_model.pkl
│  ├── churn_columns.pkl
│  ├── sales_model.pkl
│  ├── sales_columns.pkl
│  ├── house_model.pkl
│  └── house_columns.pkl
├── notebooks/
│   ├── churn_analysis.csv
│   ├── sales_analysis.csv
│   └── house_price_analysis.csv
├── src/
|   ├── __pycache__/
│   ├── __init__.py
│   ├── config.py
|   ├── data_loader.py
|   ├── evaluate.py
|   ├── feature_engineering.py
|   ├── model_factory.py
|   ├── preprocessing.py
|   ├── train_all_models.py
|   ├── train_pipeline.py
│   └── utils.csv
├── deployment/
│   └── app.py
├── reports/
│   ├── business_report.md
│   └── technical_report.md
├── presentation/
│   └── business_analytics_ml_suite_presentaion.pptx
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
