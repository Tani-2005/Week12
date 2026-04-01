import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_pickle
from src.preprocessing import encode_categorical, align_features
from src.config import MODEL_PATH


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Business Analytics ML Suite",
    layout="wide"
)


# ==============================
# LOAD MODELS + FEATURES
# ==============================

@st.cache_resource
def load_assets():

    churn_model = load_pickle(f"{MODEL_PATH}/churn_model.pkl")
    churn_cols = load_pickle(f"{MODEL_PATH}/churn_columns.pkl")

    sales_model = load_pickle(f"{MODEL_PATH}/sales_model.pkl")
    sales_cols = load_pickle(f"{MODEL_PATH}/sales_columns.pkl")

    house_model = load_pickle(f"{MODEL_PATH}/house_model.pkl")
    house_cols = load_pickle(f"{MODEL_PATH}/house_columns.pkl")

    return (
        churn_model, churn_cols,
        sales_model, sales_cols,
        house_model, house_cols
    )


(
    churn_model, churn_cols,
    sales_model, sales_cols,
    house_model, house_cols
) = load_assets()


# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.title("📊 ML Business Dashboard")

page = st.sidebar.radio(
    "Select Module",
    [
        "Customer Churn Prediction",
        "Sales Forecasting",
        "House Price Prediction"
    ]
)


# =====================================================
# FEATURE IMPORTANCE FUNCTION
# =====================================================

def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# CUSTOMER CHURN MODULE
# =====================================================

if page == "Customer Churn Prediction":

    st.title("📉 Customer Churn Risk Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider("Tenure", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 500.0, 75.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    with col2:

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    if st.button("Predict Churn"):

        input_df = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
            "Contract": [contract],
            "InternetService": [internet]
        })

        input_df = encode_categorical(input_df)
        input_df = align_features(input_df, churn_cols)

        prob = churn_model.predict_proba(input_df)[0][1]
        prediction = churn_model.predict(input_df)[0]

        st.metric("Churn Probability", f"{prob:.2%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        st.plotly_chart(fig)

        if prediction == 1:
            st.error("⚠️ High churn risk")
        else:
            st.success("✅ Low churn risk")

    st.subheader("Model Explanation")

    plot_feature_importance(churn_model, churn_cols)


# =====================================================
# SALES MODULE
# =====================================================

elif page == "Sales Forecasting":

    st.title("📊 Sales Forecast Dashboard")

    quantity = st.number_input("Quantity", 1, 100, 5)
    price = st.number_input("Price", 1.0, 10000.0, 500.0)

    region = st.selectbox(
        "Region",
        ["North", "South", "East", "West"]
    )

    month = st.slider("Month", 1, 12, 6)
    weekday = st.slider("Weekday", 0, 6, 2)

    if st.button("Predict Sales"):

        input_df = pd.DataFrame({
            "Quantity": [quantity],
            "Price": [price],
            "Region": [region],
            "Month": [month],
            "Weekday": [weekday]
        })

        input_df = encode_categorical(input_df)
        input_df = align_features(input_df, sales_cols)

        prediction = sales_model.predict(input_df)[0]

        st.metric("Predicted Sales", f"₹ {prediction:,.2f}")

        fig = px.bar(
            x=["Predicted Sales"],
            y=[prediction],
            title="Predicted Revenue"
        )

        st.plotly_chart(fig)

    st.subheader("Key Drivers")

    plot_feature_importance(sales_model, sales_cols)


# =====================================================
# HOUSE PRICE MODULE
# =====================================================

elif page == "House Price Prediction":

    st.title("🏠 Property Valuation Dashboard")

    area = st.number_input("Area", 200, 10000, 1500)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    age = st.slider("Property Age", 0, 50, 5)

    location = st.selectbox(
        "Location",
        ["Urban", "Suburban", "Rural"]
    )

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Villa", "Independent House"]
    )

    if st.button("Predict Price"):

        input_df = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Age": [age],
            "Location": [location],
            "Property_Type": [property_type]
        })

        input_df = encode_categorical(input_df)
        input_df = align_features(input_df, house_cols)

        prediction = house_model.predict(input_df)[0]

        st.metric("Estimated Price", f"₹ {prediction:,.2f}")

        fig = px.bar(
            x=["Predicted Price"],
            y=[prediction],
            title="Property Valuation"
        )

        st.plotly_chart(fig)

    st.subheader("Key Price Drivers")

    plot_feature_importance(house_model, house_cols)