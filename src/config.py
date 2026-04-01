import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATHS = {
    "churn": os.path.join(BASE_DIR, "data/customer_churn.csv"),
    "sales": os.path.join(BASE_DIR, "data/sales_data.csv"),
    "house": os.path.join(BASE_DIR, "data/house_prices.csv")
}

MODEL_PATH = os.path.join(BASE_DIR, "models")