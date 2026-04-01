from src.config import DATA_PATHS, MODEL_PATH
from src.data_loader import load_data
from src.preprocessing import handle_missing, encode_categorical
from src.train_pipeline import train_model
from src.utils import save_pickle

import os


def run_training(dataset, target, task, model_name, column_name):

    df = load_data(DATA_PATHS[dataset])

    df = handle_missing(df)
    df = encode_categorical(df)

    X = df.drop(target, axis=1)
    y = df[target]

    model, metrics = train_model(X, y, task)

    save_pickle(model, os.path.join(MODEL_PATH, model_name))

    save_pickle(
        list(X.columns),
        os.path.join(MODEL_PATH, column_name)
    )

    print(f"{model_name} trained successfully")
    print(metrics)


run_training(
    "churn",
    "Churn",
    "classification",
    "churn_model.pkl",
    "churn_columns.pkl"
)

run_training(
    "sales",
    "Total_Sales",
    "regression",
    "sales_model.pkl",
    "sales_columns.pkl"
)

run_training(
    "house",
    "Price",
    "regression",
    "house_model.pkl",
    "house_columns.pkl"
)