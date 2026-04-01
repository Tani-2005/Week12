from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def get_model(task):

    if task == "classification":
        return RandomForestClassifier(random_state=42)

    elif task == "regression":
        return RandomForestRegressor(random_state=42)

    else:
        raise ValueError("Unsupported task type")