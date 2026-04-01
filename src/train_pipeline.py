from sklearn.model_selection import train_test_split

from src.model_factory import get_model
from src.evaluate import evaluate_classification
from src.evaluate import evaluate_regression


def train_model(X, y, task):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = get_model(task)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if task == "classification":
        return model, evaluate_classification(y_test, predictions)

    return model, evaluate_regression(y_test, predictions)