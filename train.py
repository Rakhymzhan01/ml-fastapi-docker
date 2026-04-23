import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# MLflow setup — store locally in ./mlruns (no server required)
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("iris-classification")

# Hyperparameters
N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Разбиваем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)

    # Обучаем модель
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Оцениваем
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")

    # Сохраняем модель локально
    joblib.dump(model, "model.joblib")

    # Регистрируем модель в MLflow Model Registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="iris-random-forest",
    )

print("Model saved to model.joblib and registered in MLflow")
