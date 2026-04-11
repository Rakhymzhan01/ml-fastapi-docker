from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Разбиваем на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцениваем
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# Сохраняем модель
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")