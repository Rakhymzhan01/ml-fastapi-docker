from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris ML API")

# Загружаем модель при старте
model = joblib.load("model.joblib")

# Названия классов
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Схема входных данных
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }]
        }
    }

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max()

    return {
        "prediction": int(prediction),
        "class_name": CLASS_NAMES[prediction],
        "confidence": round(float(probability), 4)
    }