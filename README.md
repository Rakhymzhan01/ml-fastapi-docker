# ML FastAPI Docker — Iris Classifier

## Description
REST API for Iris flower classification using Random Forest, served with FastAPI and containerized with Docker.

## Run locally

```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

## Run with Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

## Endpoints

| Method | Endpoint   | Description          |
|--------|------------|----------------------|
| GET    | /          | Health check         |
| POST   | /predict   | Predict iris species |
| GET    | /docs      | Swagger UI           |

## Example Request

```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Example Response

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "confidence": 0.97
}
```