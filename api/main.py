# main.py
from fastapi import FastAPI
import pandas as pd
import mlflow
import os

app = FastAPI()

os.environ['MLFLOW_TRACKING_URI'] = "http://host.docker.internal:5000"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

model = mlflow.pyfunc.load_model('models:/movie-classification@prod')

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

@app.get("/")
async def predict(data: dict):
    print('Welcome into model')