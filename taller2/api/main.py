from fastapi import FastAPI
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    X: List[float]

app = FastAPI()

MODEL_PATH = "./sharedmodels/penguin_model.pkl"
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(data: InputData):
    columnas = model.feature_names_in_
    df = pd.DataFrame([data.X], columns=columnas)
    prediction = model.predict(df)
    return {"prediccion": str(prediction[0])}
