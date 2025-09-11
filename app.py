from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="House Price Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")

# Load model + features
model = joblib.load(os.path.join(BASE_DIR, "house_price.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form_house.html", {"request": request})

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file).replace("NA", np.nan)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df = df[[col for col in df.columns if col in feature_columns]]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]

    preds = model.predict(df)

    results = [
        {
            "index": i,
            "predicted_price": float(p),
            "predicted_price_formatted": f"${p:,.2f}"
        }
        for i, p in enumerate(preds)
    ]

    return {
        "predictions": results,
        "rows_processed": len(df)
    }
