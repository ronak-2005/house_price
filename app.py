from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import numpy as np


model = joblib.load("house_price.pkl")
feature_columns = joblib.load("features.pkl")

app = FastAPI(title="House Price Prediction")


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_csv")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = df.replace("NA", np.nan)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    df = df[[col for col in df.columns if col in feature_columns]]

    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_columns]

    preds = model.predict(df)

    results = pd.DataFrame({
        "Index": range(len(preds)),
        "PredictedPrice": preds,
        "PredictedPriceFormatted": [f"${p:,.2f}" for p in preds]
    })

    preview = results.head(10).to_html(index=False)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "table": preview,
        "total_predictions": len(preds)
    })
