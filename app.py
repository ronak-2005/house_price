from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="House Price Prediction")
MAX_ROWS = 10000  # Prevents too large uploads

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths (Render vs Local)
MODEL_PATH_RENDER = "/mnt/data/house_price.pkl"
FEATURES_PATH_RENDER = "/mnt/data/features.pkl"
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "house_price.pkl")
FEATURES_PATH_LOCAL = os.path.join(BASE_DIR, "features.pkl")

# Load model + features
if os.path.exists(MODEL_PATH_RENDER) and os.path.exists(FEATURES_PATH_RENDER):
    model_path, features_path = MODEL_PATH_RENDER, FEATURES_PATH_RENDER
elif os.path.exists(MODEL_PATH_LOCAL) and os.path.exists(FEATURES_PATH_LOCAL):
    model_path, features_path = MODEL_PATH_LOCAL, FEATURES_PATH_LOCAL
else:
    raise FileNotFoundError("Model or features.pkl not found.")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Serve static frontend (modern UI with Tailwind)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve modern index.html frontend."""
    with open(os.path.join(BASE_DIR, "static/index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Handle CSV upload and return predictions as JSON."""
    df = pd.read_csv(file.file, nrows=MAX_ROWS).replace("NA", np.nan)

    # Drop Id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Align with model features
    df = df[[col for col in df.columns if col in feature_columns]]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]

    # Predict
    preds = model.predict(df)

    results = []
    for i, p in enumerate(preds):
        results.append({
            "Index": int(i),
            "Predicted Price": round(float(p), 2),
            "Formatted": f"${p:,.2f}"
        })

    return JSONResponse(content={"predictions": results})
