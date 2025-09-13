from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os

app = FastAPI(title="House Price Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
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
    raise FileNotFoundError("house_price.pkl or features.pkl not found.")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form_house.html", {"request": request, "predictions": None})


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file, nrows=MAX_ROWS)
    df_test = df[feature_columns]  # Ensure correct order of features

    preds = model.predict(df_test)

    # Prepare top 10 results
    results = []
    for i, pred in enumerate(preds):
        results.append({
            "index": i,
            "Predicted Price": round(float(pred), 2)
        })

    top_10 = results[:10]

    return templates.TemplateResponse("form_house.html", {"request": request, "predictions": top_10})
