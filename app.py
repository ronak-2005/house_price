from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
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
    raise FileNotFoundError("Model or features.pkl not found in Render or Local paths.")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)


def render_page(table_html: str = "") -> str:
    """Render upload form with optional results table."""
    return f"""
    <html>
    <head>
        <title>House Price Predictor</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
            form {{ margin: 20px auto; }}
            input[type=file] {{ margin-bottom: 10px; }}
            button {{ padding: 8px 16px; background: #007BFF; color: white;
                      border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            table {{ border-collapse: collapse; width: 80%; margin: 20px auto; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
        </style>
    </head>
    <body>
        <h2>Upload CSV for House Price Prediction</h2>
        <form action="/" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit">Predict</button>
        </form>
        {table_html}
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def home():
    """Initial upload form page."""
    return HTMLResponse(content=render_page())


@app.post("/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    """Upload CSV, predict, and render results below form."""
    # Read CSV (limit rows for safety)
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

    # Build table rows
    rows_html = ""
    for i, p in enumerate(preds):
        rows_html += f"""
            <tr>
                <td>{i}</td>
                <td>{float(p):.2f}</td>
                <td>${p:,.2f}</td>
            </tr>
        """

    # Full table
    table_html = f"""
        <h2>House Price Predictions ({len(df)} rows)</h2>
        <p>⚠️ Processed only first {MAX_ROWS} rows for performance</p>
        <table>
            <thead>
                <tr><th>Index</th><th>Predicted Price</th><th>Formatted</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    """

    return HTMLResponse(content=render_page(table_html))
