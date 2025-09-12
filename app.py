from fastapi import FastAPI, UploadFile, File, Request
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

# Load model
if os.path.exists(MODEL_PATH_RENDER) and os.path.exists(FEATURES_PATH_RENDER):
    model_path, features_path = MODEL_PATH_RENDER, FEATURES_PATH_RENDER
elif os.path.exists(MODEL_PATH_LOCAL) and os.path.exists(FEATURES_PATH_LOCAL):
    model_path, features_path = MODEL_PATH_LOCAL, FEATURES_PATH_LOCAL
else:
    raise FileNotFoundError(
        f"Model file not found in either Render ({MODEL_PATH_RENDER}) "
        f"or Local ({MODEL_PATH_LOCAL})"
    )

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Upload form for CSV."""
    html_content = """
    <html>
    <head>
        <title>House Price Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
            form { margin: 20px auto; }
            input[type=file] { margin-bottom: 10px; }
            button { padding: 8px 16px; background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h2>Upload CSV for House Price Prediction</h2>
        <form action="/predict_csv" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(file: UploadFile = File(...)):
    """Predict house prices from uploaded CSV and return as HTML table."""
    # Read CSV (limit rows for safety)
    df = pd.read_csv(file.file, nrows=MAX_ROWS).replace("NA", np.nan)

    # Drop Id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Keep only model features
    df = df[[col for col in df.columns if col in feature_columns]]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]

    # Predict
    preds = model.predict(df)

    # Format results
    results = [
        {
            "index": i,
            "predicted_price": float(p),
            "predicted_price_formatted": f"${p:,.2f}"
        }
        for i, p in enumerate(preds)
    ]

    # Build HTML table dynamically
    table_html = f"""
    <html>
    <head>
        <title>House Price Predictions</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 80%; margin: auto; }}
            th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            h2 {{ text-align: center; }}
            .back {{ display: block; text-align: center; margin-top: 20px; }}
            .btn {{ padding: 6px 12px; background: #007BFF; color: white; border: none; border-radius: 4px; text-decoration: none; }}
            .btn:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <h2>House Price Predictions ({len(df)} rows)</h2>
        <table>
            <thead>
                <tr><th>Index</th><th>Predicted Price</th><th>Formatted</th></tr>
            </thead>
            <tbody>
    """

    for row in results:
        table_html += f"""
            <tr>
                <td>{row['index']}</td>
                <td>{row['predicted_price']}</td>
                <td>{row['predicted_price_formatted']}</td>
            </tr>
        """

    table_html += """
            </tbody>
        </table>
        <div class="back">
            <a class="btn" href="/">Upload Another CSV</a>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=table_html)
