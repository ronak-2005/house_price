from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="House Price Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH_RENDER = "/mnt/data/house_price.pkl"
FEATURES_PATH_RENDER = "/mnt/data/features.pkl"
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "house_price.pkl")
FEATURES_PATH_LOCAL = os.path.join(BASE_DIR, "features.pkl")

# Load model + features
try:
    if os.path.exists(MODEL_PATH_RENDER) and os.path.exists(FEATURES_PATH_RENDER):
        model_path, features_path = MODEL_PATH_RENDER, FEATURES_PATH_RENDER
        logger.info("Using Render paths for model files")
    elif os.path.exists(MODEL_PATH_LOCAL) and os.path.exists(FEATURES_PATH_LOCAL):
        model_path, features_path = MODEL_PATH_LOCAL, FEATURES_PATH_LOCAL
        logger.info("Using local paths for model files")
    else:
        raise FileNotFoundError("house_price.pkl or features.pkl not found.")

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    logger.info(f"Model loaded successfully. Features: {len(feature_columns)}")
    logger.info(f"Expected features: {feature_columns}")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    try:
        return templates.TemplateResponse("form_house.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        # Fallback to simple HTML
        return HTMLResponse("""
        <!DOCTYPE html>
        <html><head><title>House Price Predictor</title></head>
        <body>
        <h1>House Price Predictor</h1>
        <form action="/predict_csv" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Upload & Predict</button>
        </form>
        </body></html>
        """)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Handle CSV upload and return predictions as JSON"""
    
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV
        try:
            df = pd.read_csv(file.file, nrows=MAX_ROWS)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            logger.info(f"CSV columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Check if required features are present
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_features)}"
            )
        
        # Prepare test data (ensure correct order of features)
        try:
            df_test = df[feature_columns]
            logger.info(f"Test data prepared. Shape: {df_test.shape}")
        except KeyError as e:
            logger.error(f"KeyError when selecting features: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error selecting features: {str(e)}"
            )
        
        # Make predictions
        try:
            preds = model.predict(df_test)
            logger.info(f"Predictions made successfully. Count: {len(preds)}")
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
        
        # Prepare results
        results = []
        for i, pred in enumerate(preds):
            results.append({
                "index": i,
                "Predicted Price": round(float(pred), 2)
            })
        
        # Return top 10 results
        top_10 = results[:10]
        logger.info(f"Returning {len(top_10)} predictions")
        
        return JSONResponse({
            "status": "success",
            "predictions": top_10,
            "total_predictions": len(results),
            "message": f"Successfully processed {len(results)} rows"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_columns) if feature_columns else 0
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)}
    )

@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"error": "Method not allowed", "method": request.method, "path": str(request.url.path)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)