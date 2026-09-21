"""
Production FastAPI REST Interface for NSE Intraday Stock Movement Prediction.

Features:
- POST /predict: Stateless live prediction using locked production model.
- POST /paper/predict: Stateful paper prediction creating persistent record.
- POST /paper/resolve: Resolves all due predictions against market data.
- POST /paper/resolve/{prediction_id}: Resolves specific prediction.
- GET /paper/predictions: Filterable prediction history.
- GET /paper/predictions/{prediction_id}: Single prediction detail.
- GET /paper/performance: Forward paper trading metrics and benchmarks.
- GET /health: System health and model artifact validation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_movement
from src.paper_trading import (
    create_paper_prediction,
    resolve_prediction,
    resolve_all_pending,
    list_predictions,
    get_prediction,
    validate_prediction_inputs
)
from src.performance import generate_paper_performance_summary

app = FastAPI(
    title="NSE Intraday Stock Movement Prediction API",
    description="Production-grade API for stock movement classification (UP, DOWN, STABLE) and paper trading evaluation.",
    version="1.0.0"
)

# Enable CORS for UI dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Request & Response Schemas ---

class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="360ONE", description="NSE equity ticker symbol")
    reference_timestamp: str = Field(..., example="2026-09-10 10:30", description="Reference timestamp (IST)")
    target_timestamp: str = Field(..., example="2026-09-15 10:30", description="Target timestamp (IST)")
    model_type: Optional[str] = Field("pooled", example="pooled", description="'pooled' or 'h1'")

class PredictionResponse(BaseModel):
    symbol: str
    reference_timestamp: str
    target_timestamp: str
    reference_price: float
    observed_sessions_ahead: int
    calendar_days_ahead: int
    predicted_class: str
    probability_down: float
    probability_stable: float
    probability_up: float
    model_name: str
    model_version: str

class PaperPredictionResponse(BaseModel):
    prediction_id: str
    symbol: str
    reference_timestamp: str
    target_timestamp: str
    reference_price: float
    observed_sessions_ahead: int
    calendar_days_ahead: int
    predicted_class: str
    probability_down: float
    probability_stable: float
    probability_up: float
    model_name: str
    model_version: str
    status: str
    status_reason: Optional[str] = None
    created_at: Any
    actual_price: Optional[float] = None
    future_return_pct: Optional[float] = None
    actual_class: Optional[str] = None
    is_resolved: bool
    resolved_at: Optional[Any] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    canonical_source: str
    supported_horizons: str

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """System health check and loaded model diagnostics."""
    return {
        "status": "HEALTHY",
        "model_loaded": True,
        "model_name": "Logistic Regression (L2, C=1.0, class_weight='balanced')",
        "model_version": "1.0.0",
        "canonical_source": "yahoo",
        "supported_horizons": "1 to 7 observed trading sessions ahead"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Stateless production prediction endpoint.
    Guarantees: Zero target OHLCV is queried; prediction strictly uses data <= reference timestamp.
    """
    try:
        # Strict validation
        val = validate_prediction_inputs(request.symbol, request.reference_timestamp, request.target_timestamp)
        
        # Execute prediction
        res = predict_movement(
            symbol=val['symbol'],
            reference_timestamp=request.reference_timestamp,
            target_timestamp=request.target_timestamp,
            model_type=request.model_type or "pooled"
        )
        
        model_name = "Logistic Regression (L2, C=1.0, class_weight='balanced')" if request.model_type == "pooled" else "H1 Random Forest (depth=6)"
        
        return {
            "symbol": res['symbol'],
            "reference_timestamp": res['reference_timestamp'],
            "target_timestamp": res['target_timestamp'],
            "reference_price": float(res['reference_price']),
            "observed_sessions_ahead": int(res['target_horizon_observed_sessions']),
            "calendar_days_ahead": int(res['target_horizon_calendar_days']),
            "predicted_class": res['predicted_class'],
            "probability_down": float(res['probability_down']),
            "probability_stable": float(res['probability_stable']),
            "probability_up": float(res['probability_up']),
            "model_name": model_name,
            "model_version": "1.0.0"
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

@app.post("/paper/predict", response_model=PaperPredictionResponse, tags=["Paper Trading"])
def create_prediction_record(request: PredictionRequest):
    """
    Stateful paper prediction endpoint.
    Creates and stores an OPEN prediction record in paper_predictions.
    """
    try:
        rec = create_paper_prediction(
            symbol=request.symbol,
            reference_timestamp=request.reference_timestamp,
            target_timestamp=request.target_timestamp,
            model_type=request.model_type or "pooled"
        )
        return rec
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Paper prediction error: {str(e)}")

@app.post("/paper/resolve", tags=["Paper Trading"])
def resolve_all():
    """Resolves all eligible pending OPEN paper predictions against market data."""
    try:
        counts = resolve_all_pending()
        return {"status": "SUCCESS", "resolution_counts": counts}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/paper/resolve/{prediction_id}", response_model=PaperPredictionResponse, tags=["Paper Trading"])
def resolve_single(prediction_id: str):
    """Resolves a specific paper prediction by ID."""
    try:
        rec = resolve_prediction(prediction_id)
        return rec
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/paper/predictions", tags=["Paper Trading"])
def get_predictions(
    symbol: Optional[str] = Query(None, description="Filter by ticker symbol"),
    status_filter: Optional[str] = Query(None, alias="status", description="OPEN, RESOLVED, EXPIRED, INVALID"),
    horizon: Optional[int] = Query(None, description="Observed sessions ahead (1-7)"),
    predicted_class: Optional[str] = Query(None, description="UP, DOWN, STABLE"),
    limit: int = Query(200, ge=1, le=1000)
):
    """Retrieves list of persisted paper predictions with optional filters."""
    return list_predictions(symbol=symbol, status=status_filter, horizon=horizon, predicted_class=predicted_class, limit=limit)

@app.get("/paper/predictions/{prediction_id}", response_model=PaperPredictionResponse, tags=["Paper Trading"])
def get_single_prediction(prediction_id: str):
    """Retrieves details of a single paper prediction."""
    rec = get_prediction(prediction_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Prediction '{prediction_id}' not found.")
    return rec

@app.get("/paper/performance", tags=["Performance"])
def get_performance():
    """
    Returns forward paper evaluation metrics, horizon breakdown, confusion matrix,
    and comparison with Phase 5 historical test benchmark.
    """
    return generate_paper_performance_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
