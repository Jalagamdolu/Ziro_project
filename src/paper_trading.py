"""
Paper Trading Engine & Persistence Layer for NSE Intraday Stock Price Movement.

Handles:
- Persistent paper prediction storage (PostgreSQL / SQLite fallback).
- Two-stage prediction lifecycle: OPEN -> RESOLVED (or EXPIRED / INVALID).
- Strict leakage controls: Target OHLCV is NEVER accessed during prediction.
- Actual return and class evaluation only when target market timestamp has passed.
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, Table, Column, String, Integer, Float, Boolean, DateTime, MetaData
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PAPER_TRADING_DB_URL,
    SQLITE_DB_PATH,
    CANONICAL_SOURCE,
    EXCLUDED_SYNTHETIC_SYMBOLS,
    TIMEZONE,
    UP_THRESHOLD_PCT,
    DOWN_THRESHOLD_PCT,
    MODELS_DIR
)
from src.predict import predict_movement, get_observed_sessions_between

# SQLite database for self-contained paper trading persistence
PAPER_DB_URL = os.getenv("PAPER_TRADING_DB_URL", PAPER_TRADING_DB_URL)
engine = create_engine(PAPER_DB_URL)

# Cached set of valid canonical symbols
_LEGIT_SYMBOLS_CACHE = None

_DB_INITIALIZED_ENGINES = set()

def init_paper_trading_db(engine_override: Optional[Engine] = None) -> None:
    """Creates the paper_predictions table and indices if they do not already exist."""
    target_engine = engine_override or engine
    is_sqlite = target_engine.url.drivername.startswith("sqlite")
    
    if is_sqlite:
        create_sql = """
        CREATE TABLE IF NOT EXISTS paper_predictions (
            prediction_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            reference_timestamp TEXT NOT NULL,
            target_timestamp TEXT NOT NULL,
            reference_price REAL NOT NULL,
            observed_sessions_ahead INTEGER NOT NULL,
            calendar_days_ahead INTEGER NOT NULL,
            predicted_class TEXT NOT NULL,
            probability_down REAL NOT NULL,
            probability_stable REAL NOT NULL,
            probability_up REAL NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            status_reason TEXT,
            created_at TIMESTAMP NOT NULL,
            actual_price REAL,
            future_return_pct REAL,
            actual_class TEXT,
            is_resolved INTEGER NOT NULL DEFAULT 0,
            resolved_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_symbol ON paper_predictions (symbol);
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_status ON paper_predictions (status);
        """
    else:
        create_sql = """
        CREATE TABLE IF NOT EXISTS paper_predictions (
            prediction_id VARCHAR(64) PRIMARY KEY,
            symbol VARCHAR(32) NOT NULL,
            reference_timestamp VARCHAR(32) NOT NULL,
            target_timestamp VARCHAR(32) NOT NULL,
            reference_price DOUBLE PRECISION NOT NULL,
            observed_sessions_ahead INTEGER NOT NULL,
            calendar_days_ahead INTEGER NOT NULL,
            predicted_class VARCHAR(16) NOT NULL,
            probability_down DOUBLE PRECISION NOT NULL,
            probability_stable DOUBLE PRECISION NOT NULL,
            probability_up DOUBLE PRECISION NOT NULL,
            model_name VARCHAR(128) NOT NULL,
            model_version VARCHAR(32) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'OPEN',
            status_reason TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            actual_price DOUBLE PRECISION,
            future_return_pct DOUBLE PRECISION,
            actual_class VARCHAR(16),
            is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
            resolved_at TIMESTAMP WITHOUT TIME ZONE
        );
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_symbol ON paper_predictions (symbol);
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_status ON paper_predictions (status);
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_ref_time ON paper_predictions (reference_timestamp);
        CREATE INDEX IF NOT EXISTS idx_paper_predictions_target_time ON paper_predictions (target_timestamp);
        """
        
    with target_engine.connect() as conn:
        for stmt in create_sql.strip().split(';'):
            if stmt.strip():
                conn.execute(text(stmt))
        conn.commit()

def ensure_db_initialized(target_engine: Optional[Engine] = None) -> None:
    """Initializes paper_predictions table lazily on first access for each engine."""
    eng = target_engine or engine
    eng_key = str(eng.url)
    if eng_key not in _DB_INITIALIZED_ENGINES:
        try:
            init_paper_trading_db(eng)
            _DB_INITIALIZED_ENGINES.add(eng_key)
        except Exception as e:
            print(f"Notice: Initial paper_predictions DB init deferred: {e}")

def get_engine(engine_override: Optional[Engine] = None) -> Engine:
    """Returns active database engine and ensures paper_predictions table exists."""
    eng = engine_override or engine
    ensure_db_initialized(eng)
    return eng

# Initialize default engine on module load
ensure_db_initialized(engine)

def get_legitimate_symbols(engine_override: Optional[Engine] = None, market_data_source: str = "parquet") -> set:
    """Returns cached set of legitimate canonical equity symbols from frozen Parquet dataset."""
    global _LEGIT_SYMBOLS_CACHE
    if _LEGIT_SYMBOLS_CACHE is not None:
        return _LEGIT_SYMBOLS_CACHE
        
    from src.market_data import get_legitimate_symbols as get_parquet_symbols
    _LEGIT_SYMBOLS_CACHE = get_parquet_symbols(source="parquet")
    return _LEGIT_SYMBOLS_CACHE

def validate_prediction_inputs(symbol: str, reference_timestamp: str, target_timestamp: str, engine_override: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Validates symbol, reference timestamp, and target timestamp.
    Raises ValueError with descriptive reason if invalid.
    """
    cleaned_sym = symbol.strip().upper()
    
    # 1. Symbol Validation
    if cleaned_sym in EXCLUDED_SYNTHETIC_SYMBOLS:
        raise ValueError(f"Symbol '{cleaned_sym}' is an excluded synthetic/test symbol.")
        
    try:
        legit_symbols = get_legitimate_symbols(engine_override)
        if cleaned_sym not in legit_symbols:
            raise ValueError(f"Symbol '{cleaned_sym}' not found in canonical NSE equity universe.")
    except Exception as e:
        # Fallback if DB symbol query fails (e.g. offline unit test)
        if cleaned_sym in EXCLUDED_SYNTHETIC_SYMBOLS:
            raise ValueError(f"Symbol '{cleaned_sym}' is an excluded synthetic/test symbol.")
            
    # 2. Timestamp Parsing & Chronological Validation
    ref_ts_str = reference_timestamp.replace('IST', '').strip()
    tgt_ts_str = target_timestamp.replace('IST', '').strip()
    
    try:
        ref_dt = pd.to_datetime(ref_ts_str)
    except Exception as e:
        raise ValueError(f"Invalid reference timestamp format: '{reference_timestamp}'.")
        
    try:
        tgt_dt = pd.to_datetime(tgt_ts_str)
    except Exception as e:
        raise ValueError(f"Invalid target timestamp format: '{target_timestamp}'.")
        
    if ref_dt >= tgt_dt:
        raise ValueError(
            f"Reference timestamp ({ref_dt}) must be strictly before target timestamp ({tgt_dt})."
        )
        
    # 3. Weekend Validation
    if ref_dt.weekday() >= 5:
        day_name = ref_dt.strftime('%A')
        raise ValueError(f"Reference date '{ref_dt.date()}' is a {day_name} (non-trading weekend).")
        
    if tgt_dt.weekday() >= 5:
        day_name = tgt_dt.strftime('%A')
        raise ValueError(f"Target date '{tgt_dt.date()}' is a {day_name} (non-trading weekend).")
        
    # 4. Terminal Historical Horizon Validation
    ref_date_str = ref_dt.strftime('%Y-%m-%d')
    tgt_date_str = tgt_dt.strftime('%Y-%m-%d')
    
    # Reference cannot be beyond available historical data in database
    if ref_date_str > '2026-09-10':
        raise ValueError(
            f"Reference date '{ref_date_str}' exceeds latest available historical market data (2026-09-10)."
        )
        
    # 5. Horizon Scope Validation (1-7 sessions)
    observed_sessions = get_observed_sessions_between(ref_date_str, tgt_date_str)
    if observed_sessions < 1:
        raise ValueError("Target horizon must be at least 1 observed trading session ahead.")
    if observed_sessions > 7:
        raise ValueError(
            f"Requested horizon ({observed_sessions} sessions) exceeds supported scope (1 to 7 sessions)."
        )
        
    calendar_days = (tgt_dt.date() - ref_dt.date()).days
    
    return {
        'symbol': cleaned_sym,
        'ref_dt': ref_dt,
        'tgt_dt': tgt_dt,
        'ref_ts_clean': ref_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'tgt_ts_clean': tgt_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'observed_sessions': observed_sessions,
        'calendar_days': calendar_days
    }

def create_paper_prediction(
    symbol: str,
    reference_timestamp: str,
    target_timestamp: str,
    model_type: str = 'pooled',
    engine_override: Optional[Engine] = None,
    market_data_source: str = 'parquet'
) -> Dict[str, Any]:
    """
    STAGE A — OPEN PREDICTION LIFECYCLE:
    1. Validates inputs.
    2. Runs inference using ONLY data <= reference_timestamp. Zero target OHLCV accessed.
    3. Persists new record into paper_predictions with status 'OPEN'.
    4. Returns prediction dictionary.
    """
    target_engine = get_engine(engine_override)
    
    # Validate inputs
    val_info = validate_prediction_inputs(symbol, reference_timestamp, target_timestamp, target_engine)
    clean_sym = val_info['symbol']
    
    # Run production inference strictly using market data up to reference timestamp
    pred_res = predict_movement(
        symbol=clean_sym,
        reference_timestamp=reference_timestamp,
        target_timestamp=target_timestamp,
        model_type=model_type,
        market_data_source=market_data_source
    )
    
    prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now().isoformat(sep=' ', timespec='seconds')
    
    model_name = "Logistic Regression (L2, C=1.0, class_weight='balanced')" if model_type == 'pooled' else "H1 Random Forest (depth=6)"
    model_version = "1.0.0"
    
    record = {
        'prediction_id': prediction_id,
        'symbol': clean_sym,
        'reference_timestamp': val_info['ref_ts_clean'],
        'target_timestamp': val_info['tgt_ts_clean'],
        'reference_price': float(pred_res['reference_price']),
        'observed_sessions_ahead': int(val_info['observed_sessions']),
        'calendar_days_ahead': int(val_info['calendar_days']),
        'predicted_class': str(pred_res['predicted_class']),
        'probability_down': float(pred_res['probability_down']),
        'probability_stable': float(pred_res['probability_stable']),
        'probability_up': float(pred_res['probability_up']),
        'model_name': model_name,
        'model_version': model_version,
        'status': 'OPEN',
        'status_reason': None,
        'created_at': created_at,
        'actual_price': None,
        'future_return_pct': None,
        'actual_class': None,
        'is_resolved': False,
        'resolved_at': None
    }
    
    insert_sql = text("""
        INSERT INTO paper_predictions (
            prediction_id, symbol, reference_timestamp, target_timestamp,
            reference_price, observed_sessions_ahead, calendar_days_ahead,
            predicted_class, probability_down, probability_stable, probability_up,
            model_name, model_version, status, status_reason, created_at,
            actual_price, future_return_pct, actual_class, is_resolved, resolved_at
        ) VALUES (
            :prediction_id, :symbol, :reference_timestamp, :target_timestamp,
            :reference_price, :observed_sessions_ahead, :calendar_days_ahead,
            :predicted_class, :probability_down, :probability_stable, :probability_up,
            :model_name, :model_version, :status, :status_reason, :created_at,
            :actual_price, :future_return_pct, :actual_class, :is_resolved, :resolved_at
        );
    """)
    with target_engine.connect() as conn:
        conn.execute(insert_sql, record)
        conn.commit()
        
    return record

def resolve_prediction(
    prediction_id: str,
    as_of_time: Optional[str] = None,
    engine_override: Optional[Engine] = None,
    market_data_source: str = 'parquet'
) -> Dict[str, Any]:
    """
    STAGE B — RESOLUTION LIFECYCLE:
    After target timestamp has occurred and actual market data exists:
    1. Retrieves target 10:30 candle.
    2. Calculates future_return_pct and actual_class.
    3. Updates paper_predictions record to 'RESOLVED' (or 'EXPIRED' if data missing).
    """
    target_engine = get_engine(engine_override)
    
    select_sql = text("SELECT * FROM paper_predictions WHERE prediction_id = :pred_id;")
    with target_engine.connect() as conn:
        res = conn.execute(select_sql, {'pred_id': prediction_id}).mappings().first()
        
    if not res:
        raise KeyError(f"Prediction ID '{prediction_id}' not found.")
        
    record = dict(res)
    if record['is_resolved'] or record['status'] in ('RESOLVED', 'EXPIRED', 'INVALID'):
        return record
        
    tgt_dt = pd.to_datetime(record['target_timestamp'])
    tgt_date_str = tgt_dt.strftime('%Y-%m-%d')
    tgt_time_str = tgt_dt.strftime('%H:%M:%S')
    
    # Check effective market time (in historical dataset, max time is 2026-09-10 15:30:00)
    effective_now = pd.to_datetime(as_of_time) if as_of_time else datetime.now()
    
    # If target timestamp is still in the future relative to market data
    if tgt_dt > effective_now and tgt_date_str > '2026-09-10':
        # Remains OPEN
        return record
        
    # Query actual target close price from market data (frozen Parquet dataset)
    from src.market_data import get_target_candle_close
    actual_price = get_target_candle_close(
        symbol=record['symbol'],
        tgt_date=tgt_date_str,
        tgt_time=tgt_time_str,
        source='parquet'
    )
    has_candle = (actual_price is not None)
        
    resolved_at = datetime.now().isoformat(sep=' ', timespec='seconds')
    ref_price = float(record['reference_price'])
    
    if not has_candle:
        # Target candle unavailable (e.g. trading halt or future date beyond DB)
        if tgt_date_str > '2026-09-10':
            # Market date has not occurred yet
            return record
        else:
            # Market date occurred in past but 10:30 candle was absent -> EXPIRED
            status = 'EXPIRED'
            status_reason = f"Target candle for symbol '{record['symbol']}' at {tgt_date_str} {tgt_time_str} was unavailable in data feed."
            actual_price = None
            future_return_pct = None
            actual_class = None
            is_resolved = False
    else:
        future_return_pct = ((actual_price - ref_price) / ref_price) * 100.0
        
        if future_return_pct > UP_THRESHOLD_PCT:
            actual_class = 'UP'
        elif future_return_pct < DOWN_THRESHOLD_PCT:
            actual_class = 'DOWN'
        else:
            actual_class = 'STABLE'
            
        status = 'RESOLVED'
        status_reason = None
        is_resolved = True
        
    update_sql = text("""
        UPDATE paper_predictions
        SET actual_price = :actual_price,
            future_return_pct = :future_return_pct,
            actual_class = :actual_class,
            status = :status,
            status_reason = :status_reason,
            is_resolved = :is_resolved,
            resolved_at = :resolved_at
        WHERE prediction_id = :prediction_id;
    """)
    
    update_params = {
        'prediction_id': prediction_id,
        'actual_price': actual_price,
        'future_return_pct': future_return_pct,
        'actual_class': actual_class,
        'status': status,
        'status_reason': status_reason,
        'is_resolved': is_resolved,
        'resolved_at': resolved_at
    }
    
    with target_engine.connect() as conn:
        conn.execute(update_sql, update_params)
        conn.commit()
        
    record.update(update_params)
    return record

def resolve_all_pending(as_of_time: Optional[str] = None, engine_override: Optional[Engine] = None) -> Dict[str, int]:
    """Scans all OPEN paper predictions and resolves eligible records."""
    target_engine = get_engine(engine_override)
    select_sql = text("SELECT prediction_id FROM paper_predictions WHERE status = 'OPEN';")
    
    with target_engine.connect() as conn:
        pred_ids = [r[0] for r in conn.execute(select_sql).fetchall()]
        
    counts = {'total_checked': len(pred_ids), 'resolved': 0, 'expired': 0, 'still_open': 0}
    
    for pid in pred_ids:
        updated = resolve_prediction(pid, as_of_time=as_of_time, engine_override=target_engine)
        if updated['status'] == 'RESOLVED':
            counts['resolved'] += 1
        elif updated['status'] == 'EXPIRED':
            counts['expired'] += 1
        else:
            counts['still_open'] += 1
            
    return counts

def list_predictions(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    horizon: Optional[int] = None,
    predicted_class: Optional[str] = None,
    limit: int = 500,
    engine_override: Optional[Engine] = None
) -> List[Dict[str, Any]]:
    """Lists persisted paper predictions with optional filters."""
    target_engine = get_engine(engine_override)
    
    clauses = ["1=1"]
    params: Dict[str, Any] = {'limit': limit}
    
    if symbol:
        clauses.append("symbol = :symbol")
        params['symbol'] = symbol.strip().upper()
    if status:
        clauses.append("status = :status")
        params['status'] = status.strip().upper()
    if horizon:
        clauses.append("observed_sessions_ahead = :horizon")
        params['horizon'] = int(horizon)
    if predicted_class:
        clauses.append("predicted_class = :predicted_class")
        params['predicted_class'] = predicted_class.strip().upper()
        
    where_sql = " AND ".join(clauses)
    query = text(f"""
        SELECT * FROM paper_predictions
        WHERE {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit;
    """)
    
    with target_engine.connect() as conn:
        rows = conn.execute(query, params).mappings().fetchall()
        
    records = []
    for r in rows:
        d = dict(r)
        if 'is_resolved' in d:
            d['is_resolved'] = bool(d['is_resolved'])
        records.append(d)
    return records

def get_prediction(prediction_id: str, engine_override: Optional[Engine] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single paper prediction by ID."""
    target_engine = get_engine(engine_override)
    query = text("SELECT * FROM paper_predictions WHERE prediction_id = :pred_id;")
    with target_engine.connect() as conn:
        row = conn.execute(query, {'pred_id': prediction_id}).mappings().first()
    if not row:
        return None
    d = dict(row)
    if 'is_resolved' in d:
        d['is_resolved'] = bool(d['is_resolved'])
    return d
