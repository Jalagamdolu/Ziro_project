"""
Configuration and constants for NSE Intraday ML Project.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database URL - uses the existing TimescaleDB/PostgreSQL instance
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://nse_user:nse_password@localhost:5433/nse_minute"
)

# Canonical Ingestion Source
CANONICAL_SOURCE = "yahoo"

# Verified 17 Synthetic / Test Symbols to exclude
EXCLUDED_SYNTHETIC_SYMBOLS = (
    "OKSYM", "SYM01", "SYM02", "SYM03", "SYM04", "SYM06", "SYM07", "SYM08", "SYM09",
    "SYM1", "SYM10", "SYM2", "SYM3", "SYMCOMP", "TESTBATCHSYM", "TESTRUNAUDITSYM", "TESTSYM"
)

# SQL clause for synthetic exclusion
SYNTH_SQL_TUPLE = "('" + "','".join(EXCLUDED_SYNTHETIC_SYMBOLS) + "')"

# Market Hours & Timezone
TIMEZONE = "Asia/Kolkata"
MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"

# Target Definition Threshold
UP_THRESHOLD_PCT = 1.0
DOWN_THRESHOLD_PCT = -1.0
