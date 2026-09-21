"""
Market Data Access Layer for Ziro.

Provides clean abstraction for market-data retrieval from the Frozen Parquet Dataset
using PyArrow Dataset with column and predicate pushdown.

Ensures zero-copy / low-memory streaming and strict data preservation:
- UTC timestamps with Asia/Kolkata timezone conversion
- IEEE 754 float64 price and volume values
- Exact symbol preservation
"""

import os
from pathlib import Path
from typing import Optional, Set, List, Dict, Any
from datetime import datetime, time as dt_time
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc

from src.config import (
    PROJECT_ROOT,
    FROZEN_PARQUET_PATH,
    TIMEZONE,
    MARKET_OPEN,
)

class MarketDataReader:
    """Market Data Reader querying the Frozen Parquet Dataset."""

    def __init__(self, parquet_path: Optional[Path] = None):
        self.parquet_path = parquet_path or FROZEN_PARQUET_PATH
        self._dataset: Optional[ds.Dataset] = None
        self._symbols_cache: Optional[Set[str]] = None
        self._dates_cache: Optional[List[str]] = None

    @property
    def dataset(self) -> ds.Dataset:
        if self._dataset is None:
            if not self.parquet_path.exists():
                raise FileNotFoundError(
                    f"Frozen Parquet dataset not found at: {self.parquet_path}. "
                    "Ensure data/ziro_frozen_dataset.parquet exists."
                )
            self._dataset = ds.dataset(str(self.parquet_path), format="parquet")
        return self._dataset

    def get_symbols(self, source: str = "parquet", **kwargs) -> Set[str]:
        """Returns set of legitimate canonical equity symbols from frozen Parquet dataset."""
        if self._symbols_cache is not None:
            return self._symbols_cache
        symbols = set()
        for batch in self.dataset.to_batches(columns=["symbol"]):
            symbols.update(pc.unique(batch["symbol"]).to_pylist())
        self._symbols_cache = {s.upper() for s in symbols}
        return self._symbols_cache

    def get_trading_dates(self, source: str = "parquet", **kwargs) -> List[str]:
        """Returns sorted list of distinct trading dates (YYYY-MM-DD) in IST."""
        if self._dates_cache is not None:
            return self._dates_cache
        filter_expr = ds.field("symbol").isin(["RELIANCE", "360ONE"])
        table = self.dataset.to_table(filter=filter_expr, columns=["ts"])
        ts_series = table.column("ts").to_pandas()
        dates = sorted(ts_series.dt.tz_convert(TIMEZONE).dt.date.astype(str).unique().tolist())
        self._dates_cache = dates
        return self._dates_cache

    def get_morning_candles(
        self,
        symbol: str,
        trade_date: str,
        ref_time: str = "10:30:00",
        source: str = "parquet",
        **kwargs
    ) -> pd.DataFrame:
        """
        Retrieves morning candles for a symbol on a specific date from 09:15 up to ref_time.
        Columns returned: ['ts', 'time_ist', 'open', 'high', 'low', 'close', 'volume']
        """
        symbol = symbol.strip().upper()

        # Convert IST boundaries to UTC for predicate pushdown
        start_utc = pd.to_datetime(f"{trade_date} {MARKET_OPEN}").tz_localize(TIMEZONE).tz_convert("UTC")
        end_utc = pd.to_datetime(f"{trade_date} {ref_time}").tz_localize(TIMEZONE).tz_convert("UTC")

        filter_expr = (
            (ds.field("symbol") == symbol) &
            (ds.field("ts") >= start_utc) &
            (ds.field("ts") <= end_utc)
        )
        table = self.dataset.to_table(
            filter=filter_expr,
            columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df = table.to_pandas()
        if df.empty:
            return pd.DataFrame(columns=["ts", "time_ist", "open", "high", "low", "close", "volume"])

        df = df.sort_values("ts").reset_index(drop=True)
        df["time_ist"] = df["ts"].dt.tz_convert(TIMEZONE).dt.time
        return df[["ts", "time_ist", "open", "high", "low", "close", "volume"]]

    def get_target_candle_close(
        self,
        symbol: str,
        tgt_date: str,
        tgt_time: str = "10:30:00",
        source: str = "parquet",
        **kwargs
    ) -> Optional[float]:
        """
        Retrieves the exact close price for a symbol at target date and target time.
        Returns float if found, None if candle is absent.
        """
        symbol = symbol.strip().upper()

        tgt_utc = pd.to_datetime(f"{tgt_date} {tgt_time}").tz_localize(TIMEZONE).tz_convert("UTC")
        filter_expr = (
            (ds.field("symbol") == symbol) &
            (ds.field("ts") == tgt_utc)
        )
        table = self.dataset.to_table(
            filter=filter_expr,
            columns=["close"]
        )
        if len(table) > 0:
            return float(table.column("close")[0].as_py())
        return None


# Global default instance
default_reader = MarketDataReader()

def get_legitimate_symbols(source: str = "parquet", **kwargs) -> Set[str]:
    return default_reader.get_symbols(source=source, **kwargs)

def get_historical_dates(source: str = "parquet", **kwargs) -> List[str]:
    return default_reader.get_trading_dates(source=source, **kwargs)

def get_morning_candles(
    symbol: str,
    trade_date: str,
    ref_time: str = "10:30:00",
    source: str = "parquet",
    **kwargs
) -> pd.DataFrame:
    return default_reader.get_morning_candles(
        symbol=symbol,
        trade_date=trade_date,
        ref_time=ref_time,
        source=source,
        **kwargs
    )

def get_target_candle_close(
    symbol: str,
    tgt_date: str,
    tgt_time: str = "10:30:00",
    source: str = "parquet",
    **kwargs
) -> Optional[float]:
    return default_reader.get_target_candle_close(
        symbol=symbol,
        tgt_date=tgt_date,
        tgt_time=tgt_time,
        source=source,
        **kwargs
    )
