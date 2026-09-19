"""
Runs the extraction queries against the live DB and caches results to
data/raw/*.parquet, so subsequent EDA iterations don't need to re-hit the
database every time.

Usage:
    python -m src.extract              # pulls everything, uses cache if fresh
    python -m src.extract --refresh    # forces a fresh pull from the DB
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.constants import DEFAULT_LOOKBACK_YEARS
from src.db import get_engine
from src.queries import ALL_QUERIES, LOOKBACK_PARAMETERIZED

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"


def extract_table(name: str, sql: str, refresh: bool) -> pd.DataFrame:
    cache_path = _cache_path(name)
    if cache_path.exists() and not refresh:
        print(f"[cache] {name} <- {cache_path.name} ({len(pd.read_parquet(cache_path)):,} rows)")
        return pd.read_parquet(cache_path)

    print(f"[db]    {name} ...")
    engine = get_engine()
    params = {"lookback_years": DEFAULT_LOOKBACK_YEARS} if name in LOOKBACK_PARAMETERIZED else {}
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    df.to_parquet(cache_path, index=False)
    print(f"        -> {len(df):,} rows, cached to {cache_path.name}")
    return df


def extract_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {
        name: extract_table(name, sql, refresh)
        for name, sql in ALL_QUERIES.items()
    }

