"""
Runs the extraction queries against the live DB and caches results to
data/raw/*.parquet, so subsequent EDA iterations don't need to re-hit the
database every time.

Usage:
    python -m src.extract              # pulls everything, uses cache if fresh
    python -m src.extract --refresh    # forces a fresh pull from the DB
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine
from src.queries import ALL_QUERIES, LOOKBACK_PARAMETERIZED

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOOKBACK_DAYS = int(os.getenv("RESY_EDA_LOOKBACK_DAYS", "2500"))  


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"


def extract_table(name: str, sql: str, lookback_days: int, refresh: bool) -> pd.DataFrame:
    cache_path = _cache_path(name)
    if cache_path.exists() and not refresh:
        print(f"[cache] {name} <- {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"[db]    {name} ...")
    engine = get_engine()
    params = {"lookback_days": lookback_days} if name in LOOKBACK_PARAMETERIZED else {}
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    df.to_parquet(cache_path, index=False)
    print(f"        -> {len(df):,} rows, cached to {cache_path.name}")
    return df


def extract_all(lookback_days: int = DEFAULT_LOOKBACK_DAYS, refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {
        name: extract_table(name, sql, lookback_days, refresh)
        for name, sql in ALL_QUERIES.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract resy tables for EDA")
    parser.add_argument("--refresh", action="store_true", help="bypass local parquet cache")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    args = parser.parse_args()

    tables = extract_all(lookback_days=args.lookback_days, refresh=args.refresh)
    print("\nExtraction summary:")
    for name, df in tables.items():
        print(f"  {name:22s} {len(df):>8,} rows  {len(df.columns):>3} cols")
