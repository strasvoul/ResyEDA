"""
End-to-end EDA run:
    1. Extract raw tables from Postgres (or load from local cache)
    2. Enrich DailyRoutes with season_type, holiday flag, sentinel flags
    3. Run the full EDA suite (prints diagnostics, saves plots)

Usage:
    python run_eda.py                 # use cached data if available
    python run_eda.py --refresh       # force a fresh pull from the DB
    python run_eda.py --lookback-days 730
"""
import argparse

from src.enrich import build_demand_table
from src.extract import extract_all
from src.eda import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"refresh": args.refresh}
    if args.lookback_days is not None:
        kwargs["lookback_days"] = args.lookback_days

    print("Extracting tables...")
    tables = extract_all(**kwargs)

    print("\nBuilding enriched demand table...")
    demand_df = build_demand_table(tables)
    demand_df.to_parquet("data/raw/demand_table.parquet", index=False)

    print("\nRunning EDA...")
    run_all(demand_df)
