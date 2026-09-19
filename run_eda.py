"""
End-to-end EDA run - revised to bring Bookings in as the primary demand
signal, not just a secondary lead-time feature.

Usage:
    python run_eda.py                     # cached data if available
    python run_eda.py --refresh           # force a fresh DB pull
    python run_eda.py --lookback-years 2
"""
import argparse
from src.constants import DEFAULT_LOOKBACK_YEARS, NON_RESERVED_STATUS_CODES
from src.enrich import (
    build_demand_table,
    compare_derived_vs_stored_by_type,
    compute_lead_time_days,
    derive_demand_from_bookings,
    compare_derived_vs_stored,
    derive_demand_from_bookings_by_type,
    find_worst_mismatches,
    investigate_route_day,
    bookings_to_legs,
)
from src.extract import extract_all
from src.eda import run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--lookback-years", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"refresh": args.refresh}
    if args.lookback_years is not None:
        kwargs["lookback_years"] = args.lookback_years

    print(f"Extracting tables for the past {kwargs.get('lookback_years', DEFAULT_LOOKBACK_YEARS) + 1} seasons:")
    tables = extract_all(**kwargs)

    print("\nBuilding enriched daily-routes demand table...")
    demand_df = build_demand_table(tables)
    demand_df.to_parquet("data/raw/demand_table.parquet", index=False)

    legs = bookings_to_legs(tables["bookings"])
    print("\nDeriving demand from Bookings (cross-check against stored counters)...")
    derived = derive_demand_from_bookings(
        legs, tables["booking_transportations"],
        NON_RESERVED_STATUS_CODES,
    )
    comparison_df = compare_derived_vs_stored(demand_df, derived)
    
    derived_by_type = derive_demand_from_bookings_by_type(
        legs, tables["booking_transportations"],
        excluded_status_codes=NON_RESERVED_STATUS_CODES,
    )
    comparison_by_type_df = compare_derived_vs_stored_by_type(demand_df, derived_by_type)
    
    worst = find_worst_mismatches(comparison_by_type_df, "booked_individual_seats", "derived_individual_guests", top_n=20)
    worst_for_print = worst["daily_route_id"].copy().apply(lambda x: x.hex() if isinstance(x, bytes) else x)
    print(f"[worst mismatches] {len(worst)} route-days by absolute mismatch")
    print(worst.assign(daily_route_id=worst_for_print)[["daily_route_id", "booked_individual_seats", "derived_individual_guests", "diff"]])
    

    # pick the biggest offender and drill in
    investigate_route_day(worst.iloc[0]["daily_route_id"], legs, tables["booking_transportations"],
                       excluded_status_codes=NON_RESERVED_STATUS_CODES) 

    print("\nComputing booking lead time / pickup features...")
    bookings_lt = compute_lead_time_days(tables["bookings"])
    

    print("\nRunning EDA...")
    run_all(demand_df, bookings_with_lead_time=bookings_lt, demand_comparison_df=comparison_df)
