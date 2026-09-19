"""
Builds analysis-ready tables from the raw extracts.
Includes functions to transform raw Bookings and BookingTransportations into
analysis-ready DailyRoutes demand tables, and to compare derived demand
against stored DailyRoutes counters.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import BOOKING_TYPE_GROUP, BOOKING_TYPE_INDIVIDUAL



# ---------------------------------------------------------------------------
# NEW: Bookings -> DailyRoutes demand derivation
# ---------------------------------------------------------------------------

def bookings_to_legs(bookings: pd.DataFrame) -> pd.DataFrame:
    """Explodes each booking into its available leg(s).

    Bookings with Details_NoTransportation = True are dropped entirely -
    there's no route to attribute their guest count to.
    """
    b = bookings[~bookings["no_transportation"].fillna(False)].copy()

    base_cols = ["booking_id", "number_of_guests", "status", "travel_date",
                 "created_date", "booking_type"]

    originating = b.loc[b["originating_transportation_id"].notna(), base_cols + ["originating_transportation_id"]].copy()
    originating = originating.rename(columns={"originating_transportation_id": "transportation_id"})
    originating["leg"] = "originating"

    returning = b.loc[b["return_transportation_id"].notna(), base_cols + ["return_transportation_id"]].copy()
    returning = returning.rename(columns={"return_transportation_id": "transportation_id"})
    returning["leg"] = "returning"

    legs = pd.concat([originating, returning], ignore_index=True)

    n_no_leg_at_all = b["booking_id"].nunique() - legs["booking_id"].nunique()
    if n_no_leg_at_all > 0:
        print(
            f"[bookings->legs] {n_no_leg_at_all:,} bookings had NEITHER "
            "originating_transportation_id nor return_transportation_id "
            "populated and were dropped."
        )

    return legs


def derive_demand_from_bookings(
    legs: pd.DataFrame,
    booking_transportations: pd.DataFrame,
    excluded_status_codes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Aggregates Bookings -> BookingTransportations -> DailyRouteId into a
    per-daily_route_id demand table, for cross-checking against DailyRoutes'
    own stored booked-seat counters.

    Exclusion rule (per spec): a booking is excluded entirely only when
    NONE of its legs resolve to a daily_route_id - i.e. it's one-way
    (ascending or descending) and its single leg's transportation has no
    daily_route_id, OR it's two-way and BOTH legs have no daily_route_id.
    A two-way booking where exactly one leg resolves is KEPT - that leg
    still represents real seat demand on a real route; only the unresolved
    leg is left out of the aggregate.
    """
    legs = legs[~legs["status"].isin(list(excluded_status_codes))]

    merged = legs.merge(
        booking_transportations[["transportation_id", "daily_route_id"]],
        on="transportation_id",
        how="left",
    )

    resolved = merged.dropna(subset=["daily_route_id"])

    bookings_seen = merged["booking_id"].nunique()
    bookings_fully_unrouted = bookings_seen - resolved["booking_id"].nunique()
    if bookings_fully_unrouted > 0:
        rate = bookings_fully_unrouted / bookings_seen
        print(
            f"[demand_from_bookings] {bookings_fully_unrouted:,} bookings "
            f"({rate:.1%}) have NO leg that resolves to a daily_route_id "
            "(one-way with an unrouted leg, or two-way with both legs "
            "unrouted) - excluded entirely, per spec. Two-way bookings "
            "with exactly one resolved leg are kept via that leg."
        )

    agg = (
        resolved.groupby("daily_route_id")
        .agg(
            derived_booked_guests=("number_of_guests", "sum"),
            derived_n_bookings=("booking_id", "nunique"),
        )
        .reset_index()
    )
    return agg

def compare_derived_vs_stored(demand_df: pd.DataFrame, derived_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the bookings-derived demand against DailyRoutes' own stored
    counters, as a data-quality cross-check. Large, systematic mismatches
    mean one of the two sources shouldn't be trusted blindly."""
    comp = demand_df[["daily_route_id", "total_booked_seats"]].merge(
        derived_df, on="daily_route_id", how="left"
    )
    comp["derived_booked_guests"] = comp["derived_booked_guests"].fillna(0)
    comp["diff"] = comp["total_booked_seats"] - comp["derived_booked_guests"]

    corr = comp[["total_booked_seats", "derived_booked_guests"]].corr().iloc[0, 1]
    within_5 = (comp["diff"].abs() <= 5).mean()
    print(f"[demand comparison] correlation(stored, derived) = {corr:.3f}")
    print(f"[demand comparison] {within_5:.1%} of routes within +/-5 seats of each other")
    print(f"[demand comparison] mean diff (stored - derived) = {comp['diff'].mean():.2f}")
    return comp


def derive_demand_from_bookings_by_type(
    legs: pd.DataFrame,
    booking_transportations: pd.DataFrame,
    excluded_status_codes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Same as derive_demand_from_bookings, but split by BookingType (Group
    vs Individual), so it can be compared against DailyRoutes'
    booked_group_seats and booked_individual_seats SEPARATELY rather than
    as one combined total. This is the direct test of whether a
    stored-vs-derived gap concentrates in one seat type or the other."""
    if excluded_status_codes is None:
        print(
            "[demand_from_bookings_by_type] WARNING: no excluded_status_codes "
            "given - including ALL bookings regardless of status."
        )
        excluded_status_codes = []

    legs = legs[~legs["status"].isin(list(excluded_status_codes))]

    merged = legs.merge(
        booking_transportations[["transportation_id", "daily_route_id"]],
        on="transportation_id",
        how="left",
    )
    resolved = merged.dropna(subset=["daily_route_id"])

    unknown_type = ~resolved["booking_type"].isin([BOOKING_TYPE_GROUP, BOOKING_TYPE_INDIVIDUAL])
    if unknown_type.any():
        print(
            f"[demand_from_bookings_by_type] WARNING: {unknown_type.sum():,} legs "
            f"have a booking_type outside {{Group={BOOKING_TYPE_GROUP}, "
            f"Individual={BOOKING_TYPE_INDIVIDUAL}}} - excluded from the split "
            "(check for nulls or an unexpected enum value)."
        )
    resolved = resolved[~unknown_type]

    pivot = (
        resolved.groupby(["daily_route_id", "booking_type"])["number_of_guests"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={BOOKING_TYPE_GROUP: "derived_group_guests",
                          BOOKING_TYPE_INDIVIDUAL: "derived_individual_guests"})
        .reset_index()
    )
    for col in ["derived_group_guests", "derived_individual_guests"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[["daily_route_id", "derived_group_guests", "derived_individual_guests"]]


def compare_derived_vs_stored_by_type(demand_df: pd.DataFrame, derived_by_type: pd.DataFrame) -> pd.DataFrame:
    """The apples-to-apples version of compare_derived_vs_stored: group
    seats vs. group guests, individual seats vs. individual guests,
    reported separately so a gap that concentrates in one type isn't
    hidden by averaging with the other."""
    comp = demand_df[["daily_route_id", "booked_group_seats", "booked_individual_seats"]].merge(
        derived_by_type, on="daily_route_id", how="left"
    ).fillna(0)

    for label, stored_col, derived_col in [
        ("group", "booked_group_seats", "derived_group_guests"),
        ("individual", "booked_individual_seats", "derived_individual_guests"),
    ]:
        diff = comp[stored_col] - comp[derived_col]
        corr = comp[[stored_col, derived_col]].corr().iloc[0, 1]
        within_5 = (diff.abs() <= 5).mean()
        print(f"[{label}] correlation = {corr:.3f}  |  within +/-5 = {within_5:.1%}  |  mean diff = {diff.mean():.2f}")

    return comp
# ---------------------------------------------------------------------------
# NEW: booking lead-time / pickup features
# ---------------------------------------------------------------------------

def compute_lead_time_days(bookings: pd.DataFrame) -> pd.DataFrame:
    """Adds lead_time_days = travel_date - created_date, in days. Negative
    values (created after travel - shouldn't normally happen) are flagged,
    not silently dropped, since they may indicate a data-entry pattern
    worth knowing about (e.g. backfilled walk-up bookings)."""
    out = bookings[~bookings["no_transportation"].fillna(False)].copy()
    out["travel_date"] = pd.to_datetime(out["travel_date"])
    out["created_date"] = pd.to_datetime(out["created_date"])
    out["lead_time_days"] = (
        out["travel_date"].dt.normalize()
        - out["created_date"].dt.normalize()
    ).dt.days
    # ensure lead_time_days is an integer
    out["lead_time_days"] = out["lead_time_days"].astype(int)
    
    n_negative = int((out["lead_time_days"] < 0).sum())
    if n_negative:
        print(
            f"[lead_time] {n_negative:,} bookings ({n_negative / len(out):.1%}) have a "
            "NEGATIVE lead time (created after travel_date) - kept in the data "
            "but worth investigating before using lead time as a model feature."
        )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_demand_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Top-level entry point for the DailyRoutes side of the pipeline.
    Bookings-derived demand and lead-time features are built separately
    (derive_demand_from_bookings / compute_lead_time_days) since they're
    at a different grain (per-booking / per-leg, not per-daily-route) and
    forcing them into this same frame would require picking an aggregation
    upfront - better done deliberately at the EDA/reporting step.
    """
    daily_routes = tables["daily_routes"].copy()

    daily_routes["day"] = pd.to_datetime(daily_routes["day"])
    daily_routes["day_of_week"] = daily_routes["day"].dt.day_name()
    daily_routes["month"] = daily_routes["day"].dt.month
    daily_routes["day_of_year"] = daily_routes["day"].dt.dayofyear
    daily_routes["year"] = daily_routes["day"].dt.year
    daily_routes["total_booked_seats"] = (
        daily_routes["booked_group_seats"] + daily_routes["booked_individual_seats"]
    )
    daily_routes["total_capacity"] = (
        daily_routes["group_capacity"] + daily_routes["individual_capacity"] + daily_routes["reserve_capacity"]
    )
    return daily_routes


def find_worst_mismatches(comparison_df: pd.DataFrame, stored_col: str, derived_col: str, top_n: int = 20) -> pd.DataFrame:
    """Surfaces the top_n route-days by ABSOLUTE mismatch. A near-zero MEAN
    diff can still hide sizeable per-route errors that cancel out in
    aggregate (some routes over, some under) - this finds the actual
    outliers worth manually inspecting, rather than trusting the mean."""
    df = comparison_df.copy()
    df["diff"] = df[stored_col] - df[derived_col]
    df["abs_diff"] = df["diff"].abs()

    print(f"[mismatch distribution] mean={df['diff'].mean():.2f}  "
          f"std={df['diff'].std():.2f}  "
          f"p1={df['diff'].quantile(0.01):.1f}  p99={df['diff'].quantile(0.99):.1f}  "
          f"max_abs={df['abs_diff'].max():.0f}")
    df = df[df["abs_diff"] > 0]
    return df.sort_values("abs_diff", ascending=False)


def investigate_route_day(
    daily_route_id: str,
    legs: pd.DataFrame,
    booking_transportations: pd.DataFrame,
    excluded_status_codes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Shows EVERY booking leg attributed - or not - to one specific
    daily_route_id, so a mismatch can be explained at the row level rather
    than guessed at. Includes legs excluded by status, so you can see
    exactly which bookings are contributing (or not) to the derived total.
    Pick the daily_route_id from find_worst_mismatches()."""
    excluded_status_codes = list(excluded_status_codes or [])

    merged = legs.merge(
        booking_transportations[["transportation_id", "daily_route_id"]],
        on="transportation_id",
        how="left",
    )
    route_legs = merged[merged["daily_route_id"] == daily_route_id].copy()
    route_legs["excluded_by_status"] = route_legs["status"].isin(excluded_status_codes)

    included = route_legs[~route_legs["excluded_by_status"]]
    route_id_to_print = daily_route_id.hex() if isinstance(daily_route_id, bytes) else daily_route_id
    print(f"[route {route_id_to_print}] {len(route_legs)} total legs, "
          f"{len(included)} included after status filter, "
          f"{included['number_of_guests'].sum()} derived guests")

    return route_legs.sort_values(["excluded_by_status", "leg", "created_date"])