"""
Builds the analysis-ready demand table from the raw extracts.

Two joins here are deliberately NOT simple foreign keys, because of how the
schema models seasons and holidays:

1. Season lookup: DailyRoutes.YearParametersId doesn't point at a season
   directly. It points at a YearParameters row, which belongs to an
   OperatingYears row (via one of three possible FK columns depending on
   whether it's the cogwheel/cablecar/summer-facility parameters), which
   in turn has a PricingCatalogId, whose SeasonDefinitions rows define
   SeasonType by date range. So: DailyRoute -> YearParameters ->
   OperatingYear -> PricingCatalog -> SeasonDefinitions (matched on date).

2. Holiday flag: Tags rows have their own date range and YearParametersId,
   independent of TimetableRoutes. We flag a DailyRoutes row as
   "in a tagged period" if its Day falls inside any Tag's
   [StartingDate, EndingDate] for the same YearParametersId.

Both are exactly the kind of relationship that's easy to get subtly wrong,
so keep an eye on the coverage stats this module prints - a low match rate
usually means a join assumption above doesn't hold for this data and needs
revisiting before it feeds a model.
"""
from __future__ import annotations

import pandas as pd


def build_year_parameters_to_operating_year(
    year_parameters: pd.DataFrame, operating_years: pd.DataFrame
) -> pd.DataFrame:
    """Maps a YearParametersId to its OperatingYears row, regardless of
    which of the three FK slots (cogwheel/cablecar/summer) it fills."""
    melted = operating_years.melt(
        id_vars=["operating_year_id", "pricing_catalog_id", "is_active"],
        value_vars=[
            "cogwheel_year_parameters_id",
            "cablecar_year_parameters_id",
            "summer_facility_year_parameters_id",
        ],
        var_name="facility_kind",
        value_name="year_parameters_id",
    ).dropna(subset=["year_parameters_id"])
    return melted




def build_holiday_flag(daily_routes: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Adds an `is_tagged_period` boolean column to daily_routes."""
    out = daily_routes.copy()
    out["day"] = pd.to_datetime(out["day"])
    out["is_tagged_period"] = False

    tags = tags.copy()
    tags["starting_date"] = pd.to_datetime(tags["starting_date"])
    tags["ending_date"] = pd.to_datetime(tags["ending_date"])

    # Small tables - an explicit loop-join is fine here and much easier to
    # audit than a vectorized interval join.
    for yp_id, group in tags.groupby("year_parameters_id"):
        mask_yp = out["year_parameters_id"] == yp_id
        if not mask_yp.any():
            continue
        for _, tag in group.iterrows():
            in_range = out.loc[mask_yp, "day"].between(tag["starting_date"], tag["ending_date"])
            out.loc[mask_yp & out.index.isin(out.loc[mask_yp][in_range].index), "is_tagged_period"] = True

    rate = out["is_tagged_period"].mean()
    print(f"[holiday_flag] {rate:.1%} of daily routes fall inside a tagged period")
    return out


def build_demand_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Top-level entry point: takes the dict returned by extract_all() and
    returns one enriched, analysis-ready DataFrame keyed by daily_route_id.
    """
    daily_routes = tables["daily_routes"].copy()
    # daily_routes = build_season_lookup(
    #     daily_routes,
    #     tables["year_parameters"],
    #     tables["operating_years"]
    # )
    daily_routes = build_holiday_flag(daily_routes, tables["tags"])

    daily_routes["day"] = pd.to_datetime(daily_routes["day"])
    daily_routes["day_of_week"] = daily_routes["day"].dt.day_name()
    daily_routes["month"] = daily_routes["day"].dt.month
    daily_routes["year"] = daily_routes["day"].dt.year
    daily_routes["total_booked_seats"] = (
        daily_routes["booked_group_seats"]
        + daily_routes["booked_individual_seats"]
    )
    daily_routes["total_capacity"] = (
        daily_routes["group_capacity"]
        + daily_routes["individual_capacity"]
        + daily_routes["reserve_capacity"]
    )
    return daily_routes
