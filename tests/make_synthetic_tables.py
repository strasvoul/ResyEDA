"""
NOT part of the deliverable pipeline. Generates fake tables shaped like
extract_all()'s output, purely so enrich.py / eda.py logic can be exercised
end-to-end without a live DB. Delete this file (and tests/) once you've
validated against your real database - it has no purpose after that.
"""
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_TRAINS = 4
DAYS = pd.date_range(date(2023, 3, 1), date(2024, 10, 31), freq="D")

year_params_ids = {yr: str(uuid.uuid4()) for yr in ["cogwheel_2023", "cogwheel_2024"]}
operating_year_id = str(uuid.uuid4())
pricing_catalog_id = str(uuid.uuid4())

# --- daily_routes ------------------------------------------------------
rows = []
for train in range(1, N_TRAINS + 1):
    for day in DAYS:
        # simulate a winter closure for one train, to test gap detection
        if train == 4 and 5 <= day.month <= 9:
            continue
        yp_id = year_params_ids["cogwheel_2023"] if day.year == 2023 else year_params_ids["cogwheel_2024"]
        base = 20 + 10 * np.sin(day.dayofyear / 365 * 2 * np.pi) + rng.normal(0, 5)
        rows.append(
            {
                "daily_route_id": str(uuid.uuid4()),
                "day": day,
                "train_number": train,
                "departure_station": 1,
                "departure_time": pd.Timestamp("1900-01-01 08:00:00") if rng.random() > 0.01 else pd.Timestamp("0001-01-01"),
                "is_special_route": False,
                "is_enabled": True,
                "year_parameters_id": yp_id,
                "group_capacity": 80,
                "individual_capacity": 40,
                "reserve_capacity": 10,
                "booked_group_seats": max(0, int(base)),
                "booked_individual_seats": max(0, int(base / 3 + rng.normal(0, 3))),
                "booked_spontaneous_seats": max(0, int(rng.poisson(2))),
                "minutes_of_delay": max(0, int(rng.normal(1, 2))),
                "boat_scheduled_time": pd.NaT,
                "created_date": pd.Timestamp("2021-01-01"),
            }
        )
daily_routes = pd.DataFrame(rows)

# --- year_parameters -----------------------------------------------------
year_parameters = pd.DataFrame(
    [
        {
            "year_parameters_id": year_params_ids["cogwheel_2023"],
            "operating_year_id": operating_year_id,
            "parameter_type": 0,
            "starting_date": pd.Timestamp("2023-01-01"),
            "ending_date": pd.Timestamp("2023-12-31"),
            "default_group_capacity": 80,
            "default_individual_capacity": 40,
            "default_reserve_capacity": 10,
            "fifteen_minute_capacity_limit": None,
            "three_hour_capacity_limit": None,
        },
        {
            "year_parameters_id": year_params_ids["cogwheel_2024"],
            "operating_year_id": operating_year_id,
            "parameter_type": 0,
            "starting_date": pd.Timestamp("2024-01-01"),
            "ending_date": pd.Timestamp("2024-12-31"),
            "default_group_capacity": 80,
            "default_individual_capacity": 40,
            "default_reserve_capacity": 10,
            "fifteen_minute_capacity_limit": None,
            "three_hour_capacity_limit": None,
        },
    ]
)

# --- operating_years -------------------------------------------------
operating_years = pd.DataFrame(
    [
        {
            "operating_year_id": operating_year_id,
            "cogwheel_year_parameters_id": year_params_ids["cogwheel_2023"],
            "cablecar_year_parameters_id": None,
            "summer_facility_year_parameters_id": None,
            "is_active": True,
            "pricing_catalog_id": pricing_catalog_id,
        }
    ]
)

# --- tags ----------------------------------------------------------------
tags = pd.DataFrame(
    [
        {
            "tag_id": str(uuid.uuid4()),
            "tag_name": "Christmas",
            "tag_description": "Christmas holidays",
            "starting_date": pd.Timestamp("2023-12-20"),
            "ending_date": pd.Timestamp("2024-01-05"),
            "year_parameters_id": year_params_ids["cogwheel_2023"],
            "is_for_boat": False,
        }
    ]
)

tables = {
    "daily_routes": daily_routes,
    "year_parameters": year_parameters,
    "operating_years": operating_years,
    "tags": tags,
}
