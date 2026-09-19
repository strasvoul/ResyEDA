"""
Exploratory Data Analysis (EDA) functions for the daily routes dataset.
Includes volume & coverage checks, sentinel/closure checks, and target distribution reports.
"""
from __future__ import annotations

from pathlib import Path
from sre_constants import IN
from typing import List

import matplotlib
matplotlib.use("Agg")  # safe for headless / non-interactive runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.constants import BOOKING_TYPE_GROUP, BOOKING_TYPE_INDIVIDUAL, INDIVIDUAL_BOOKING_TYPE_STR, INDIVIDUAL_BOOKING_TYPES_TO_EXCLUDE

sns.set_theme(style="whitegrid")

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["booked_group_seats", "booked_individual_seats"]


# ---------------------------------------------------------------------------
# 1. Volume & coverage
# ---------------------------------------------------------------------------

def volume_and_coverage(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("train_number")
        .agg(
            n_rows=("daily_route_id", "count"),
            first_day=("day", "min"),
            last_day=("day", "max"),
            n_special_route_rows=("is_special_route", "sum"),
            n_disabled_rows=("is_enabled", lambda s: (~s).sum()),
        )
        .reset_index()
    )
    summary["span_days"] = (summary["last_day"] - summary["first_day"]).dt.days
    summary["coverage_ratio"] = summary["n_rows"] / summary["span_days"].clip(lower=1)

    cold_start = summary[summary["n_rows"] < 60]
    print(f"[coverage] {len(summary)} distinct train numbers")
    print(f"[coverage] {len(cold_start)} routes have < 60 rows total (cold-start candidates):")
    if len(cold_start):
        print(cold_start[["train_number", "n_rows", "first_day", "last_day"]].to_string(index=False))

    return summary


# ---------------------------------------------------------------------------
# 2. Sentinel / closure check
# ---------------------------------------------------------------------------

def closure_report(df: pd.DataFrame) -> None:
    n_disabled = int((~df["is_enabled"]).sum())
    print(f"[data quality] {n_disabled:,} rows with is_enabled = False")
    print(
        "[data quality] Recommendation: exclude sentinel + disabled rows from "
        "modeling, and treat calendar gaps as 'not operating' rather than "
        "'zero demand' - see calendar_gap_report()."
    )


def calendar_gap_report(df: pd.DataFrame) -> pd.DataFrame:
    gap_rows = []
    routes = df[df["is_special_route"] == False]
    for train_number, group in routes.sort_values("day").groupby("train_number"):
        days = group["day"].drop_duplicates().sort_values()
        prev_days = days.shift(1)
        diffs = (days - prev_days).dt.days
        gaps = diffs[diffs > 1]
        for idx, gap_len in gaps.items():
            gap_end = days.loc[idx]
            gap_start = prev_days.loc[idx]
            gap_rows.append(
                {
                    "train_number": train_number,
                    "gap_start_after": gap_start,
                    "gap_end_before": gap_end,
                    "gap_length_days": int(gap_len) - 1,
                }
            )
    gaps_df = pd.DataFrame(gap_rows)
    if gaps_df.empty:
        print("[calendar] no gaps found")
    else:
        long_gaps = gaps_df[gaps_df["gap_length_days"] > 14]
        print(f"[calendar] {len(gaps_df)} gaps found; {len(long_gaps)} longer than 14 days (likely seasonal closures)")
        if len(long_gaps) != len(gaps_df):
            print(gaps_df.sort_values(["gap_length_days"], ascending=False))
    return gaps_df


# ---------------------------------------------------------------------------
# 3. Target distributions
# ---------------------------------------------------------------------------

def target_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in TARGET_COLS:
        s = df[col].dropna()
        stats.append(
            {
                "target": col,
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "skew": s.skew(),
                "pct_zero": (s == 0).mean(),
                "max": s.max(),
            }
        )
    stats_df = pd.DataFrame(stats)
    print("[targets] distribution summary:")
    print(stats_df.to_string(index=False))

    fig, axes = plt.subplots(len(TARGET_COLS), 2, figsize=(11, 4 * len(TARGET_COLS)))
    if len(TARGET_COLS) == 1:
        axes = axes.reshape(1, 2)
    for i, col in enumerate(TARGET_COLS):
        sns.histplot(df[col].dropna(), bins=40, ax=axes[i, 0])
        axes[i, 0].set_title(f"{col} (raw)")
        sns.histplot(np.log1p(df[col].dropna()), bins=40, ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} (log1p)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_histograms.png", dpi=120)
    plt.close(fig)

    return stats_df


# ---------------------------------------------------------------------------
# 4. Time series structure
# ---------------------------------------------------------------------------

def time_series_plots(df: pd.DataFrame, top_n_routes: int = 3) -> None:
    """Plots total demand over time on a CONTINUOUS daily calendar (reindexed,
    gaps left as NaN / breaks in the line) rather than letting matplotlib draw
    a straight line across missing dates - a previous run's chart showed a
    misleading diagonal exactly because of this."""
    daily_total = df.groupby("day")[TARGET_COLS].sum().sum(axis=1)
    full_range = pd.date_range(daily_total.index.min(), daily_total.index.max(), freq="D")
    daily_total = daily_total.reindex(full_range)  # gaps become NaN -> visible break, not a lie

    fig, ax = plt.subplots(figsize=(12, 4))
    daily_total.plot(ax=ax)
    ax.set_title("Total booked seats per day (all routes) - gaps shown as breaks, not interpolated")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "total_demand_over_time.png", dpi=120)
    plt.close(fig)

    applicable_routes = df[df["is_special_route"] == False]
    busiest = applicable_routes["train_number"].value_counts().head(top_n_routes).index
    for train_number in busiest:
        route_df = applicable_routes[applicable_routes["train_number"] == train_number].sort_values("day")
        series = route_df.set_index("day")["booked_group_seats"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
        series.plot(ax=axes[0], title=f"Train {train_number}: booked_group_seats over time")

        clean = series.dropna()
        if len(clean) > 30:
            plot_acf(clean, ax=axes[1], lags=min(30, len(clean) // 2 - 1))
            plot_pacf(clean, ax=axes[2], lags=min(30, len(clean) // 2 - 1))
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"train_{train_number}_acf_pacf.png", dpi=120)
        plt.close(fig)

    print(f"[time series] saved overview + ACF/PACF plots for routes: {list(busiest)}")


# ---------------------------------------------------------------------------
# 5. Capacity-breach preview
# ---------------------------------------------------------------------------

def capacity_breach_preview(df: pd.DataFrame, group_only: bool = True) -> pd.Series:
    if group_only:
        breach = df["booked_group_seats"] > df["group_capacity"]
    else:
        breach = df["total_booked_seats"] > df["total_capacity"]

    rate = breach.mean()
    print(f"[capacity breach] positive class rate: {rate:.2%} ({breach.sum():,} / {len(breach):,} rows)")
    if rate < 0.05:
        print(
            "[capacity breach] WARNING: strongly imbalanced - plan for "
            "stratified splits, class weighting, or resampling later."
        )
    return breach


# ---------------------------------------------------------------------------
# 6. NEW: bookings-derived demand vs. stored-counter cross-check
# ---------------------------------------------------------------------------

def demand_source_comparison_report(comparison_df: pd.DataFrame) -> None:
    """comparison_df comes from enrich.compare_derived_vs_stored()."""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=comparison_df, x="derived_booked_guests", y="total_booked_seats", alpha=0.3, ax=ax)
    lims = [0, max(comparison_df["derived_booked_guests"].max(), comparison_df["total_booked_seats"].max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Derived from Bookings (guests)")
    ax.set_ylabel("Stored on DailyRoutes (seats)")
    ax.set_title("Demand: bookings-derived vs. DailyRoutes-stored")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "demand_source_comparison.png", dpi=120)
    plt.close(fig)
    print("[demand comparison] plot saved - large/systematic off-diagonal spread means "
          "the two sources disagree and one shouldn't be trusted blindly.")

# ---------------------------------------------------------------------------
# 7. NEW: booking lead-time distribution
# ---------------------------------------------------------------------------

def booking_lead_time_by_type_report(bookings_with_lead_time: pd.DataFrame, individual_booking_types_to_exclude: List[int] = INDIVIDUAL_BOOKING_TYPES_TO_EXCLUDE) -> pd.DataFrame:
    """Splits the lead-time distribution by BookingType (Group vs
    Individual). The overall distribution is likely bimodal precisely
    because these two populations behave very differently - this checks
    that directly rather than assuming it."""

    rows = []
    for label, code in [("group", BOOKING_TYPE_GROUP), ("individual", BOOKING_TYPE_INDIVIDUAL)]:
        filter_options = (bookings_with_lead_time["booking_type"] == code) 
        if code == BOOKING_TYPE_INDIVIDUAL and "individual_booking_type" in bookings_with_lead_time.columns:
            filter_options &= (~bookings_with_lead_time["individual_booking_type"].isin(individual_booking_types_to_exclude))
        s = bookings_with_lead_time.loc[filter_options, ["lead_time_days", "individual_booking_type"]]

        if len(s) == 0:
            continue
        rows.append({
            "booking_type": label,
            "n": len(s),
            "median": s["lead_time_days"].median(),
            "p10": s["lead_time_days"].quantile(0.10),
            "p90": s["lead_time_days"].quantile(0.90),
            "pct_same_day": s["lead_time_days"].between(0, 1, inclusive="left").mean(),
        })
    stats_df = pd.DataFrame(rows)
    print("[lead_time by type]")
    print(stats_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, code in [("group", BOOKING_TYPE_GROUP), ("individual", BOOKING_TYPE_INDIVIDUAL)]:
        s = bookings_with_lead_time.loc[bookings_with_lead_time["booking_type"] == code, "lead_time_days"]
        if len(s):
            sns.kdeplot(s.clip(lower=0, upper=s.quantile(0.99)), label=label, ax=ax, fill=True, alpha=0.3)
    ax.set_title("Booking lead time by BookingType")
    ax.set_xlabel("Days before travel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "booking_lead_time_by_type.png", dpi=120)
    plt.close(fig)

    return stats_df

def booking_lead_time_for_individual(bookings_with_lead_time: pd.DataFrame) -> pd.DataFrame:
    """Computes lead time stats specifically for individual bookings, excluding certain types."""
    s = bookings_with_lead_time.loc[bookings_with_lead_time["booking_type"] == BOOKING_TYPE_INDIVIDUAL, ["lead_time_days", "individual_booking_type"]]
    individual_count_per_type = s["individual_booking_type"].value_counts(dropna=False).sort_index()
    print("[individual count per type]")
    print(individual_count_per_type)
    
    individual_with_payment = s.loc[s["individual_booking_type"] == 0]
    individual_partner = s.loc[s["individual_booking_type"] == 1]
    individual_time_tickets = s.loc[s["individual_booking_type"] == 2]
    individual_api = s.loc[s["individual_booking_type"] == 3]
    individual_not_defined = s.loc[~s["individual_booking_type"].isin([0, 1, 2, 3])]
    
    # calculate lead time stats for each individual booking type
    individual_stats = {}
    for label, df_subset in [
        ("with_payment", individual_with_payment),
        ("partner", individual_partner),
        ("time_tickets", individual_time_tickets),
        ("api", individual_api),
        ("not_defined", individual_not_defined),
    ]:
        if len(df_subset):
            individual_stats[label] = {
                "n": len(df_subset),
                "median": df_subset["lead_time_days"].median(),
                "p10": df_subset["lead_time_days"].quantile(0.10),
                "p90": df_subset["lead_time_days"].quantile(0.90),
                "pct_same_day": df_subset["lead_time_days"].between(0, 1, inclusive="left").mean(),
            }

    print("[individual lead time stats]")
    individual_stats_df = pd.DataFrame.from_dict(individual_stats, orient="index")
    print(individual_stats_df.to_string())
    
        
    # design a kernel density estimate (KDE) plot
    lead_time_min = s["lead_time_days"].min() - 25
    lead_time_max = s["lead_time_days"].max()
    fig, ax = plt.subplots()
    for label, df_subset in [
        ("with_payment", individual_with_payment),
        # ("partner", individual_partner),
        # ("time_tickets", individual_time_tickets),
        ("api", individual_api),
        # ("not_defined", individual_not_defined),
    ]:
        if len(df_subset):
            sns.kdeplot(df_subset["lead_time_days"], fill=True, alpha=0.5, label=label)
    ax.set_xlim(lead_time_min, lead_time_max)
    ax.set_xlabel("Lead Time Days")
    ax.set_ylabel("Frequency")
    ax.set_title("Lead Time Distribution for Individual Booking Types")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "individual_lead_time_distribution.png")
    plt.close()
    
    return individual_stats

def print_individual_stats_df(individual_rows_case2: List[pd.DataFrame]) -> None:
    """Prints individual booking stats in a readable format."""
    if not individual_rows_case2:
        print("[individual stats] No individual bookings to report.")
        return
    combined = pd.concat(individual_rows_case2, ignore_index=True)
    stats = {
        "booking_type": "individual (excl. tt)",
        "n": len(combined),
        "median": combined["lead_time_days"].median(),
        "p10": combined["lead_time_days"].quantile(0.10),
        "p90": combined["lead_time_days"].quantile(0.90),
        "pct_same_day": combined["lead_time_days"].between(0, 1, inclusive="left").mean(),
    }
    stats_df = pd.DataFrame([stats])
    print("[individual stats]")
    print(stats_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. NEW: pickup curve (fraction of final demand booked by N days out)
# ---------------------------------------------------------------------------

def _pickup_curve_report(
    bookings_with_lead_time: pd.DataFrame,
    snapshot_days: list[int] = [60, 45, 30, 21, 14, 7, 3, 1, 0],
) -> pd.DataFrame:
    """For each travel_date, computes what fraction of its FINAL total
    guest count had already been booked by each snapshot horizon (days
    before travel). Averages this fraction across travel dates to get a
    typical pickup curve - the standard revenue-management technique for
    this kind of business (airlines/hotels use the same idea)."""
    df = bookings_with_lead_time
    final_totals = df.groupby("travel_date")["number_of_guests"].sum().rename("final_total")

    rows = []
    for horizon in snapshot_days:
        booked_by_horizon = (
            df[df["lead_time_days"] >= horizon]
            .groupby("travel_date")["number_of_guests"]
            .sum()
        )
        merged = pd.concat([final_totals, booked_by_horizon.rename("booked_by_horizon")], axis=1).fillna(0)
        merged = merged[merged["final_total"] > 0]
        frac = (merged["booked_by_horizon"] / merged["final_total"]).mean()
        rows.append({"days_before_travel": horizon, "avg_fraction_booked": frac})

    curve = pd.DataFrame(rows).sort_values("days_before_travel", ascending=False)

    return curve

def _draw_pickup_curve(curve: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["days_before_travel"], curve["avg_fraction_booked"], marker="o")
    ax.invert_xaxis()  # far out on the left, travel day on the right
    ax.set_xlabel("Days before travel")
    ax.set_ylabel("Avg. fraction of final demand already booked")
    ax.set_title(f"Booking pickup curve ({name})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"pickup_curve_{name}.png", dpi=120)
    plt.close(fig)
    _print_pickup_curve(curve, name)
    
def _print_pickup_curve(curve: pd.DataFrame, name: str) -> None:
    print(f"Pickup curve for {name}:")
    print(curve)
    

def pickup_curve_report_by_type(
    bookings_with_lead_time: pd.DataFrame,
    snapshot_days: list[int] = [60, 45, 30, 21, 14, 7, 3, 1, 0],
) -> dict[str, pd.DataFrame]:
    """Computes pickup curves separately for each individual booking type using the 
    internal _pickup_curve_report function."""
    # curve for group bookings
    curves = {}
    curves["group"] = _pickup_curve_report(
        bookings_with_lead_time[bookings_with_lead_time["booking_type"] == 0],
        snapshot_days=snapshot_days,
    )
    _draw_pickup_curve(curves["group"], name="group")

    # curves for individual booking
    curves["individual"] = _pickup_curve_report(
            bookings_with_lead_time[bookings_with_lead_time["booking_type"] == 1],
            snapshot_days=snapshot_days,
        )
    _draw_pickup_curve(curves["individual"], name="individual")
    
    # curves for each individual booking type
    individual_booking_types: pd.Series = bookings_with_lead_time[bookings_with_lead_time["booking_type"] == 1]["individual_booking_type"]
    for ind_book_type in individual_booking_types.unique():
        type_name = INDIVIDUAL_BOOKING_TYPE_STR.get(float(ind_book_type) if isinstance(ind_book_type, float) and not np.isnan(ind_book_type) else np.nan)
        filter = (bookings_with_lead_time["individual_booking_type"] == ind_book_type) if isinstance(ind_book_type, float) and not np.isnan(ind_book_type) else (bookings_with_lead_time["individual_booking_type"].notna())
        df_type = bookings_with_lead_time[filter]
        curves[f"type_{type_name}"] = _pickup_curve_report(df_type, snapshot_days=snapshot_days)
        _draw_pickup_curve(curves[f"type_{type_name}"], name=f"type_{type_name}")
        
    # mixed curve 
    curves["mixed"] = _pickup_curve_report(
        bookings_with_lead_time,
        snapshot_days=snapshot_days,
    )
    _draw_pickup_curve(curves["mixed"], name="mixed")

    return curves

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame,
    bookings_with_lead_time: pd.DataFrame | None = None,
    demand_comparison_df: pd.DataFrame | None = None,
) -> None:
    print("=" * 70)
    print("1. VOLUME & COVERAGE")
    volume_and_coverage(df)

    print("\n" + "=" * 70)
    print("2. CALENDAR GAPS")
    closure_report(df)
    calendar_gap_report(df)

    print("\n" + "=" * 70)
    print("3. TARGET DISTRIBUTIONS")
    target_distribution_report(df)

    print("\n" + "=" * 70)
    print("4. TIME SERIES STRUCTURE")
    time_series_plots(df)

    print("\n" + "=" * 70)
    print("5. CAPACITY BREACH PREVIEW")
    capacity_breach_preview(df)

    if demand_comparison_df is not None:
        print("\n" + "=" * 70)
        print("6. BOOKINGS-DERIVED vs. STORED DEMAND")
        demand_source_comparison_report(demand_comparison_df)

    if bookings_with_lead_time is not None:
        print("\n" + "=" * 70)
        print("7. BOOKING LEAD TIME")
        booking_lead_time_for_individual(bookings_with_lead_time)
        booking_lead_time_by_type_report(bookings_with_lead_time, individual_booking_types_to_exclude=INDIVIDUAL_BOOKING_TYPES_TO_EXCLUDE)

        print("\n" + "=" * 70)
        print("8. PICKUP CURVE")
        pickup_curve_report_by_type(bookings_with_lead_time)

    print("\nAll plots saved to:", PLOTS_DIR)
