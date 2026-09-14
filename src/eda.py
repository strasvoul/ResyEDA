"""
EDA checks for the enriched demand table produced by src.enrich.build_demand_table.

Covers, in order:
  1. Volume & coverage        - which routes/seasons have enough history?
  2. Sentinel / closure check - separate "off-season" from "missing data"
  3. Target distributions     - histograms, skew, zero-inflation, log1p
  4. Time series structure    - trend/seasonality plots, ACF/PACF
  5. Capacity-breach preview  - class balance for the future L03 classifier

Each function both prints a short diagnostic to stdout AND (where relevant)
saves a plot to outputs/plots/, so this can run non-interactively.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / non-interactive runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid")

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["booked_group_seats", "booked_individual_seats"]


# ---------------------------------------------------------------------------
# 1. Volume & coverage
# ---------------------------------------------------------------------------

def volume_and_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts and date-range span per route. Flags routes with thin
    history ("cold start") that likely need pooling or a simpler model
    rather than an individually-trained sequence model."""
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

    cold_start = summary[summary["n_rows"] < 60]  # ~2 months of daily data
    print(f"[coverage] {len(summary)} distinct train numbers")
    print(f"[coverage] {len(cold_start)} routes have < 60 rows total (cold-start candidates):")
    if len(cold_start):
        print(cold_start[["train_number", "n_rows", "first_day", "last_day"]].to_string(index=False))

    return summary


# ---------------------------------------------------------------------------
# 2. Sentinel / closure check
# ---------------------------------------------------------------------------

def sentinel_and_closure_report(df: pd.DataFrame) -> None:
    n_sentinel = int(df["is_sentinel_departure_time"].sum() + df["is_sentinel_year_parameters"].sum())
    n_disabled = int((~df["is_enabled"]).sum())
    print(f"[data quality] {n_sentinel:,} sentinel-value rows flagged (see enrich.flag_sentinel_rows)")
    print(f"[data quality] {n_disabled:,} rows with is_enabled = False")
    print(
        "[data quality] Recommendation: exclude sentinel + disabled rows from "
        "modeling, and treat calendar gaps as 'not operating' rather than "
        "'zero demand' - see calendar_gap_report()."
    )


def calendar_gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """For each route, finds gaps in the daily calendar. A gap does NOT
    necessarily mean missing data - for a seasonal operation, most of the
    year is a genuine closure. This just surfaces the gaps so a human can
    classify them (e.g. cross-check against YearParameters.StartingDate/
    EndingDate) before deciding how to encode them for modeling."""
    gap_rows = []
    for train_number, group in df.sort_values("day").groupby("train_number"):
        days = group["day"].drop_duplicates().sort_values()
        diffs = days.diff().dt.days
        gaps = diffs[diffs > 1]
        for idx, gap_len in gaps.items():
            gap_end = days.loc[idx]
            gap_start = days.loc[idx - 1] if (idx - 1) in days.index else None
            gap_rows.append(
                {
                    "train_number": train_number,
                    "gap_start_after": gap_start,
                    "gap_end_before": gap_end,
                    "gap_length_days": int(gap_len) - 1,
                }
            )
    gaps_df = pd.DataFrame(gap_rows)
    if len(gaps_df):
        long_gaps = gaps_df[gaps_df["gap_length_days"] > 14]
        print(f"[calendar] {len(gaps_df)} gaps found; {len(long_gaps)} longer than 14 days (likely seasonal closures)")
    else:
        print("[calendar] no gaps found in the pulled date range")
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
    for i, col in enumerate(TARGET_COLS):
        sns.histplot(df[col].dropna(), bins=40, ax=axes[i, 0])
        axes[i, 0].set_title(f"{col} (raw)")

        sns.histplot(np.log1p(df[col].dropna()), bins=40, ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} (log1p)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_histograms.png", dpi=120)
    plt.close(fig)

    # if "season_type" in df.columns and df["season_type"].notna().any():
    #     fig, ax = plt.subplots(figsize=(9, 5))
    #     sns.boxplot(data=df, x="season_type", y="booked_group_seats", ax=ax)
    #     ax.set_title("booked_group_seats by season_type")
    #     fig.tight_layout()
    #     fig.savefig(PLOTS_DIR / "booked_group_seats_by_season.png", dpi=120)
    #     plt.close(fig)

    return stats_df


# ---------------------------------------------------------------------------
# 4. Time series structure
# ---------------------------------------------------------------------------

def time_series_plots(df: pd.DataFrame, top_n_routes: int = 3) -> None:
    """Plots the overall daily demand series, plus per-route series and
    ACF/PACF for the busiest routes (by row count)."""
    daily_total = df.groupby("day")[TARGET_COLS].sum().sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    daily_total.plot(ax=ax)
    ax.set_title("Total booked seats per day (all routes)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "total_demand_over_time.png", dpi=120)
    plt.close(fig)

    busiest = df["train_number"].value_counts().head(top_n_routes).index

    for train_number in busiest:
        route_df = df[df["train_number"] == train_number].sort_values("day")
        series = route_df.set_index("day")["booked_group_seats"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
        series.plot(ax=axes[0], title=f"Train {train_number}: booked_group_seats over time")

        # ACF/PACF need a reasonable number of non-null points
        clean = series.dropna()
        if len(clean) > 30:
            plot_acf(clean, ax=axes[1], lags=min(30, len(clean) // 2 - 1))
            plot_pacf(clean, ax=axes[2], lags=min(30, len(clean) // 2 - 1))
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"train_{train_number}_acf_pacf.png", dpi=120)
        plt.close(fig)

    print(f"[time series] saved overview + ACF/PACF plots for routes: {list(busiest)}")


# ---------------------------------------------------------------------------
# 5. Capacity-breach preview (for the downstream L03 classifier)
# ---------------------------------------------------------------------------

def capacity_breach_preview(df: pd.DataFrame, group_only: bool = True) -> pd.Series:
    """Previews class balance for 'will booked seats exceed capacity'.
    Kept simple here (group seats vs group capacity) - the real feature
    engineering for the classifier stage can extend this."""
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
# Orchestration
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. VOLUME & COVERAGE")
    volume_and_coverage(df)

    print("\n" + "=" * 70)
    print("2. SENTINELS & CALENDAR GAPS")
    # sentinel_and_closure_report(df)
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

    print("\nAll plots saved to:", PLOTS_DIR)
