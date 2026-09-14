# resy demand-forecasting EDA

Stage A (native-data-only) exploratory data analysis for the seat-demand
forecasting project, built directly against the `resy` schema.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your real DB host/port/name/user/password
# (use a read-only DB user if at all possible)
```

## Run

```bash
python run_eda.py                    # first run: pulls from DB, caches to data/raw/*.parquet
python run_eda.py                    # subsequent runs: reads from cache, instant
python run_eda.py --refresh          # force a fresh pull from the DB
python run_eda.py --lookback-days 730
```

Test just the DB connection on its own:

```bash
python -m src.db
```

## What this does

1. **`src/db.py`** — connection, credentials from `.env` only, never hardcoded.
2. **`src/queries.py`** — one SQL query per source table (kept close to raw
   on purpose; joins happen in pandas so each step is inspectable).
3. **`src/extract.py`** — runs the queries, caches results as parquet in
   `data/raw/` so you're not re-hitting the DB on every EDA iteration.
4. **`src/enrich.py`** — the non-trivial joins:
   - `DailyRoutes` → `YearParameters` → `OperatingYears` → `PricingCatalogs`
     → `SeasonDefinitions` to attach a `season_type` per day (there's no
     direct FK from a route to a season, see the module docstring).
   - `Tags` → `is_tagged_period` flag as a holiday/special-period proxy.
   - Flags EF-Core sentinel values (`DepartureTime = 0001-01-01`,
     `YearParametersId` = all-zero UUID) so they don't get mistaken for
     real observations.
5. **`src/eda.py`** — the actual checks:
   - volume/coverage per route (flags cold-start routes)
   - sentinel + calendar-gap report (closure vs. missing data)
   - target histograms (raw + log1p), skew, zero-inflation, by season
   - time series plots + ACF/PACF for the busiest routes
   - a first look at capacity-breach class balance, for the downstream
     classification stage

Plots land in `outputs/plots/`. Everything prints a short diagnostic to
stdout as it runs, so you can `python run_eda.py | tee eda_log.txt` for a
record of the run.

## Validating the pipeline without touching the DB

`tests/make_synthetic_tables.py` builds fake tables shaped exactly like
`extract_all()`'s output (including a deliberately partial season mapping
and a seasonal closure gap), so you can sanity-check `enrich.py`/`eda.py`
logic before ever pointing it at the real database:

```bash
python -c "
from tests.make_synthetic_tables import tables
from src.enrich import build_demand_table
from src.eda import run_all
run_all(build_demand_table(tables))
"
```

This has already been run once during development (output showed everything
working correctly, including the season-match-rate and sentinel warnings
firing as expected). Feel free to delete `tests/` once you've validated
against your real DB — it has no role in the production pipeline.

## Notes / known caveats

- `enrich.build_season_lookup` prints a match rate — if it's low, the
  season-join assumption likely doesn't hold for some `YearParameters`
  rows and needs a closer look before this feeds a model.
- `IsSpecialRoute` and `IsEnabled` rows are included in the raw pull but
  **not** excluded automatically — decide deliberately whether to filter
  them before modeling (the coverage report flags how many there are).
- This is Stage A only (native schema data). Weather and public-holiday
  enrichment are a separate, later step per the Stage A/B comparison plan.
