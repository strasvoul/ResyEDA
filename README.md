# resy demand-forecasting EDA

Exploratory data analysis of seat demand for the `resy` schema (Postgres). Stage A: native DB data only.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in DB host/port/name/user/password (read-only user preferred)
```

## Run

```bash
python run_eda.py             # pulls from DB on first run, then reads cache in data/raw/*.parquet
python run_eda.py --refresh   # force a fresh DB pull
python -m src.db              # test the DB connection only
```

Plots are saved to `outputs/plots/`.

## Configuration

[src/constants.py](src/constants.py) holds:
- `DEFAULT_LOOKBACK_YEARS` - how many seasons back to pull `daily_routes` and `bookings`.
- `NON_RESERVED_STATUS_CODES` - booking statuses excluded from derived demand (4 cancelled, 5 no-show, 7 cancelled and refunded).
- Booking type codes and the individual booking types excluded from lead-time analysis.

## Modules

- `src/db.py` - DB connection; credentials from `.env`.
- `src/queries.py` - one SQL query per table.
- `src/extract.py` - runs queries, caches to parquet.
- `src/enrich.py` - sentinel flagging, holiday flag from `tags`, demand derived from bookings (`bookings` -> `booking_transportations` -> `daily_routes`, one row per leg), derived-vs-stored comparison, lead time per booking.
- `src/eda.py` - coverage, closures and calendar gaps, target distributions, time series, capacity breach preview, derived-vs-stored demand, lead time, pickup curves (group / individual / per type / mixed).

## Notes

- `Bookings` has no direct link to `DailyRoutes`; the join goes through `booking_transportations` (originating and return transportation IDs).
- Bookings with `Details_NoTransportation = True` are dropped from derived demand.
- `IsSpecialRoute` and `IsEnabled` rows are kept in the raw pull and not excluded automatically.

## Testing without the DB

`tests/make_synthetic_tables.py` builds synthetic tables to run the pipeline offline. Correlation between derived and stored demand is near zero on this data by design.
