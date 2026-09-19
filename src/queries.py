"""
Extraction queries for building analysis-ready DailyRoutes demand tables.
"""

# --- Core demand signal (capacity/context side) -------------------------

def time_window_filter(column_name: str) -> str:
    return f"""
WHERE EXTRACT(YEAR FROM "{column_name}") >= EXTRACT(YEAR FROM CURRENT_DATE) - :lookback_years
AND "{column_name}" <= CURRENT_DATE - INTERVAL '1 DAY'
"""

DAILY_ROUTES = f"""
SELECT
    "Id"                    AS daily_route_id,
    "Day"                   AS day,
    "TrainNumber"           AS train_number,
    "DepartureStation"      AS departure_station,
    "DepartureTime"         AS departure_time,
    "IsSpecialRoute"        AS is_special_route,
    "IsEnabled"             AS is_enabled,
    "YearParametersId"      AS year_parameters_id,
    "GroupCapacity"         AS group_capacity,
    "IndividualCapacity"    AS individual_capacity,
    "ReserveCapacity"       AS reserve_capacity,
    "BookedGroupSeats"      AS booked_group_seats,
    "BookedIndividualSeats" AS booked_individual_seats,
    "MinutesOfDelay"        AS minutes_of_delay,
    "BoatScheduledTime"     AS boat_scheduled_time,
    "CreatedDate"           AS created_date
FROM "DailyRoutes"
{time_window_filter("Day")}
"""

# --- Capacity / operating-year context ---------------------------------

YEAR_PARAMETERS = """
SELECT
    "Id"                          AS year_parameters_id,
    "OperatingYearId"             AS operating_year_id,
    "Type"                        AS parameter_type,
    "StartingDate"                AS starting_date,
    "EndingDate"                  AS ending_date,
    "DefaultGroupCapacity"        AS default_group_capacity,
    "DefaultIndividualCapacity"   AS default_individual_capacity,
    "DefaultReserveCapacity"      AS default_reserve_capacity,
    "FifteenMinuteCapacityLimit"  AS fifteen_minute_capacity_limit,
    "ThreeHourCapacityLimit"      AS three_hour_capacity_limit
FROM "YearParameters"
"""

# --- Holiday / special-period proxy --------------------------------------

TIMETABLE_ROUTES = """
SELECT
    "Id"                        AS timetable_route_id,
    "TrainNumber"               AS train_number,
    "StartingDate"              AS starting_date,
    "EndingDate"                AS ending_date,
    "DepartureStation"          AS departure_station,
    "TagId"                     AS tag_id,
    "YearParametersId"          AS year_parameters_id,
    "DefaultGroupCapacity"      AS default_group_capacity,
    "DefaultIndividualCapacity" AS default_individual_capacity,
    "DefaultReserveCapacity"    AS default_reserve_capacity
FROM "TimetableRoutes"
"""

# --- Bookings: now the PRIMARY demand signal -----------------------------

BOOKINGS = f"""
SELECT
    "Id"                                   AS booking_id,
    "Details_Date"                         AS travel_date,
    "Details_Status"                       AS status,
    "Details_NumberOfGuests"               AS number_of_guests,
    "Details_SingleWay"                    AS single_way,
    "Details_NoTransportation"             AS no_transportation,
    "Details_Boat"                         AS is_boat,
    "Details_Bus"                          AS is_bus,
    "OriginatingTransportationId"          AS originating_transportation_id,
    "ReturnTransportationId"               AS return_transportation_id,
    "History_CreatedDate"                  AS created_date,
    "Type"                                 AS booking_type,
    "IndividualBookingType"                AS individual_booking_type
FROM "Bookings"
{time_window_filter("Details_Date")}
"""

BOOKING_TRANSPORTATIONS = """
SELECT
    "Id"                AS transportation_id,
    "DailyRouteId"       AS daily_route_id,
    "DepartureTime"      AS departure_time,
    "DepartureStation"   AS departure_station,
    "ArrivalStation"     AS arrival_station,
    "Direction"          AS direction
FROM "BookingTransportations"
"""

ALL_QUERIES = {
    "daily_routes": DAILY_ROUTES,
    "year_parameters": YEAR_PARAMETERS,
    "timetable_routes": TIMETABLE_ROUTES,
    "bookings": BOOKINGS,
    "booking_transportations": BOOKING_TRANSPORTATIONS,
}

# Queries that take a :lookback_years parameter (the rest are small reference
# tables, pulled in full).
LOOKBACK_PARAMETERIZED = {"daily_routes", "bookings"}
