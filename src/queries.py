"""
Extraction queries for Stage A (native-data-only) demand forecasting EDA.

Design choice: we pull each table close to raw (light filtering only) and
join everything in pandas rather than in one giant SQL query. For EDA this
is deliberate - it lets us inspect each join step and catch schema
surprises (e.g. sentinel values, unexpected nulls) before they're buried
inside a black-box query. Once the feature pipeline is finalized, the
joins can be pushed back into SQL for production efficiency.

All identifiers are double-quoted because the schema uses PascalCase
column/table names, which Postgres otherwise folds to lowercase.
"""

# --- Core demand signal -----------------------------------------------

DAILY_ROUTES = """
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
WHERE "Day" >= (CURRENT_DATE - (:lookback_days || ' days')::interval)
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

OPERATING_YEARS = """
SELECT
    "Id"                                AS operating_year_id,
    "CogwheelYearParametersId"          AS cogwheel_year_parameters_id,
    "CablecarGondolaYearParametersId"   AS cablecar_year_parameters_id,
    "SummerFacilityYearParametersId"    AS summer_facility_year_parameters_id,
    "IsActive"                          AS is_active,
    "PricingCatalogId"                  AS pricing_catalog_id
FROM "OperatingYears"
"""

# --- Holiday / special-period proxy --------------------------------------

TAGS = """
SELECT
    "Id"               AS tag_id,
    "Name"             AS tag_name,
    "Description"      AS tag_description,
    "StartingDate"     AS starting_date,
    "EndingDate"       AS ending_date,
    "YearParametersId" AS year_parameters_id,
    "IsForBoat"        AS is_for_boat
FROM "Tags"
"""

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

# --- Bookings (for lead time, cancellations, tour-operator effects) ------

BOOKINGS = """
SELECT
    "Id"                                   AS booking_id,
    "Details_Date"                         AS travel_date,
    "Details_Status"                       AS status,
    "Details_NumberOfGuests"               AS number_of_guests,
    "Details_SingleWay"                    AS single_way,
    "Details_Boat"                         AS is_boat,
    "Details_Bus"                          AS is_bus,
    "Details_NoTransportation"             AS no_transportation,
    "ContactInformation_TourOperatorId"    AS tour_operator_id,
    "History_CreatedDate"                  AS created_date,
    "Type"                                 AS booking_type,
    "IndividualBookingType"                AS individual_booking_type
FROM "Bookings"
WHERE "Details_Date" >= (CURRENT_DATE - (:lookback_days || ' days')::interval)
"""

OVERBOOKING_DETAILS = """
SELECT
    "GroupBookingId"      AS booking_id,
    "DateRequested"       AS date_requested,
    "DateAnswered"        AS date_answered,
    "OldCapacity"         AS old_capacity,
    "RequestedCapacity"   AS requested_capacity,
    "BergfahrtRouteDecision" AS uphill_decision,
    "TalfahrtRouteDecision"  AS downhill_decision
FROM "OverbookingDetails"
"""

ALL_QUERIES = {
    "daily_routes": DAILY_ROUTES,
    "year_parameters": YEAR_PARAMETERS,
    "operating_years": OPERATING_YEARS,
    "tags": TAGS,
    "timetable_routes": TIMETABLE_ROUTES,
    "bookings": BOOKINGS,
    "overbooking_details": OVERBOOKING_DETAILS,
}

# Queries that take a :lookback_days parameter (the rest are pulled in full,
# since they're small reference/dimension tables).
LOOKBACK_PARAMETERIZED = {"daily_routes", "bookings"}
