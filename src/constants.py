# BookingType C# enum: internal enum BookingType { Group, Individual }
# Default numbering (no explicit values assigned) -> Group = 0, Individual = 1.


BOOKING_TYPE_GROUP = 0
BOOKING_TYPE_INDIVIDUAL = 1

DEFAULT_LOOKBACK_YEARS = 3 # Default number of years (seasons) to look back for data extraction


NON_RESERVED_STATUS_CODES = [
    4, # Cancelled
    5, # No-show
    7  # Cancelled and refunded
]

INDIVIDUAL_BOOKING_TYPES = [
    0,  # with payment
    1,  # from Partner
    2,  # time-ticket
    3   # from third-party
]

INDIVIDUAL_BOOKING_TYPES_TO_EXCLUDE = [
    1,  # from Partner
    2,  # time-ticket
]

INDIVIDUAL_BOOKING_TYPE_STR = {
    0.0: "with_payment",
    1.0: "from_Partner",
    2.0: "time_ticket",
    3.0: "from_API",
    None: "undefined_booking_type"
}