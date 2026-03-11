# For month
month_map = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Generate month_rules for ConditionalMapper
month_rules = [(lambda x, k=k, v=v: x == k, int(v)) for k, v in month_map.items()]

# For day_of_week
dow_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}

# Generate dow_rules for ConditionalMapper
dow_rules = [(lambda x, v=v, k=k: x == k, int(v)) for k, v in dow_map.items()]
