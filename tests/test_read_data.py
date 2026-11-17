﻿from utilities.data_reader import MONTHS, TIME_COLUMNS, read_file, read_monthly_data

# prints the original records first then the prepared monthly datasets
# this is only for testing
def print_data():
    data = read_file()
    if not data:
        return

    headers = ["PARAMETER", "YEAR"] + TIME_COLUMNS + ["GRID_LAYER"]
    for region in data:
        print(f"\n{region}")
        print(",".join(headers))

    regions, monthly_values, monthly_headers, monthly_averages = read_monthly_data()
    if not monthly_values:
        return # the reader already printed the error

    print("\n------------------------------------------\n")
    print("Regions:", len(regions))
    print("Monthly datasets:", len(monthly_values))

    for month in MONTHS:
        print(f"\n{month}")
        print("Parameters per region:", len(monthly_headers[month]))
        print("Column names:", monthly_headers[month])
        print("First region:", regions[0])

print_data()
