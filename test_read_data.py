from read_data import TIME_COLUMNS, read_data, read_file

def print_data():
    data = read_file()
    if not data:
        return

    headers = ["PARAMETER", "YEAR"] + TIME_COLUMNS + ["GRID_LAYER"]
    for region in data:
        print(f"\n{region}")
        print(",".join(headers)) 

        for record in data[region]:
            values = []
            for column in headers:
                values.append(str(record[column]))
            print(",".join(values))

    from read_data import read_data

    regions, climate_values, headers = read_data()

    print("\n------------------------------------------\n")
    print("Regions:", len(regions))
    print("Values per region:", len(headers))
    print("Column names:", headers)
    print("First region's values:", climate_values[0])

print_data()
