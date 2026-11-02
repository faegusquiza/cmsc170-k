"""
Regional Climate Data Processing Module

This module reads, validates, and processes regional climate data from CSV
files. It averages each calendar month across the selected years, then
prepares one separate dataset per month for clustering.

In each monthly dataset, a row represents one region and the four columns
contain precipitation, temperature, wind, and solar values. Each column can
be standardized across regions. ANN is kept in the original records but is
excluded from the monthly clustering datasets.

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import math
from pathlib import Path

DATA_FOLDER = "data_gathering/regions" # the folder containing the CSV files produced by retriever.py

# column names used in each regional CSV file
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
TIME_COLUMNS = MONTHS + ["ANN"]

def read_file(folder=DATA_FOLDER):
    """
    Reads and validates all regional CSV files in the specified folder.

    Each CSV file represents a region and contains monthly and annual
    climate measurements for different parameters, years, and grid layers.
    The function checks the file structure, data types, missing values,
    and duplicate records before storing the data.

    Args:
        folder (str or Path): Directory containing the regional CSV files.
            Defaults to DATA_FOLDER.

    Returns:
        dict: A dictionary where each key is a region name and its value
            is a list of climate records. Each record contains the
            parameter, year, grid layer, and monthly and annual values.
            Returns an empty dictionary if an error occurs or no valid
            regional CSV files are found.
    """

    data = {}
    folder = Path(folder) 

    if not folder.is_dir():
        print(f"Error: {folder} is not a folder.")
        return {}

    # sorts to make the files appear in the same order every time
    for file in sorted(folder.iterdir()):
        if not file.is_file() or file.suffix.lower() != ".csv":
            continue # skip subfolders and files that are not CSV files

        try:
            f = open(file, "r", newline="", encoding="utf-8")
        except OSError:
            print(f"Error: Cannot open {file.name}.")
            return {}

        # read the first line (column/header names)
        first_line = f.readline().strip()
        if first_line == "":
            print(f"Error: {file.name} is empty.")
            f.close()
            return {}

        headers = first_line.split(",")

        # list the columns we need before reading the data below them
        required = ["PARAMETER", "YEAR", "GRID_LAYER"]
        for column in TIME_COLUMNS:
            required.append(column)

        for column in required: # if required columns are present
            if column not in headers:
                print(f"Error: {file.name} is missing the {column} column.")
                f.close()
                return {}

        # convert the filename to a region name ("region_1.csv" to "REGION 1")
        region = file.name
        region = region[:-4] # remove the last four characters (.csv)
        region = region.replace("_", " ").upper()

        # each region should have only one CSV file
        if region in data:
            print(f"Error: More than one file has the region name {region}.")
            f.close()
            return {}

        records, previous_records = [], []
        line_number = 1

        # read each remaining file line and match its values to the column names
        for line in f:
            line_number += 1
            line = line.strip()
            if line == "":
                continue # skip empty lines

            # checks if the number of values matches the number of columns
            values = line.split(",")
            if len(values) != len(headers):
                print(f"Error: {file.name}, line {line_number} has the wrong number of values.")
                f.close()
                return {}

            # match each column name to its value in the current line
            row = {}
            for i in range(len(headers)):
                column = headers[i]
                row[column] = values[i]

            # remove extra spaces and check that the parameter and grid layer are not empty
            parameter = row["PARAMETER"].strip()
            layer = row["GRID_LAYER"].strip()
            if parameter == "" or layer == "":
                print(f"Error: {file.name}, line {line_number} has an empty parameter or grid layer.")
                f.close()
                return {}

            try:
                year = int(row["YEAR"])
            except (ValueError, TypeError):
                print(f"Error: {file.name}, line {line_number} has an invalid year.")
                f.close()
                return {}

            # one parameter should appear only once per year for the same grid layer
            record_id = [layer, parameter, year]
            if record_id in previous_records:
                print(f"Error: {file.name}, line {line_number} repeats {parameter} for {year} in {layer}.")
                f.close()
                return {}
            previous_records.append(record_id)

            record = {"PARAMETER": parameter, "YEAR": year, "GRID_LAYER": layer}
            for month in TIME_COLUMNS:
                try:
                    value = float(row[month])
                except (ValueError, TypeError):
                    print(f"Error: {file.name}, line {line_number} has an invalid value in {month}.")
                    f.close()
                    return {}

                # NASA uses -999 for missing data & isfinite() rejects infinity and non numbers
                if value == -999 or not math.isfinite(value):
                    print(f"Error: {file.name}, line {line_number} has a missing or invalid value in {month}.")
                    f.close()
                    return {}
                record[month] = value

            records.append(record) # keep the completed row in this region's list

        f.close() # finished reading this file

        if len(records) == 0:
            print(f"Error: {file.name} has column names but no climate records.")
            return {}
        data[region] = records

    if len(data) == 0:
        print(f"Error: No regional CSV files found in {folder}.")
    return data # dictionary of regions and its climate records


def regional_average(data, start_year=2001, end_year=2025, time_columns=None):
    """
    Calculates the average climate measurements for each region across
    the selected year range.

    The function calculates the average of each selected calendar month
    for every grid layer and parameter. For example, JAN temperature is
    averaged across all selected years, separately for each region.
    Annual values can still be selected explicitly with ["ANN"].

    Args:
        data (dict): Dictionary of regional climate records returned by
            read_file().
        start_year (int): First year included in the calculation.
            Defaults to 2001.
        end_year (int): Last year included in the calculation.
            Defaults to 2025.
        time_columns (list or None): Time columns to include. None selects
            JAN to DEC. Use ["JAN"] to get four January values per region.

    Returns:
        tuple: A tuple containing:
            - regions (list): Sorted list of region names.
            - avg (list): Regional averages, with each row containing
              the selected climate measurements for one region.
            - headers (list): Column names corresponding to the
              calculated averages.

            Returns three empty lists if an error occurs.
    """
    
    if not data or start_year > end_year:
        print("Error: Provide climate data and a valid year range.")
        return [], [], []

    if time_columns is None: # use JAN to DEC by default
        time_columns = MONTHS

    if len(time_columns) == 0:
        print("Error: Select at least one monthly or annual column.")
        return [], [], []

    selected_columns = []
    for column in time_columns:
        if column not in TIME_COLUMNS or column in selected_columns:
            print(f"Error: {column} is an unknown or repeated time column.")
            return [], [], []
        selected_columns.append(column)

    # find the parameters present in the selected years
    parameters = [] # stores each different grid layer and parameter pair

    # goes through each region and its climate records to get grid layer and parameter pairs
    for region in data: 
        for record in data[region]:
            if start_year <= record["YEAR"] <= end_year:
                pair = [record["GRID_LAYER"], record["PARAMETER"]] # e.g. ["MERRA2", "T2M"]

                if pair not in parameters: # check whether this pair has already been added
                    parameters.append(pair)
    parameters = sorted(parameters)

    if len(parameters) == 0:
        print("Error: No records found within the selected years.")
        return [], [], []

    regions = sorted(data)
    avg, headers = [], [] # regional averages and their column names
    num_years = end_year - start_year + 1

    for layer, parameter in parameters: # each grid layer and parameter pair
        for month in time_columns: # each monthly or annual column
            headers.append(f"{layer}_{parameter}_{month}") # e.g. MERRA2_T2M_JAN

    # calculates the averages for one region at a time
    for region in regions: 
        values = []

        for layer, parameter in parameters:
            totals = [0.0] * len(time_columns) # initialize with zero

            for year in range(start_year, end_year + 1): 
                found = False # no matching record has been found for this year yet

                for record in data[region]:
                    # match the grid layer, parameter, and year
                    if record["GRID_LAYER"] == layer and record["PARAMETER"] == parameter and record["YEAR"] == year:
                        for j in range(len(time_columns)): # go through each time column
                            totals[j] += record[time_columns[j]] # add this column's value to the column's total
                        found = True # matching record found
                        break

                if not found:
                    print(f"Error: {region} is missing {parameter} for {year} in {layer}.") 
                    return [], [], []

            # goes through the monthly or annual totals
            for total in totals: 
                values.append(total / num_years)
        avg.append(values) # add this region's averages to the list for all regions

    return regions, avg, headers

def standardize_data(climate_values):
    """
    Standardizes climate measurements using Z-score normalization.

    Each column is standardized independently using its mean and
    population standard deviation across all regions. This places
    measurements on a comparable scale before calculating distances
    between regions.

    Args:
        climate_values (list): A two-dimensional list of climate
            measurements, where each row represents a region and
            each column represents one parameter's value for a selected month.

    Returns:
        list: A two-dimensional list containing the standardized
            climate measurements. Columns with a standard deviation
            of zero are assigned 0.0 for all regions.
            Returns an empty list if the input data is invalid.
    """

    # # check if the data or the first region's list is empty
    if not climate_values or not climate_values[0]: 
        print("Error: No climate values to standardize.")
        return []

    num_regions = len(climate_values)
    num_columns = len(climate_values[0])

    # checks each region's list before doing the calculations
    for row in climate_values: 
        if len(row) != num_columns:
            print("Error: All regions must have the same number of climate values.")
            return []

        # checks each climate value's validity in this region
        for value in row: 
            try:
                valid = math.isfinite(value) 
            except TypeError: 
                valid = False

            if not valid:
                print("Error: Climate values must be valid numbers.")
                return []

    means, deviations = [], [] 
    for j in range(num_columns):
        total = 0.0 

        for i in range(num_regions): # get this column's value from every region
            total += climate_values[i][j]

        mean = total / num_regions # average this column across all regions
        means.append(mean) 

        # variance measures how spread out the column's values are around their average
        total = 0.0

        for i in range(num_regions): # compare each region's value with the column's average
            total += (climate_values[i][j] - mean) ** 2 # sum((value - mean)^2)

        deviations.append((total / num_regions) ** 0.5) # sqrt(sum((value - mean)^2) / number of regions)

    scaled = []
    for i in range(num_regions): # standardize one region's values at a time
        values = []
        for j in range(num_columns):
            if deviations[j] == 0: # this column has the same value for every region
                values.append(0.0) 
            else:
                values.append((climate_values[i][j] - means[j]) / deviations[j])
        scaled.append(values)

    return scaled

# function that future clustering methods can import
def read_data(folder=DATA_FOLDER, start_year=2001, end_year=2025, time_columns=None, standardize=True):
    """
    Reads and prepares the selected time columns for clustering.

    This function combines the CSV reading, regional averaging, and
    optional standardization steps into one process. For one monthly run,
    use time_columns=["JAN"], ["FEB"], etc. Use read_monthly_data() to
    prepare all 12 separate monthly datasets at once.

    Args:
        folder (str or Path): Directory containing the regional CSV files.
            Defaults to DATA_FOLDER.
        start_year (int): First year included in the calculation.
            Defaults to 2001.
        end_year (int): Last year included in the calculation.
            Defaults to 2025.
        time_columns (list or None): Monthly or annual columns to include.
            Defaults to None, which selects all months.
            This default returns 48 columns together; ["JAN"] returns four.
        standardize (bool): Determines whether to apply Z-score
            standardization to the calculated climate measurements.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - regions (list): Sorted list of region names.
            - climate_values (list): Processed climate measurements,
              where each row represents a region.
            - headers (list): Column names corresponding to the
              climate measurements.

            Returns three empty lists if an error occurs.
    """
    
    data = read_file(folder)
    if not data:
        return [], [], []

    regions, climate_values, headers = regional_average(data, start_year, end_year, time_columns)
    if not climate_values:
        return [], [], []

    if standardize:
        climate_values = standardize_data(climate_values)
        if not climate_values:
            return [], [], []

    return regions, climate_values, headers


# one separate dataset for each calendar month
def read_monthly_data(folder=DATA_FOLDER, start_year=2001, end_year=2025, standardize=True):
    """
    Prepares 12 separate monthly datasets using the saved regional records.

    Each month's values are averaged across the selected years. In the
    current dataset, each month contains 18 regions and four parameters.
    Standardization is calculated separately for each month's four columns.
    Year-to-year fluctuations are averaged out; ANN is not included.

    Args:
        folder (str or Path): Folder containing the regional CSV files.
        start_year (int): First year included. Defaults to 2001.
        end_year (int): Last year included. Defaults to 2025.
        standardize (bool): True returns standardized values. False returns
            the monthly averages in their original units.

    Returns:
        tuple: Region names, a dictionary of monthly values, and a dictionary
            of monthly column names. For example, monthly_values["JAN"][0]
            contains the four January values for regions[0].
            Returns ([], {}, {}) if reading or preparation fails.
    """
    data = read_file(folder) # read the original CSV files for all 12 months
    if not data:
        return [], {}, {}

    monthly_values, monthly_headers = {}, {}
    regions = []
    for month in MONTHS:
        regions, values, headers = regional_average(data, start_year, end_year, [month])
        if not values:
            return [], {}, {} # stop if any selected year or parameter is missing

        if standardize:
            values = standardize_data(values) # compare regions using this month's column averages and deviations
            if not values:
                return [], {}, {}

        monthly_values[month] = values # each month gets its own list of regional values
        monthly_headers[month] = headers # e.g. MERRA2_T2M_JAN

    return regions, monthly_values, monthly_headers
