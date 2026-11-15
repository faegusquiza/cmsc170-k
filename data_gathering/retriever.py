"""
NASA POWER Regional Climate Data Extraction

Retrieves monthly climate data from the NASA POWER API for selected
Philippine regions using the MERRA-2 and SYN1DEG grid layers.

For each region, the script:
1. Loads the corresponding grid-point coordinates.
2. Retrieves climate data for each grid point.
3. Calculates the regional average for each climate parameter.
4. Combines the MERRA-2 and SYN1DEG datasets.
5. Exports the combined data into a regional CSV file.

Input:
    Coordinate CSV files containing the columns:
    'region', 'latitude', and 'longitude'.

Output:
    One CSV file per Philippine region, containing the averaged
    climate data from both grid layers.

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import io
import csv
import math
import requests
from pathlib import Path


# Directory paths for input coordinate files and exported regional datasets
OUTPUT_FOLDER = Path('regions')
GRID_COORDINATE_FOLDER = Path('grid_coordinates')

# Monthly columns returned by the NASA POWER API
# ANN represents the annual value
TIME_COLUMNS = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
    'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN'
]

RESULT_COLUMNS = ['PARAMETER', 'YEAR'] + TIME_COLUMNS

# Philippine regions included in the data extraction
REGIONS = [
    'REGION 1', 'REGION 2', 'CAR', 'REGION 3', 'NCR', 'REGION 4-A', 'MIMAROPA',
    'REGION 5', 'REGION 6', 'NIR', 'REGION 7', 'REGION 8', 'REGION 9', 'REGION 10',
    'REGION 11', 'REGION 12', 'REGION 13', 'BARMM'
]

# NASA POWER API configuration
NASA_POWER_API_URL = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
START_YEAR = '2001'
END_YEAR = '2025'
NASA_POWER_COMMUNITY = 'AG'
RESPONSE_FORMAT = 'CSV'

# Climate parameters retrieved from each NASA POWER grid layer
# MERRA-2: Near-surface temperature, corrected precipitation, and wind speed
# SYN1DEG: All-sky surface shortwave downward radiation
MERRA2_PARAMETERS = ['T2M', 'PRECTOTCORR', 'WS10M']
SYN1DEG_PARAMETERS = ['ALLSKY_SFC_SW_DWN']

def main():
    """
    Executes the regional climate data extraction workflow.

    Loads the grid coordinates for both NASA POWER grid layers,
    retrieves and averages their climate data per region, combines
    the resulting datasets, and exports one CSV file per region.
    """

    # Load and group grid-point coordinates by Philippine region
    syn1_coordinates = get_grid_coordinates_per_region(
        GRID_COORDINATE_FOLDER / 'syn1_coordinates_weighted.csv'
    )
    merra2_coordinates = get_grid_coordinates_per_region(
        GRID_COORDINATE_FOLDER / 'merra2_coordinates_weighted.csv'
    )

    # Create the output directory if it does not already exist.
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Process each Philippine region independently.
    for region in REGIONS:

        # Retrieve and calculate regional averages for the MERRA-2 grid layer
        print(f'[MERRA2 Grid Layer] Extracting {", ".join(MERRA2_PARAMETERS)} for {region}')
        merra2_data = fetch_regional_parameter_average(
            merra2_coordinates[region], MERRA2_PARAMETERS
        )
        merra2_data['GRID_LAYER'] = ["MERRA2"] *((int(END_YEAR) - int(START_YEAR) + 1)*len(MERRA2_PARAMETERS))

        # Retrieve and calculate regional averages for the SYN1DEG grid layer
        print(f'[SYN1DEG Grid Layer] Extracting {", ".join(SYN1DEG_PARAMETERS)} for {region}')
        syn1deg_data = fetch_regional_parameter_average(
            syn1_coordinates[region], SYN1DEG_PARAMETERS
        )
        syn1deg_data['GRID_LAYER'] = ["SYN1DEG"] *((int(END_YEAR) - int(START_YEAR) + 1)*len(SYN1DEG_PARAMETERS))

        # Vertically concatenate both datasets
        # The GRID_LAYER column identifies the source of each record
        regional_data = {}

        for column in merra2_data:
            regional_data[column] = merra2_data[column] + syn1deg_data[column]

        # Prepare CSV File data
        csv_headers = [list(regional_data.keys())]
        csv_rows = list(zip(*regional_data.values()))
        csv_content = csv_headers + csv_rows

        # Generate the output filename using the region name
        output_filename = region.lower().replace(" ", "_") + '.csv'
        output_filepath = OUTPUT_FOLDER / output_filename

        # Export the combined regional dataset without the DataFrame index
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)

        print(f'Successfully retrieved data for {region} to {output_filepath}')


def fetch_regional_parameter_average(grid_coordinates, parameters) -> dict[tuple]:
    """
    Retrieves climate data from NASA POWER and calculates the regional average.

    Sends an API request for each grid point, retrieves the requested
    climate parameters, and accumulates their monthly values. The
    accumulated values are divided by the sum of their weights to
    obtain the weighted regional average.

    Args:
        grid_coordinates: List of (latitude, longitude, weight) tuples
            representing the grid points belonging to a region.
        parameters: List of NASA POWER climate parameter codes to retrieve.

    Returns:
        dict: Dictionary of the the averaged climate data
            across all grid points in the region where the keys are
            the column names.

    Raises:
        RuntimeError: If a NASA POWER API request fails.
        ValueError: If the retrieved data contains a -999 missing-data
            placeholder in any of the expected time columns.
    """

    # Initialize a dictionary to accumulate weighted grid-point climate data.
    regional_average_df = {}

    # Count the total number of grid points to process.
    grid_point_count = len(grid_coordinates)
    total_weight = 0.0
    if grid_point_count == 0:
        raise ValueError('Region have no grid points.')    

    for grid_index, (latitude, longitude, weight) in enumerate(grid_coordinates):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid weight {weight} at ({latitude}, {longitude})')

        total_weight += weight

        # Construct the API request using the current grid point
        # and the requested climate parameters.
        query_params = {
            'latitude': latitude,
            'longitude': longitude,
            'parameters': ','.join(parameters),
            'community': NASA_POWER_COMMUNITY,
            'start': START_YEAR,
            'end': END_YEAR,
            'header': 'false',
            'format': RESPONSE_FORMAT,
        }

        # Send the request to NASA POWER.
        response = requests.get(NASA_POWER_API_URL, params=query_params)

        # Stop processing if the API request was unsuccessful.
        if not response.ok:
            raise RuntimeError(f'Grid {str(grid_index + 1).zfill(2)}/{grid_point_count}: Error fetching data at ({latitude}, {longitude})')

        response_dict = {columns: [] for columns in RESULT_COLUMNS}


        response_csv = csv.DictReader(io.StringIO(response.text))
        # Convert the API's CSV response into a dict of results
        for row in response_csv :
            for columns in response_dict:
                if columns in TIME_COLUMNS:
                    response_dict[columns].append(float(row[columns]))
                else:
                    response_dict[columns].append(row[columns])
        


        # Check for NASA POWER's -999 placeholder, which indicates missing data.
        for time_cols in TIME_COLUMNS:
            if -999 in response_dict[time_cols]:
                raise ValueError(f"Grid {str(grid_index + 1).zfill(2)}/{grid_point_count}: Missing data (-999 placeholder found) at ({latitude}, {longitude})")
                
        print(f'\tGrid {str(grid_index + 1).zfill(2)}/{grid_point_count}: Data retrieved at ({latitude}, {longitude})')

        # Preserve parameter/year labels and initialize weighted time values.
        if len(regional_average_df) == 0:
            regional_average_df = response_dict.copy()
            for time_cols in TIME_COLUMNS:
                regional_average_df[time_cols] = [
                    value * weight for value in response_dict[time_cols]
                ]

        # Add this grid point's weighted values to the accumulated values.
        else:
            for time_cols in TIME_COLUMNS:
                for i in range(len(response_dict[time_cols])):
                    regional_average_df[time_cols][i] += response_dict[time_cols][i] * weight

    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError('The sum of grid-point weights must be positive and finite.')

    #Normalize weights for floating point errors
    for time_cols in TIME_COLUMNS:
        regional_average_df[time_cols] = [
            value / total_weight for value in regional_average_df[time_cols]
        ]

    return regional_average_df


def get_grid_coordinates_per_region(filepath_or_buffer) -> dict[str, list[tuple[float, float, float]]]:
    """
    Reads grid coordinates from a CSV file and groups them by region.

    Each region is mapped to a list of coordinate tuples, where each
    tuple contains the latitude and longitude of a grid point.

    Args:
        filepath_or_buffer: Path to the CSV file or a file-like object
            containing the grid coordinates. The CSV must have the
            columns 'region', 'latitude', and 'longitude'.

    Returns:
        dict[str, list[tuple[float, float]]]: A dictionary where each key
        is a region name and each value is a list of (latitude, longitude)
        tuples.

    Example:
        {
            'REGION 1': [
                (18.5, 120.5),
                (18.5, 121.25)
            ],
            'REGION 2': [
                (21.0, 122.0),
                (20.5, 122.0)
            ]
        }
    """

    grid_coordinates_per_region = {region: [] for region in REGIONS}
    # Read the coordinate CSV file into a DataFrame.
    with open(filepath_or_buffer, mode='r', newline='', encoding='utf-8') as file:
        grid_coordinates = csv.DictReader(file)
    
        for row in grid_coordinates:
            region = row['region'].strip()
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            weight = float(row['weight'])

            # Only include coordinates for recognized regions.
            if region in grid_coordinates_per_region:
                grid_coordinates_per_region[region].append((latitude, longitude, weight))


    return grid_coordinates_per_region


if __name__ == '__main__':
    main()