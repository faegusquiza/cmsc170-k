import io
import time
import pandas as pd
import requests

# Processes the data, right now we only have merra2 csv
data_file_raw = pd.read_csv("CS170 DATA LOCATIONS - MERRA2.csv", header=None)

i = 0
# groups regions int a list of dictionaries, which have the name
# along with the latitude and longitude of the grids associated with that region
regions_data = []
while i < len(data_file_raw.columns):
    temp_dict = {}
    temp_dict["name"] = data_file_raw.iloc[0, i]
    temp_dict["lat"] = data_file_raw.iloc[2:, i+1].dropna().to_list()
    temp_dict["long"] = data_file_raw.iloc[2:, i+2].dropna().to_list()
    regions_data.append(temp_dict)
    i += 3

# This is just a list of what the api's columns returns
# We will use this to average the data_points across these columns
MONTH_AND_AVG = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']


# Initialization of the API
url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
start_date = "2001"
end_date = "2025"
community = "AG"

# The code for parsing the data that the API returns
# we take the averages of the data points in every region for a specific category
# and then, we will compile them in a single csv (not yet implemented, and may change depending on our needs)
for region in regions_data:
    parameters = [
        "T2M",
        "PRECTOTCORR",
        "WS10M"
    ]
    num_region_datapoints = len(region["lat"])
    for idx, params in enumerate(parameters):
        df = None
        for i in range(num_region_datapoints): 
            query_params = {
                "latitude": region["lat"][i],
                "longitude": region["long"][i],
                "parameters": params,
                "community": community,
                "start": start_date,
                "end": end_date,
                "header": "false", 
                "format": "CSV",
            }
            response = requests.get(url, params=query_params)
            

            if response.status_code == 200:
                if i == 0:
                    df = pd.read_csv(io.StringIO(response.text))
                    for month in MONTH_AND_AVG:
                        df[month] = df[month]/num_region_datapoints
                else:
                    temp_df = pd.read_csv(io.StringIO(response.text))
                    for month in MONTH_AND_AVG:
                        temp_df[month] = temp_df[month]/num_region_datapoints
                        df[month] += temp_df[month]
            else:
                print(f"Error fetching {params} for region {region['name']} at ({region['lat'][i]}, {region['long'][i]}): {response.text}")
            print(df)
        time.sleep(1) # API rate limit
        break   #Temporary break, for testing purposes and for not flooding the server or something
    break   #Temporary break, for testing purposes





