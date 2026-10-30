import io
import time
import pandas as pd
import requests


data_file_raw = pd.read_csv("CS170 DATA LOCATIONS - MERRA2.csv", header=None)

i = 0

regions_data = []

while i < len(data_file_raw.columns):
    temp_dict = {}
    temp_dict["name"] = data_file_raw.iloc[0, i]
    temp_dict["lat"] = data_file_raw.iloc[2:, i+1].dropna().to_list()
    temp_dict["long"] = data_file_raw.iloc[2:, i+2].dropna().to_list()
    regions_data.append(temp_dict)
    # print(data_file_raw.iloc[2:, i].dropna().to_list())
    i += 3

MONTH_AND_AVG = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']

url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
start_date = "2001"
end_date = "2025"
community = "AG"

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
                    temp_df = pd.read_csv(io.StringIO(response.text)) #what the sigma sigma boy
                    for month in MONTH_AND_AVG:
                        temp_df[month] = temp_df[month]/num_region_datapoints
                        df[month] += temp_df[month]
            else:
                print(f"Error fetching {params} for region {region['name']} at ({region['lat'][i]}, {region['long'][i]}): {response.text}")
            print(df)
        time.sleep(1) # API rate limit
        break   #Temporary break, for testing purposes
    break   #Temporary break, for testing purposes





