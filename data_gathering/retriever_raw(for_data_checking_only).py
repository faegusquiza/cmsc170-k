import io
import time
import pandas as pd
import requests
def main():
    # Processes the data for merra2
    data_file_raw_merra = pd.read_csv("CS170 DATA LOCATIONS - MERRA2.csv", header=None)

    i = 0
    # groups regions int a list of dictionaries, which have the name
    # along with the latitude and longitude of the grids associated with that region
    regions_data_merra = []
    while i < len(data_file_raw_merra.columns):
        temp_dict = {}
        temp_dict["name"] = data_file_raw_merra.iloc[0, i]
        temp_dict["lat"] = data_file_raw_merra.iloc[2:, i+1].dropna().to_list()
        temp_dict["long"] = data_file_raw_merra.iloc[2:, i+2].dropna().to_list()
        regions_data_merra.append(temp_dict)
        i += 3

    # Processes the data for syn1
    data_file_raw_syn = pd.read_csv("CS170 DATA LOCATIONS - SYN1.csv", header=None)

    i = 0
    # groups regions int a list of dictionaries, which have the name
    # along with the latitude and longitude of the grids associated with that region
    regions_data_syn = []
    while i < len(data_file_raw_syn.columns):
        temp_dict = {}
        temp_dict["name"] = data_file_raw_syn.iloc[0, i]
        temp_dict["lat"] = data_file_raw_syn.iloc[2:, i+1].dropna().to_list()
        temp_dict["long"] = data_file_raw_syn.iloc[2:, i+2].dropna().to_list()
        regions_data_syn.append(temp_dict)
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
    for region_merra, region_syn in zip(regions_data_merra, regions_data_syn):
        parameters = [
            "T2M",
            "PRECTOTCORR",
            "WS10M"
        ]
        regional_dataset_list = []
        num_region_datapoints_merra = len(region_merra["lat"])

        #This part is for retrieving data that the merra2 grid provide
        for idx, params in enumerate(parameters):
            param_df = None
            for i in range(num_region_datapoints_merra):
                print(region_merra["lat"][i], " ,", region_merra["long"][i]) 
                query_params = {
                    "latitude": region_merra["lat"][i],
                    "longitude": region_merra["long"][i],
                    "parameters": params,
                    "community": community,
                    "start": start_date,
                    "end": end_date,
                    "header": "false", 
                    "format": "CSV",
                }
                response = requests.get(url, params=query_params)
                

                if response.status_code == 200:
                    param_df = pd.read_csv(io.StringIO(response.text))
                    output_filename = f"RAW_{region_merra['name']}_{params}_at_{region_merra['lat'][i]}_{region_merra['long'][i]}.csv"
                    param_df.to_csv(output_filename, index=False)
                else:
                    print(f"Error fetching {params} for region {region_merra['name']} at ({region_merra['lat'][i]}, {region_merra['long'][i]}): {response.text}")
                    return
                print(param_df)
                time.sleep(1) # API rate limit

        #This part will be for retrieving data for the syn1 grid
        num_region_datapoints_syn = len(region_syn["lat"])
        param_df = None
        for i in range(num_region_datapoints_syn):
            print(region_syn["lat"][i], " ,", region_syn["long"][i]) 
            query_params = {
                "latitude": region_syn["lat"][i],
                "longitude": region_syn["long"][i],
                "parameters": "ALLSKY_SFC_SW_DWN",
                "community": community,
                "start": start_date,
                "end": end_date,
                "header": "false", 
                "format": "CSV",
            }
            response = requests.get(url, params=query_params)


            if response.status_code == 200:
                param_df = pd.read_csv(io.StringIO(response.text))
                output_filename = f"RAW_{region_syn['name']}_ALLSKY_SFC_SW_DWN_at_{region_syn['lat'][i]}_{region_syn['long'][i]}.csv"
                param_df.to_csv(output_filename, index=False)

            else:
                print(f"Error fetching ALLSKY_SFC_SW_DWN for region {region_syn['name']} at ({region_syn['lat'][i]}, {region_syn['long'][i]}): {response.text}")
                return
            print(param_df)
            time.sleep(1) # API rate limit

main()



