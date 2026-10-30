import io
import time
import pandas as pd
import requests


data_file_raw = pd.read_csv("CS170 DATA LOCATIONS - MERRA2.csv", header=None)

i = 0

regions_data = []

while i < len(data_file_raw.columns):
    temp_dict = {}
    temp_dict["region"] = data_file_raw.iloc[0, i]
    temp_dict["lat"] = data_file_raw.iloc[2:, i+1].dropna().to_list
    temp_dict["lon"] = data_file_raw.iloc[2:, i+2].dropna().to_list
    regions_data.append(temp_dict)
    # print(data_file_raw.iloc[2:, i].dropna().to_list())
    i += 3

