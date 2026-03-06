import cdsapi
import xarray as xr
import multiprocessing as mp
from pathlib import Path
import os

dictionary = [
    ["cordex_eur_11", "historical", "euro_cordex", "1970-2005"],
    ["cordex_eur_11", "rcp_4_5", "euro_cordex", "2006-2100"],
    ["cordex_eur_11", "rcp_8_5", "euro_cordex", "2006-2100"],
]


def download_data_era5(variable, origin, domain, period):
    dataset = "multi-origin-c3s-atlas"
    request = {
        "origin": origin,
        "domain": domain,
        "period": period,
        "variable": variable,
        "bias_adjustment": "no_bias_adjustment",
    }

    client = cdsapi.Client()
    target = f"{variable}_{origin}_{domain}_{period}.nc"
    # print(f"Downloading {target}...")
    client.retrieve(dataset, request, target)
    print(f"{variable}_{origin}_{domain}_{period}.nc")


def download_data(variable, origin, experiment, domain, period):
    dataset = "multi-origin-c3s-atlas"
    request = {
        "origin": origin,
        "experiment": experiment,
        "domain": domain,
        "period": period,
        "variable": variable,
        "bias_adjustment": "no_bias_adjustment",
    }

    apikey = '82a6b233-71db-4887-b976-0ce3f290e99c' # Enter your personal API key
    
    client = cdsapi.Client(key=f"{apikey}", url="https://cds.climate.copernicus.eu/api")
    
    target = f"{variable}_{origin}_{experiment}_{domain}_{period}.nc"
    # print(f"Downloading {target}...")
    client.retrieve(dataset, request, target)
    print(f"{variable}_{origin}_{experiment}_{domain}_{period}.nc")


if __name__ == "__main__":

    pool = mp.Pool(processes=12)

    variables = [
        "monthly_maximum_1_day_precipitation",
    ]

    for vs in variables:
        new_dictionary = [[vs] + entry for entry in dictionary]
        pool.starmap(download_data, new_dictionary)
        print(f"Finished downloading {vs} data.")