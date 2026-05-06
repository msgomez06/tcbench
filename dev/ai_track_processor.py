# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da

# Backend Libraries
import joblib as jl

from utils import toolbox, constants
from utils import data_lib as dlib

# %% Load the seasons to process

data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"

seasons = toolbox.get_TC_seasons(
    season_list=[*range(2023, 2024)],
    datadir_path=data_dir,
)

# %% Control flags

process = True
# perfect_prog = True

# %% Process the tracks
input_samples = None
target_samples = None
for season, storms in seasons.items():
    print(f"Starting to process {season}. which contains {len(storms)} storms...")

    if process:
        # vars_to_keep = (
        #     {
        #         "u10": [
        #             None,
        #             {
        #                 "units": "m s**-1",
        #                 "long_name": "10 metre U wind component",
        #             },
        #         ],
        #         "v10": [
        #             None,
        #             {
        #                 "units": "m s**-1",
        #                 "long_name": "10 metre V wind component",
        #             },
        #         ],
        #         "msl": [
        #             None,
        #             {
        #                 "units": "Pa",
        #                 "long_name": "Mean sea level pressure",
        #                 "standard_name": "air_pressure_at_sea_level",
        #             },
        #         ],
        #         "z": [
        #             [300, 500],
        #             {
        #                 "units": "m**2 s**-2",
        #                 "long_name": "Geopotential",
        #                 "standard_name": "geopotential",
        #             },
        #         ],
        #         "t": [
        #             [200, 850],
        #             {
        #                 "units": "K",
        #                 "long_name": "Temperature",
        #                 "standard_name": "air_temperature",
        #             },
        #         ],
        #         "q": [
        #             [300, 850],
        #             {
        #                 "units": "K",
        #                 "long_name": "Specific Humidity",
        #                 "standard_name": "specific_humidity",
        #             },
        #         ],
        #     },
        # )

        # Load the data collection
        # data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha/hugging/TCBench/neural_weather_models/panguweather"
        data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha/hugging/TCBench/neural_weather_models/fourcastnetv2_small"
        # data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
        dc = dlib.AI_Data_Collection(data_dir)

        # determine the number of processors that can be used
        n_jobs = jl.cpu_count()

        # data_dir = (
        #     "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
        # )
        # dc = dlib.AI_Data_Collection(data_dir)

        # process the tracks
        jl.Parallel(n_jobs=n_jobs)(
            jl.delayed(storm.process_data_collection)(dc, num_days=5)
            for storm in storms
        )

    else:
        # for storm in storms:
        #     inputs = None
        #     outputs = None
        #     try:
        #         inputs, outputs, t, leads = storm.serve_ai_data()
        #     except Exception as e:
        #         print(f"Failed to process {str(storm)}")
        #         print(f"Error: {e}")
        #     if inputs is not None and outputs is not None:
        #         if input_samples is None:
        #             input_samples = inputs
        #             target_samples = outputs
        #         else:
        #             input_samples = da.vstack((input_samples, inputs))
        #             target_samples = da.vstack((target_samples, outputs))
        pass

# %%
