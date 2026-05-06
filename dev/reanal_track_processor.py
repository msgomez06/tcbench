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

import argparse

# %% Load the seasons to process
emulate = True
if emulate:
    sys.argv = [
        "reanal_track_processor.py",
        "--season",
        "2017",
    ]


# Read in the arguments
parser = argparse.ArgumentParser(description="Process the reanalysis data for the TCs")
parser.add_argument(
    "--season",
    type=int,
    help="The seasons to process",
    default=2019,
)

args = parser.parse_args()

# %%
target_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/ilia/"
seasons = toolbox.get_TC_seasons(
    season_list=[args.season],
    datadir_path=target_dir,
)
# seasons = toolbox.get_TC_seasons(
#     season_list=[2023],  # *range(2019, 2020)],
#     datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
# )

# %%
# get the list of files in the target directory for each season
calculated_storms = []
season_dir = os.path.join(target_dir, str(args.season))
if os.path.exists(season_dir):
    files = os.listdir(season_dir)
    for file in files:
        if file.endswith(".nc"):
            storm_id = file.split(".")[0]
            calculated_storms.append(storm_id)

# %% Control flags
process = True

# %% Process the tracks
input_samples = None
target_samples = None
for season, storms in seasons.items():
    print(
        f"Starting to process {season}. which contains {len(storms)} storms...",
        flush=True,
    )

    if process:
        # Load the data collection
        data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/"
        dc = dlib.Data_Collection(data_dir)

        # determine the number of processors that can be used
        # n_jobs = jl.cpu_count()
        n_jobs = 2

        for storm in storms:
            if storm.uid in calculated_storms:
                print(f"Skipping {str(storm)} as it has already been processed.")
                continue
            try:
                storm.process_data_collection(
                    dc,
                    reanal_variables=[
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure",
                        # "temperature",
                        # "geopotential",
                    ],
                    masktype="rect",
                    circum_points=5 * 4,
                    # plevels={"temperature": [850], "geopotential": [500]},
                    verbose=True,
                    n_jobs=4,
                )
            except Exception as e:
                print(f"Failed to process {str(storm)}")
                print(f"Error: {e}")

        # # process the tracks
        # jl.Parallel(n_jobs=n_jobs, backend="threading")(
        #     jl.delayed(storm.process_data_collection)(
        #         dc,
        #         reanal_variables=[
        #             "10m_u_component_of_wind",
        #             "10m_v_component_of_wind",
        #             "mean_sea_level_pressure",
        #             "temperature",
        #             "geopotential",
        #         ],
        #         plevels={"temperature": [850], "geopotential": [500]},
        #         masktype="rect",
        #         circum_points=30 * 4,
        #         n_jobs=n_jobs,
        #         verbose=False,
        #     )
        #     for storm in storms[6:]
        # )

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
