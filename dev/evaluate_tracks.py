# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib
import metrics

# %%

ibtracs = full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)

ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == 2023]

# %%
eval_folder = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/TCBench Results"
track_files = os.listdir(eval_folder)
track_files = [f for f in track_files if f.endswith(".csv")]

# %%
for track_file in track_files:
    print(f"Working on {track_file}...")

    # skip if results in name
    if "results" in track_file:
        print("Skipping...")
        continue

    # Read the track file
    track_df = pd.read_csv(os.path.join(eval_folder, track_file))

    # Check which unique SIDS in track_df are present in ibtracs
    # and remove ones that are not present
    track_df = track_df[track_df["SID"].isin(ibtracs["SID"].unique())]

    # Check if the ensemble dimension is present
    probabilistic = False
    for col in track_df.columns:
        if "ensemble" in col:
            ensemble_col = col
            probabilistic = True
            break

    # clean up the track_df by removing duplicates for the same time and valid time
    # for each storm ID, but only if the track is not probabilistic
    if not probabilistic:
        # Remove duplicates for the same time and valid time
        track_df = track_df.drop_duplicates(
            subset=["Initial Time", "Valid Time", "SID"], keep="first"
        )


    if probabilistic:
        # Select one ensemble member to copy
        ensemble_idx = track_df.iloc[0][ensemble_col]
        result_df = track_df.loc[track_df[ensemble_col] == ensemble_idx].copy()
        # drop the ensemble column
        result_df = result_df.drop(columns=[ensemble_col])
    else:
        # Select the first ensemble member
        result_df = track_df.copy()

    error_metrics = [metrics.DPE, metrics.AE, metrics.SE]
    if probabilistic:
        error_metrics += [metrics.FCRPS, metrics.HCRPS]

    for metric in error_metrics:
        print(f"Calculating {metric}...")

        # Calculate the metric
        result = metric(
            reference=ibtracs,
            predictions=track_df,
        )

        ensemble_labels = None
        if probabilistic and len(result) == len(track_df):
            # make a dataframe with a column for the ensemble member
            # a column for each result label, a column for the init time and a column for the valid time

            ensemble_labels = pd.DataFrame(
                index=track_df.index,
                columns=[
                    ensemble_col,
                    *metric.return_labels,
                    "Initial Time",
                    "Valid Time",
                    "SID",
                ],
            )
            ensemble_labels.loc[:, ensemble_col] = track_df[ensemble_col]
            ensemble_labels.loc[:, "Initial Time"] = track_df["Initial Time"]
            ensemble_labels.loc[:, "Valid Time"] = track_df["Valid Time"]
            ensemble_labels.loc[:, "SID"] = track_df["SID"]
            if len(result.shape) == 1:
                # If the result is a 1D array, add it to the track_df
                ensemble_labels[metric.return_labels[0]] = result
            else:
                for idx, label in enumerate(metric.return_labels):
                    ensemble_labels[label] = result[:, idx]

            # group by init, valid time, & SID
            ensemble_labels = ensemble_labels.groupby(
                ["Initial Time", "Valid Time", "SID"]
            )

            mean_result = ensemble_labels.mean()
            std_result = ensemble_labels.std()
            # Where the standard deviation is 0, set it to NaN
            std_result[std_result == 0] = np.nan

            mean_result = mean_result.reset_index()
            std_result = std_result.reset_index()

            # add the mean and std to the result_df
            for idx, label in enumerate(metric.return_labels):
                result_df[label + "_mean"] = mean_result[label]
                result_df[label + "_std"] = std_result[label]

        # elif probabilistic and len(result) == len(result_df):
        #     if len(result.shape) == 1:
        #         # If the result is a 1D array, add it to the track_df
        #         track_df[metric.return_labels[0]] = result
        #     else:
        #         for idx, label in enumerate(metric.return_labels):
        #             track_df[label] = result[:, idx]

        else:
            if len(result.shape) == 1:
                # If the result is a 1D array, add it to the track_df
                result_df[metric.return_labels[0]] = result
            else:
                for idx, label in enumerate(metric.return_labels):
                    result_df[label] = result[:, idx]

    filename = track_file.split(".")[0]

    # Save the track_df
    result_df.to_csv(os.path.join(eval_folder, f"{filename}_results.csv"), index=False)

# %%
