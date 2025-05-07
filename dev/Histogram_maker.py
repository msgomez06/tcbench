# %%
#  Imports
# OS and IO
import os
import sys
import pickle
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
from sklearn.preprocessing import StandardScaler
import torch
import dask
import multiprocessing
import json
import zarr as zr

# from dask import optimize
import time


# Backend Libraries
import joblib as jl

from utils import toolbox, ML_functions as mlf
import metrics, baselines

# Importing the sklearn metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import argparse


# Function to Normalize the data and targets
def transform_data(data, scaler):
    return scaler.transform(data)


# %%
if __name__ == "__main__":
    # emulate system arguments
    emulate = True
    # Simulate command line arguments
    if emulate:
        sys.argv = [
            "script_name",  # Traditionally the script name, but it's arbitrary in Jupyter
            # "--ai_model",
            # "fourcastnetv2",
            # "--overwrite_cache",
            # "--min_leadtime",
            # "6",
            # "--max_leadtime",
            # "24",
            # "--use_gpu",
            # "--verbose",
            # "--reanalysis",
            "--cache_dir",
            "/scratch/mgomezd1/cache",
            # "/srv/scratch/mgomezd1/cache",
            "--mask",
            "--magAngle_mode",
            "--dask_array",
        ]
    # check if the context has been set for torch multiprocessing
    if torch.multiprocessing.get_start_method() != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Read in arguments with argparse
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
        help="Directory where the data is stored",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch/mgomezd1/cache",
        help="Directory where the cache is stored",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/",
        help="Directory where the results are stored",
    )

    parser.add_argument(
        "--ignore_gpu",
        action="store_true",
        help="Whether to ignore the GPU for training",
    )

    parser.add_argument(
        "--ai_model",
        type=str,
        default="panguweather",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="deterministic",
    )

    parser.add_argument(
        "--deterministic_loss",
        type=str,
        default="RMSE",
    )

    parser.add_argument(
        "--overwrite_cache",
        help="Enable cache overwriting",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Enable verbose mode",
        action="store_true",
    )

    parser.add_argument(
        "--debug",
        help="Enable debug mode",
        action="store_true",
    )

    parser.add_argument(
        "--min_leadtime",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max_leadtime",
        type=int,
        default=168,
    )

    parser.add_argument(
        "--cnn_width",
        type=str,
        default="[32,64,128]",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--ablate_cols",
        type=str,
        default="[]",
    )

    parser.add_argument(
        "--fc_width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--magAngle_mode",
        action="store_true",
        help="Whether to use magnitude and angle instead of u and v",
    )

    parser.add_argument(
        "--raw_target",
        action="store_true",
        help="Whether to use intensity instead of intensification",
    )

    parser.add_argument(
        "--reanalysis",
        action="store_true",
        help="Whether to use reanalysis data instead of forecast data",
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Whether to apply a leadtime driven mask to the data",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/mask_dict.pkl",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="linear",
    )

    parser.add_argument(
        "--aux_loss",
        action="store_true",
        help="Whether to use an auxiliary loss",
    )

    parser.add_argument(
        "--dask_array",
        action="store_true",
        help="Whether to use dask arrays for the dataset",
    )

    parser.add_argument(
        "--search",
        action="store_true",
        help="Whether to perform a hyperparameter search",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="CNN",
    )

    parser.add_argument(
        "--l1_reg",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    print("Imports successful", flush=True)

    # print the multiprocessing start method
    print(torch.multiprocessing.get_start_method(allow_none=True))
    dask.config.set(scheduler="processes")

    #  Setup
    datadir = args.datadir
    cache_dir = args.cache_dir + f"_{args.ai_model}"
    result_dir = args.result_dir

    # Check for GPU availability
    if torch.cuda.is_available() and not args.ignore_gpu:
        calc_device = torch.device("cuda:0")
    else:
        calc_device = torch.device("cpu")

    num_cores = int(subprocess.check_output(["nproc"], text=True).strip())

    # %%
    # Check if the cache directory includes a zarray store for the data
    zarr_path = os.path.join(cache_dir, "zarray_store")
    if not os.path.exists(zarr_path):
        os.makedirs(zarr_path)
    zarr_store = zr.DirectoryStore(zarr_path)

    # Check if the root group exists, if not create it
    root = zr.group(zarr_store)
    root = zr.open_group(zarr_path)

    # Check that the training and validation groups exist, if not create them
    if "train" not in root:
        root.create_group("train")
    if "validation" not in root:
        root.create_group("validation")

    # Check if the data is already stored in the cache
    train_zarr = root["train"]
    valid_zarr = root["validation"]

    train_arrays = list(train_zarr.array_keys())
    valid_arrays = list(valid_zarr.array_keys())

    # assert that AIX_masked, target_scaled, base_intensity_scaled, base_position, leadtime_scaled are in each group
    assert np.all(
        [
            "masked_AIX" in train_arrays,
            "target_scaled" in train_arrays,
            "base_intensity_scaled" in train_arrays,
            "base_position" in train_arrays,
            "leadtime_scaled" in train_arrays,
            "leadtime" in train_arrays,
            "masked_AIX" in valid_arrays,
            "target_scaled" in valid_arrays,
            "base_intensity_scaled" in valid_arrays,
            "base_position" in valid_arrays,
            "leadtime_scaled" in valid_arrays,
            "leadtime" in valid_arrays,
        ]
    ), "Missing Data in the cache"
    # %%
    if args.dask_array:
        # Load the data from the zarr store using dask array
        if args.mask:
            train_data = da.from_zarr(zarr_store, component="train/masked_AIX")
            valid_data = da.from_zarr(zarr_store, component="validation/masked_AIX")
        else:
            train_data = da.from_zarr(zarr_store, component="train/AIX_scaled")
            valid_data = da.from_zarr(zarr_store, component="validation/AIX_scaled")
        train_target = da.from_zarr(zarr_store, component="train/target_scaled")
        valid_target = da.from_zarr(zarr_store, component="validation/target_scaled")
        train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime_scaled")
        validation_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime_scaled"
        )
        train_base_intensity = da.from_zarr(
            zarr_store, component="train/base_intensity_scaled"
        )
        valid_base_intensity = da.from_zarr(
            zarr_store, component="validation/base_intensity_scaled"
        )
        train_base_position = da.from_zarr(zarr_store, component="train/base_position")
        valid_base_position = da.from_zarr(
            zarr_store, component="validation/base_position"
        )
        train_unscaled_leadtimes = da.from_zarr(zarr_store, component="train/leadtime")
        validation_unscaled_leadtimes = da.from_zarr(
            zarr_store, component="validation/leadtime"
        )
    else:
        # load data with zarr
        if args.mask:
            train_data = train_zarr["masked_AIX"]
            valid_data = valid_zarr["masked_AIX"]
        else:
            train_data = train_zarr["AIX_scaled"]
            valid_data = valid_zarr["AIX_scaled"]
        train_target = train_zarr["target_scaled"]
        valid_target = valid_zarr["target_scaled"]
        train_leadtimes = train_zarr["leadtime_scaled"]
        validation_leadtimes = valid_zarr["leadtime_scaled"]
        train_base_intensity = train_zarr["base_intensity_scaled"]
        valid_base_intensity = valid_zarr["base_intensity_scaled"]
        train_base_position = train_zarr["base_position"]
        valid_base_position = valid_zarr["base_position"]
        train_unscaled_leadtimes = train_zarr["leadtime"]
        validation_unscaled_leadtimes = valid_zarr["leadtime"]
    # %%
    fit = False
    if fit:
        scaled_trainX = train_data
        scaled_validX = valid_data

        if args.magAngle_mode:
            feature_names = ["Magnitude", "Angle", "MSLP", "Z500", "T850"]
        else:
            feature_names = ["U10", "V10", "MSLP", "Z500", "T850"]

        #  Histogram creation
        # Start with creating a dictionary to store the histogram data
        histograms = {}
        edge_val = 4
        num_bins = 25
        for i in range(scaled_trainX.shape[1]):
            print(f"Processing feature {i+1} of {scaled_trainX.shape[1]}", flush=True)
            # Step 2: Create a Dask array from the scaled data
            train_hist, train_bin_edges = da.histogram(
                scaled_trainX[:, i], bins=num_bins, range=[-edge_val, edge_val]
            )

            # Compute the histogram (this triggers the computation)
            train_hist = train_hist.compute()
            # train_bin_edges = train_bin_edges.compute()

            # Step 2: Create a Dask array from the scaled data
            valid_hist, valid_bin_edges = da.histogram(
                scaled_validX[:, i], bins=num_bins, range=[-edge_val, edge_val]
            )

            # Compute the histogram (this triggers the computation)
            valid_hist = valid_hist.compute()
            valid_bin_edges = valid_bin_edges

            fig, ax = plt.subplots(1, figsize=(10, 6), dpi=150)
            ax.hist(
                train_bin_edges[:-1],
                train_bin_edges,
                weights=train_hist,
                label="Training Data",
                alpha=0.5,
                density=True,
            )
            ax.hist(
                valid_bin_edges[:-1],
                valid_bin_edges,
                weights=valid_hist,
                label="Validation Data",
                alpha=0.5,
                color="red",
                density=True,
            )
            ax.set_xlabel("Normalized Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram for feature {feature_names[i]} using Dask Array")
            ax.legend()

            plt.show()
            # Save the histogram data to the dictionary
            histograms[feature_names[i]] = {
                "train_hist": train_hist,
                "train_bin_edges": train_bin_edges,
                "valid_hist": valid_hist,
                "valid_bin_edges": valid_bin_edges,
            }
        # Save the histogram data
        with open(
            os.path.join(
                result_dir, f"histograms_{'mask' if args.mask else 'unmasked'}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(histograms, f)

        print("Histogram data saved successfully.")
        print("Histogram creation completed.", flush=True)
    else:
        # Load the histogram data from the pickle file
        with open(
            os.path.join(
                result_dir, f"histograms_{'mask' if args.mask else 'unmasked'}.pkl"
            ),
            "rb",
        ) as f:
            histograms = pickle.load(f)
    # %%
    # Print the loaded histogram data
    for feature_name, hist_data in histograms.items():
        print(f"Feature: {feature_name}")

        # calculate the bin centers
        train_bin_centers = 0.5 * (
            hist_data["train_bin_edges"][1:] + hist_data["train_bin_edges"][:-1]
        )
        valid_bin_centers = 0.5 * (
            hist_data["valid_bin_edges"][1:] + hist_data["valid_bin_edges"][:-1]
        )

        # make counts density
        train_density = hist_data["train_hist"] / np.sum(hist_data["train_hist"])
        valid_density = hist_data["valid_hist"] / np.sum(hist_data["valid_hist"])

        # find indices of the bins that are not empty
        non_empty_bins_train = np.where(train_density > 0)[0]
        non_empty_bins_valid = np.where(valid_density > 0)[0]

        print("Train Bin Centers: ")
        for center, count in zip(train_bin_centers, train_density):
            print(f"({center},{count}) ", end="")
        print("\nValid Bin Centers: ")
        for center, count in zip(valid_bin_centers, valid_density):
            print(f"({center},{count}) ", end="")
        print("\n")
# %%
