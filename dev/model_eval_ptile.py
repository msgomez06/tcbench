# %% Important TODO: Separate preprocessing script from training script
# This should be easily doable now with on disk data caching.

# %%
# Imports
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
import json
import zarr as zr
from itertools import cycle
import shap
import gc
import pandas as pd


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
            # "panguweather",
            # "fourcastnetv2",
            # "--overwrite_cache",
            # "--min_leadtime",
            # "6",
            # "--max_leadtime",
            # "24",
            # "--use_gpu",
            # "--verbose",
            # "--reanalysis",
            # "--mode",
            # "probabilistic",
            "--cache_dir",
            "/scratch/mgomezd1/cache",
            # "/srv/scratch/mgomezd1/cache",
            "--magAngle_mode",
            "--dask_array",
            "--tag",
            # "ptile-unmasked"
            "ptile_unet"
            # "pangu_validation-reval",
            # "pangu_test-reval_revision",
            # "fcast_test-reval",
            # "fcast_valid-reval",
            # "Pangu_validation-reval",
            # "ERA5_validation-reval",
            # "ERA5_test-reval",
            # "--shap",
            "--test-set",
            "--report-global",
            "--batch_size",
            "32",
            "--mask_ptile",
            "84",
        ]
    # %%
    # check if the context has been set for torch multiprocessing
    if torch.multiprocessing.get_start_method() != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Read in arguments with argparse
    parser = argparse.ArgumentParser(description="Train an MLR model")
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
        default="MSE",
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
        "--dropout",
        type=float,
        default=0.25,
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
        "--tag",
        type=str,
        default="",
        help="Tag for the model",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Whether to use SHAP for the model",
    )
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Use the test set instead of the validation set",
    )
    parser.add_argument(
        "--report-global",
        action="store_true",
        help="Whether to report global (all leadtime) metrics",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for deep evaluation",
    )
    parser.add_argument(
        "--mask_ptile",
        type=int,
        default=84,
    )


    args = parser.parse_args()

    print("Imports successful", flush=True)

    # print the multiprocessing start method
    print(torch.multiprocessing.get_start_method(allow_none=True))
    dask.config.set(scheduler="processes")

    #  Setup
    datadir = args.datadir
    cache_dir = os.path.join(
        args.cache_dir, f"_{args.ai_model}" if not args.reanalysis else "ERA5"
    )
    result_dir = args.result_dir
    if args.mask_ptile != 84:
        cache_dir = cache_dir+ f"_{args.mask_ptile}ptile"
        result_dir = result_dir + f"_{args.mask_ptile}ptile"

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    target_scaler = None
    fpath = os.path.join(cache_dir, "target_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading target scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            target_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Target scaler not found at {fpath}")


    source_set = "test" if args.test_set else "validation"

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
    if source_set not in root:
        root.create_group(source_set)

    # Check if the data is already stored in the cache
    train_zarr = root["train"]
    valid_zarr = root[source_set]

    train_arrays = list(train_zarr.array_keys())
    valid_arrays = list(valid_zarr.array_keys())
    # %%

    # assert that AIX_masked, target_scaled, base_intensity_scaled, base_position, leadtime_scaled are in each group
    assert np.all(
        [
            # "masked_AIX" in valid_arrays,
            "target_scaled" in valid_arrays,
            "base_intensity_scaled" in valid_arrays,
            "base_position" in valid_arrays,
            "leadtime_scaled" in valid_arrays,
            "leadtime" in valid_arrays,
        ]
    ), "Missing Data in the cache"

    if args.dask_array:
        # Load the data from the zarr store using dask array
        valid_data = da.from_zarr(zarr_store, component=f"{source_set}/masked_AIX")
        unmasked_valid_data = da.from_zarr(
            zarr_store, component=f"{source_set}/AIX_scaled"
        )
        valid_target = da.from_zarr(zarr_store, component=f"{source_set}/target_scaled")
        validation_leadtimes = da.from_zarr(
            zarr_store, component=f"{source_set}/leadtime_scaled"
        )

        valid_base_intensity = da.from_zarr(
            zarr_store, component=f"{source_set}/base_intensity_scaled"
        )
        valid_base_position = da.from_zarr(
            zarr_store, component=f"{source_set}/base_position"
        )
        validation_unscaled_leadtimes = da.from_zarr(
            zarr_store, component=f"{source_set}/leadtime"
        )
        # Load the training targets and unscaled leadtimes for climatology
        train_target = da.from_zarr(zarr_store, component="train/target_scaled")
        train_leadtimes = da.from_zarr(zarr_store, component="train/leadtime")
    else:
        # load data with zarr
        valid_data = valid_zarr["masked_AIX"]
        unmasked_valid_data = valid_zarr["AIX_scaled"]
        valid_target = valid_zarr["target_scaled"]
        validation_leadtimes = valid_zarr["leadtime_scaled"]
        valid_base_intensity = valid_zarr["base_intensity_scaled"]
        valid_base_position = valid_zarr["base_position"]
        validation_unscaled_leadtimes = valid_zarr["leadtime"]
        # Load the training targets and unscaled leadtimes for climatology
        train_target = train_zarr["target_scaled"]
        train_leadtimes = train_zarr["leadtime"]

    # %%
    # # Defining the models to load and the hyperparameters needed to evaluate them
    # results_dir = (
    #     "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
    # )
    results_dir = result_dir

    result_file = f"{results_dir}model_comparison_eval_{args.tag}.pkl"

    if args.overwrite_cache or not os.path.exists(result_file):
        print("Calculating...", flush=True)
        if args.dask_array:
            valid_maxima = valid_data.max(axis=(-2, -1)).compute(scheduler="threads")
            valid_minima = valid_data.min(axis=(-2, -1)).compute(scheduler="threads")
            unmasked_valid_maxima = unmasked_valid_data.max(axis=(-2, -1)).compute(
                scheduler="threads"
            )
            unmasked_valid_minima = unmasked_valid_data.min(axis=(-2, -1)).compute(
                scheduler="threads"
            )
        else:
            valid_maxima = np.max(valid_data, axis=(-2, -1))
            valid_minima = np.min(valid_data, axis=(-2, -1))
            unmasked_valid_maxima = np.max(unmasked_valid_data, axis=(-2, -1))
            unmasked_valid_minima = np.min(unmasked_valid_data, axis=(-2, -1))

        valid_range = valid_maxima - valid_minima
        unmasked_valid_range = unmasked_valid_maxima - unmasked_valid_minima

        # -------------------------------
        if args.dask_array:
            validation_leadtimes = validation_leadtimes.compute(scheduler="threads")
            valid_base_intensity = valid_base_intensity.compute(scheduler="threads")
            validation_unscaled_leadtimes = validation_unscaled_leadtimes.compute(
                scheduler="threads"
            )
            unique_leadtimes = np.unique(validation_unscaled_leadtimes)
            valid_target = valid_target.compute(scheduler="threads")

            train_target = train_target.compute(scheduler="threads")
            train_leadtimes = train_leadtimes.compute(scheduler="threads")

            deep_scalars = np.hstack(
                [valid_base_intensity, valid_base_position, validation_leadtimes]
            ).compute(scheduler="threads")
        else:
            unique_leadtimes = np.unique(validation_unscaled_leadtimes)
            valid_target = valid_target

            train_target = train_target
            train_leadtimes = train_leadtimes
            deep_scalars = np.hstack(
                [valid_base_intensity, valid_base_position, validation_leadtimes]
            )
        # var order = ["W_mag", "W_dir", "mslp", "Z500", "T850"]

        # -------------------------------
        valid_x = np.vstack(
            [
                valid_maxima[:, 0],  # Maximum wind magnitude
                valid_minima[:, 2],  # Minimum mean sea level pressure
                valid_range[:, 0],  # Range of wind magnitude
                valid_range[:, 2],  # Range of mean sea level pressure
                valid_minima[:, 3],  # Minimum geopotential height at 500 hPa
                valid_range[:, 4],  # Range of temperature at 850 hPa
                validation_leadtimes.squeeze(),  # Leadtime
                valid_base_intensity.T,  # Base intensity
            ]
        ).T

        unmasked_valid_x = np.vstack(
            [
                unmasked_valid_maxima[:, 0],  # Maximum wind magnitude
                unmasked_valid_minima[:, 2],  # Minimum mean sea level pressure
                unmasked_valid_range[:, 0],  # Range of wind magnitude
                unmasked_valid_range[:, 2],  # Range of mean sea level pressure
                unmasked_valid_minima[:, 3],  # Minimum geopotential height at 500 hPa
                unmasked_valid_range[:, 4],  # Range of temperature at 850 hPa
                validation_leadtimes.squeeze(),  # Leadtime
                valid_base_intensity.T,  # Base intensity
            ]
        ).T
        print("Data prepared.", flush=True)
        # nan_idxs = np.isnan(valid_x).any(axis=1)
        # valid_data = valid_data[~nan_idxs]
        # unmasked_valid_data
        # deep_scalars = deep_scalars[~nan_idxs]
        # valid_x = valid_x[~nan_idxs]
        unmasked_valid_target = valid_target.copy()
        # valid_target = valid_target[~nan_idxs]
        # validation_unscaled_leadtimes = validation_unscaled_leadtimes[~nan_idxs]

        # unmasked_nan_idxs = np.isnan(unmasked_valid_x).any(axis=1)
        # unmasked_valid_data = unmasked_valid_data[~unmasked_nan_idxs]
        # unmasked_valid_x = unmasked_valid_x[~unmasked_nan_idxs]
        # unmasked_valid_target = unmasked_valid_target[~unmasked_nan_idxs]

        # nan filtering the validation set - TODO: investigate why ERA5 preprocessed data has nans in validation set
        if args.ai_model == "panguweather" and not args.reanalysis:
            if args.mask_ptile == 84:
                models = [
                    # --------------- Panguweather Linear Models ---------------
                        # {
                        #     "filepath": "SimpleANN_12-10-10h11_epoch-27_panguweather_probabilistic_masked.pt",
                        #     "masked": True,
                        #     "probabilistic": True,
                        #     "tag": "ANN 84 ptile",
                        #     "results": [],
                        #     "deep": False,
                        # },
                        # {
                        #     "filepath": "SimpleANN_12-13-20h08_epoch-17_panguweather_probabilistic_unmasked.pt",
                        #     "masked": False,
                        #     "probabilistic": True,
                        #     "tag": "ANN Unmasked",
                        #     "results": [],
                        #     "deep": False,
                        # },
                        #------------------------------
                        {
                            "filepath": "UNet_v2_06-18-20h11_epoch-6_panguweather-probabilistic.pt",
                            "masked": True,
                            "probabilistic": True,
                            "tag": "Masked UNet (Probabilistic)",
                            "results": [],
                            "deep": True,
                        },
                    ]
            elif args.mask_ptile == 50:
                models = [
                        {
                            "filepath": "SimpleANN_12-10-09h58_epoch-17_panguweather_probabilistic_masked.pt",
                            "masked": True,
                            "probabilistic": True,
                            "tag": "ANN 50 ptile",
                            "results": [],
                            "deep": False,
                        },

                    ]
            elif args.mask_ptile == 75:
                models = [
                        {
                            "filepath": "SimpleANN_12-10-09h50_epoch-27_panguweather_probabilistic_masked.pt",
                            "masked": True,
                            "probabilistic": True,
                            "tag": "ANN 75 ptile",
                            "results": [],
                            "deep": False,
                        },

                    ]
            elif args.mask_ptile == 95:
                models = [
                        {
                            "filepath": "SimpleANN_12-10-09h42_epoch-20_panguweather_probabilistic_masked.pt",
                            "masked": True,
                            "probabilistic": True,
                            "tag": "ANN 95 ptile",
                            "results": [],
                            "deep": False,
                        },

                    ]
            elif args.mask_ptile == 99:
                models = [
                        {
                            "filepath": "SimpleANN_12-10-11h02_epoch-18_panguweather_probabilistic_masked.pt",
                            "masked": True,
                            "probabilistic": True,
                            "tag": "ANN 99 ptile",
                            "results": [],
                            "deep": False,
                        },

                    ]
        else:
            raise NotImplementedError("Only Panguweather models are implemented.")
        # Load each pytorch model into the models dictionary
        for model in models:
            model["model"] = torch.load(
                os.path.join(results_dir, "torch_models", model["filepath"]),
                map_location=calc_device,
            )
            model["model"].eval()
            model["shap_values"] = {}
            model["shap_data"] = {}
            model["shap_features"] = {}
        # -------------------------------
        # Train the climatology model
        climatology_model = baselines.AveClimatology()
        climatology_model.fit(
            target=train_target,
            leadtimes=train_leadtimes.squeeze(),
        )

        # -------------------------------
        # Evaluate the models
        climatology_results = {"deterministic": [], "probabilistic": []}
        persistence_results = {"deterministic": [], "probabilistic": []}

        print("Baselines calculated...", flush=True)

        if args.report_global:
            if args.dask_array:
                valid_data = valid_data.compute(scheduler="threads")
                unmasked_valid_data = unmasked_valid_data.compute(scheduler="threads")
            for model in models:
                print(f"Evaluating model {model['tag']} on the validation set...")
                if model["probabilistic"]:
                    loss_fn = metrics.CRPS_ML
                else:
                    if args.deterministic_loss == "MSE":
                        loss_fn = torch.nn.MSELoss()
                    elif args.deterministic_loss == "MAE":
                        loss_fn = torch.nn.L1Loss()
                    else:
                        raise ValueError("Loss function not recognized.")

                if model["masked"]:
                    if not model["deep"]:
                        with torch.no_grad():
                            prediction = model["model"](
                                torch.tensor(valid_x, dtype=torch.float32).to(
                                    calc_device
                                )
                            )
                    else:
                        # Work by batches
                        batch_size = args.batch_size
                        len_data = valid_data.shape[0]
                        num_dims = len(valid_data.shape)

                        with torch.no_grad():
                            prediction_list = []
                            for i in range(0, len_data, batch_size):
                                temp_x = valid_data[i : i + batch_size]
                                temp_scalars = deep_scalars[i : i + batch_size]
                                if len(temp_x.shape) < num_dims:
                                    temp_x = np.expand_dims(temp_x, axis=0)
                                    temp_scalars = np.expand_dims(temp_scalars, axis=0)
                                prediction = model["model"](
                                    torch.tensor(
                                        temp_x,
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                    torch.tensor(
                                        temp_scalars,
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                )
                                prediction_list.append(
                                    prediction.cpu().detach().numpy()
                                )
                                # Simple progress bar
                                print(
                                    f"{i}/{len_data}",
                                    end="\r",
                                    flush=True,
                                )

                            prediction = np.vstack(prediction_list, dtype=np.float32)
                            prediction = torch.tensor(prediction).to(calc_device)
                else:
                    if not model["deep"]:
                        with torch.no_grad():
                            prediction = model["model"](
                                torch.tensor(unmasked_valid_x, dtype=torch.float32).to(
                                    calc_device
                                )
                            )
                    else:
                        # Work by batches
                        batch_size = args.batch_size
                        len_data = unmasked_valid_data.shape[0]
                        num_dims = len(unmasked_valid_data.shape)

                        with torch.no_grad():
                            prediction_list = []
                            for i in range(0, len_data, batch_size):
                                temp_x = unmasked_valid_data[i : i + batch_size]
                                temp_scalars = deep_scalars[i : i + batch_size]
                                if len(temp_x.shape) < num_dims:
                                    temp_x = np.expand_dims(temp_x, axis=0)
                                    temp_scalars = np.expand_dims(temp_scalars, axis=0)

                                prediction = model["model"](
                                    torch.tensor(
                                        temp_x,
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                    torch.tensor(
                                        temp_scalars,
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                )
                                prediction_list.append(
                                    prediction.cpu().detach().numpy()
                                )
                                # Simple progress bar
                                print(
                                    f"{i}/{len_data}",
                                    end="\r",
                                    flush=True,
                                )

                            prediction = np.vstack(prediction_list, dtype=np.float32)
                            prediction = torch.tensor(prediction).to(calc_device)

                prediction = prediction.cpu().detach().numpy()
                with torch.no_grad():
                    model["global results"] = loss_fn(
                        torch.tensor(prediction, dtype=torch.float32),
                        torch.tensor(unmasked_valid_target, dtype=torch.float32),
                    )
                    model['raw predictions'] = prediction
                    model['raw targets'] = unmasked_valid_target
                    model['raw leadtimes'] = validation_unscaled_leadtimes

                    # save predictions, targets, leadtimes as dataframe
                    df = pd.DataFrame()
                    transformed_means = target_scaler.inverse_transform(prediction[:,[0,2]])
                    transformed_sigmas = target_scaler.inverse_transform(prediction[:,[1,3]])
                    transformed_target = target_scaler.inverse_transform(
                        unmasked_valid_target
                    )
                    df['Predictions_vmax_mu'] = transformed_means[:,0]
                    df['Predictions_mslp_mu'] = transformed_means[:,1]
                    df['Predictions_vmax_sigma'] = transformed_sigmas[:,0]
                    df['Predictions_mslp_sigma'] = transformed_sigmas[:,1]
                    df['Targets_vmax'] = transformed_target[:,0]
                    df['Targets_mslp'] = transformed_target[:,1]
                    df['Leadtimes'] = validation_unscaled_leadtimes.flatten()
                    df.to_csv(f"{results_dir}model_comparison_raw_predictions_{args.tag}.csv", index=False)

        for unique_leadtime in unique_leadtimes:
            bool_idxs = (validation_unscaled_leadtimes == unique_leadtime).squeeze()
            temp_x = valid_x[bool_idxs]
            temp_unmasked_x = unmasked_valid_x[bool_idxs]
            temp_target = valid_target[bool_idxs]

            if not args.report_global and args.dask_array:
                temp_deepx = valid_data[bool_idxs].compute(scheduler="threads")
                temp_deepunmasked = unmasked_valid_data[bool_idxs].compute(
                    scheduler="threads"
                )
            else:
                temp_deepx = valid_data[bool_idxs]
                temp_deepunmasked = unmasked_valid_data[bool_idxs]
            temp_scalars = deep_scalars[bool_idxs]
            print(temp_deepx.shape, temp_scalars.shape, temp_target.shape)

            temp_clim = climatology_model.predict([unique_leadtime]).flatten()

            clim_prob = np.tile(temp_clim, (temp_target.shape[0], 1))
            clim_det = np.tile(temp_clim[::2], (temp_target.shape[0], 1))

            temp_persistence = np.zeros_like(temp_target)

            deterministic_loss = (
                torch.nn.MSELoss()
                if args.deterministic_loss == "MSE"
                else torch.nn.L1Loss()
            )

            deterministic_score = deterministic_loss(
                torch.tensor(clim_det, dtype=torch.float32),
                torch.tensor(temp_target, dtype=torch.float32),
            ).item()
            persistence_score = deterministic_loss(
                torch.tensor(temp_persistence, dtype=torch.float32),
                torch.tensor(temp_target, dtype=torch.float32),
            ).item()
            mae_persistence = torch.nn.L1Loss()(
                torch.tensor(temp_persistence, dtype=torch.float32),
                torch.tensor(temp_target, dtype=torch.float32),
            ).item()

            climatology_results["deterministic"].append(
                deterministic_loss(
                    torch.tensor(clim_det, dtype=torch.float32),
                    torch.tensor(temp_target, dtype=torch.float32),
                ).item()
            )
            climatology_results["probabilistic"].append(
                metrics.CRPS_ML(
                    torch.tensor(clim_prob, dtype=torch.float32),
                    torch.tensor(temp_target, dtype=torch.float32),
                ).item()
            )
            persistence_results["deterministic"].append(persistence_score)
            persistence_results["probabilistic"].append(mae_persistence)

            i = 0
            for model in models:
                print(
                    f"Evaluating model {model['tag']} at leadtime {unique_leadtime} {i}/{len(models)}",
                    flush=True,
                )
                if model["probabilistic"]:
                    loss_func = metrics.CRPS_ML
                else:
                    if args.deterministic_loss == "MSE":
                        loss_func = torch.nn.MSELoss()
                    elif args.deterministic_loss == "MAE":
                        loss_func = torch.nn.L1Loss()
                    else:
                        raise ValueError("Loss function not recognized.")
                if model["masked"]:
                    if not model["deep"]:
                        with torch.no_grad():
                            prediction = model["model"](
                                torch.tensor(temp_x, dtype=torch.float32).to(
                                    calc_device
                                )
                            )

                        num_samples = len(temp_x)

                        if args.shap:
                            # sample a third of the data
                            explainer = shap.DeepExplainer(
                                model["model"],
                                torch.tensor(
                                    temp_x[: int(num_samples / 3)],
                                    dtype=torch.float32,
                                ).to(calc_device),
                            )
                            shap_data = torch.tensor(
                                temp_x[-300:], dtype=torch.float32
                            ).to(calc_device)
                            shap_values = explainer(shap_data)
                            model["shap_values"][unique_leadtime] = shap_values
                            model["shap_data"][unique_leadtime] = shap_data
                            model["shap_features"] = [
                                "Max Wind Magnitude",
                                "Min MSLP",
                                "Range Wind Magnitude",
                                "Range MSLP",
                                "Min Z500",
                                "Range T850",
                                "Leadtime",
                                "Base Max Wind",
                                "Base Min MSLP",
                            ]

                    else:
                        # Work by batches
                        batch_size = args.batch_size
                        len_data = temp_deepx.shape[0]

                        with torch.no_grad():
                            prediction_list = []
                            for i in range(0, len_data, batch_size):
                                prediction = model["model"](
                                    torch.tensor(
                                        temp_deepx[i : i + batch_size],
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                    torch.tensor(
                                        temp_scalars[i : i + batch_size],
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                )
                                prediction_list.append(
                                    prediction.cpu().detach().numpy()
                                )
                                # Simple progress bar
                                print(
                                    f"{i}/{len_data}",
                                    end="\r",
                                    flush=True,
                                )

                            prediction = np.vstack(prediction_list, dtype=np.float32)
                            prediction = torch.tensor(prediction).to(calc_device)

                        # prediction = model["model"](
                        #     torch.tensor(temp_deepx, dtype=torch.float32),
                        #     torch.tensor(temp_scalars, dtype=torch.float32),
                        # )
                else:
                    if not model["deep"]:
                        with torch.no_grad():
                            prediction = model["model"](
                                torch.tensor(temp_unmasked_x, dtype=torch.float32).to(
                                    calc_device
                                )
                            )
                        num_samples = len(temp_x)

                        if args.shap:
                            # sample a third of the data
                            explainer = shap.DeepExplainer(
                                model["model"],
                                torch.tensor(
                                    temp_x[: int(num_samples / 3)],
                                    dtype=torch.float32,
                                ).to(calc_device),
                            )
                            shap_data = torch.tensor(
                                temp_x[-300:], dtype=torch.float32
                            ).to(calc_device)
                            shap_values = explainer.shap_values(shap_data)
                            model["shap_values"][unique_leadtime] = shap_values
                            model["shap_data"][unique_leadtime] = shap_data
                            model["shap_features"] = [
                                "Max Wind Magnitude",
                                "Min MSLP",
                                "Range Wind Magnitude",
                                "Range MSLP",
                                "Min Z500",
                                "Range T850",
                                "Leadtime",
                                "Base Intensity",
                            ]
                    else:

                        # Work by batches
                        batch_size = args.batch_size
                        len_data = temp_deepunmasked.shape[0]

                        with torch.no_grad():
                            prediction_list = []
                            for i in range(0, len_data, batch_size):
                                prediction = model["model"](
                                    torch.tensor(
                                        temp_deepunmasked[i : i + batch_size],
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                    torch.tensor(
                                        temp_scalars[i : i + batch_size],
                                        dtype=torch.float32,
                                    ).to(calc_device),
                                )
                                prediction_list.append(
                                    prediction.cpu().detach().numpy()
                                )
                                # Simple progress bar
                                print(
                                    f"{i}/{len_data}",
                                    end="\r",
                                    flush=True,
                                )

                            prediction = np.vstack(prediction_list, dtype=np.float32)
                            prediction = torch.tensor(prediction).to(calc_device)

                        # prediction = model["model"](
                        #     torch.tensor(temp_deepunmasked, dtype=torch.float32).to(
                        #         calc_device
                        #     ),
                        #     torch.tensor(temp_scalars, dtype=torch.float32).to(
                        #         calc_device
                        #     ),
                        # )
                loss = loss_func(
                    prediction,
                    torch.tensor(temp_target, dtype=torch.float32).to(calc_device),
                )
                model["results"].append(loss.item())
                loss = None
            del temp_x, temp_unmasked_x, temp_target, temp_deepx, temp_scalars
            gc.collect()

        # delete the torch models from the model list
        for model in models:
            del model["model"]

        with open(f"{results_dir}model_comparison_eval_{args.tag}.pkl", "wb") as f:
            pickle.dump(
                {
                    "models": models,
                    "climatology_results": climatology_results,
                    "persistence_results": persistence_results,
                    "unique_leadtimes": unique_leadtimes,
                },
                f,
            )
    else:
        with open(f"{results_dir}model_comparison_eval_{args.tag}.pkl", "rb") as f:
            data = pickle.load(f)
            models = data["models"]
            climatology_results = data["climatology_results"]
            persistence_results = data["persistence_results"]
            unique_leadtimes = data["unique_leadtimes"]

# %%
# Format results into CSV for use with pgfplots

filename = f"{results_dir}model_comparison_leadtime_{args.tag}.csv"
# Make the dataframe with leadtimes from unique leadtimes
df = pd.DataFrame(unique_leadtimes, columns=["Leadtime"], dtype=int)
for model in models:
    print(f"Adding model {model['tag']} results to dataframe.")
    if model["probabilistic"]:
        df[model["tag"]] = model["results"]
    else:
        df[model["tag"]] = model["results"]
df.to_csv(filename, index=False)

# %%
# do the same for climatology and persistence results
baseline_df = pd.DataFrame(
    columns=["Leadtime", "Climatology (Deterministic)", "Climatology (Probabilistic)"]
)
baseline_df["Leadtime"] = unique_leadtimes
baseline_df["Climatology (Deterministic)"] = climatology_results["deterministic"]
baseline_df["Climatology (Probabilistic)"] = climatology_results["probabilistic"]
baseline_df["Persistence (Deterministic)"] = persistence_results["deterministic"]
baseline_df["Persistence (Probabilistic)"] = persistence_results["probabilistic"]

baseline_df.to_csv(f"{results_dir}model_comparison_baselines.csv", index=False)

# %%
if args.report_global:
    filename = f"{results_dir}model_comparison_global_{args.tag}.csv"
    global_df = pd.DataFrame(
        columns=["Model", f"{'Test' if args.test_set else 'Validation'} Result"]
    )
    for model in models:
        value = np.nan
        if "global results" in model:
            value = model["global results"].item()
        global_df.loc[-1] = [
            model["tag"],
            value,
        ]
        global_df.index += 1  # Shift index

    global_df = global_df.sort_index()
    global_df.to_csv(filename, index=False)


    # do the same for climatology and persistence results

# %%


# Print out the global results
if args.report_global:
    print("\nGlobal results:")
    for model in models:
        if "global results" in model:
            print(
                f"{model['filepath']}:\n"
                f"{model['tag']}: {model['global results'].item()} \n"
            )
        else:
            print(f"{model['tag']}: No global results available.\n")


# %%
# Print out the per leadtime results
for model in models:
    print(
        f"\n Model {model['tag']} mode: {'probabilistic' if model['probabilistic'] else 'deterministic'}"
    )
    for i, result in enumerate(model["results"]):
        print(f"({unique_leadtimes[i]}, {result}) ", end="")
# %%
# Print out the climatology and persistence results per leadtime
print("\n Deterministic Climatology Results:")
for i, result in enumerate(climatology_results["deterministic"]):
    print(
        f"({unique_leadtimes[i]}, {result}) ",
        end="",
    )
print("\n Probabilistic Climatology Results:")
for i, result in enumerate(climatology_results["probabilistic"]):
    print(
        f"({unique_leadtimes[i]}, {result}) ",
        end="",
    )
print("\n Deterministic Persistence Results:")
for i, result in enumerate(persistence_results["deterministic"]):
    print(
        f"({unique_leadtimes[i]}, {result}) ",
        end="",
    )
print("\n Probabilistic Persistence Results:")
for i, result in enumerate(persistence_results["probabilistic"]):
    print(
        f"({unique_leadtimes[i]}, {result}) ",
        end="",
    )


# %% Plotting the results

fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

# Color Cycle
colors = (
    np.array([38, 1, 36]) / 255,  # Black
    # np.array([242, 141, 168]) / 255,  # Pink
    np.array([7, 140, 91]) / 255,  # Green
    np.array([242, 179, 102]) / 255,  # Orange
    np.array([242, 110, 80]) / 255,  # Salmon
    np.array([56, 93, 166]) / 255,  # Blue
    np.array([242, 227, 182]) / 255,  # Beige
)
axs[0].set_prop_cycle(color=colors)
axs[1].set_prop_cycle(color=colors)

"""

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
"""

lines = [
    (0, (1, 5)),  # dotted
    (5, (10, 3)),  # long dash with offset
    (0, (5, 5)),  # dashed
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
    (0, ()),  # solid
    # "--",
    # "-.",
    # ":",
]
linecycler = cycle(lines)

markers = [
    # "o",  # circle
    "+",  # plus
    "p",  # pentagon
    "d",  # diamond
    "^",  # triangle up
    "v",  # triangle down
    "3",  # triange left
    "*",  # star
]
markercycler = cycle(markers)

for model in models:
    linestyle = next(linecycler)
    if model["probabilistic"]:
        axs[0].plot(
            unique_leadtimes,
            model["results"],
            label=model["tag"],
            marker=next(markercycler),
            linestyle=linestyle,
            linewidth=0.75,
            markersize=5,
            alpha=0.75,
        )
    else:
        axs[1].plot(
            unique_leadtimes,
            model["results"],
            label=model["tag"],
            marker=next(markercycler),
            linestyle=linestyle,
            linewidth=0.75,
            markersize=5,
            alpha=0.75,
        )

axs[0].plot(
    unique_leadtimes,
    climatology_results["probabilistic"],
    label="Average Climatology",
    marker="1",
    linestyle="-",
)
axs[0].plot(
    unique_leadtimes,
    persistence_results["probabilistic"],
    label="Persistence (MAE)",
    marker="x",
    linestyle="--",
)
axs[1].plot(
    unique_leadtimes,
    climatology_results["deterministic"],
    label="Average Climatology",
    marker="1",
    linestyle="-",
)
axs[1].plot(
    unique_leadtimes,
    persistence_results["deterministic"],
    label="Persistence",
    marker="x",
    linestyle="--",
)

fig.suptitle(f"Model Evaluation - {args.ai_model}", color="white")
axs[0].set_title("Probabilistic Models")
axs[1].set_title("Deterministic Models")
axs[0].set_xlabel("Leadtime (hours)")
axs[1].set_xlabel("Leadtime (hours)")
axs[0].set_ylabel("CRPS")
axs[1].set_ylabel("MSE Loss")
axs[0].legend()
axs[1].legend()

toolbox.plot_facecolors(fig=fig, axes=axs)
fig.tight_layout()
fig.savefig(f"{results_dir}model_comparison_eval_{args.tag}.png")

# %%
if args.shap:
    for model in models:
        # Plotting the SHAP values
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        shap_values = model["shap_values"]
        shap_data = model["shap_data"]
        shap_features = [
            "Max Wind Magnitude",
            "Min Mean SLP",
            "Range Wind Magnitude",
            "Range Mean SLP",
            "Min Z500",
            "Range T850",
            "Leadtime",
            "Base Max Wind",
            "Base Min SLP",
        ]

        def plot_adjuster(height, width, dpi, figtitle):
            fig = plt.gcf()
            fig.set_figheight(height)
            fig.set_figwidth(width)
            fig.set_dpi(dpi)

            # set font sizes for the axes
            for ax in fig.axes:
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.tick_params(axis="both", which="minor", labelsize=16)
                ax.set_xlabel(ax.get_xlabel(), fontsize=16)
                ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            # remove suptitle
            fig.suptitle("")

            # fig.suptitle(figtitle, horizontalalignment="center")
            plt.tight_layout()
            # set margins
            # plt.subplots_adjust(left=0.35, right=0.88, top=0.88, bottom=0.12)

            # ax = plt.gca()
            # # Set the left margin to 0.3
            # ax.spines["left"].set_position(("axes", 0.3))

        unique_leadtimes = list(shap_values.keys())
        for j, target in enumerate(
            [
                "(mu) Maximum Wind",
                "(sigma) Maximum Wind",
                "(mu) Minimum SLP",
                "(sigma) Minimum SLP",
            ]
        ):
            for i, leadtime in enumerate([6, 12, 18, 24, 48, 72, 96, 120, 144, 168]):
                shap.summary_plot(
                    shap_values[leadtime][:, :, j],
                    shap_data[leadtime],
                    feature_names=np.array(shap_features),
                    max_display=len(shap_features),
                    plot_type="dot",
                    show=False,
                    color_bar=True,
                    alpha=0.75,
                    # ax=axs[i],
                )
                plot_adjuster(
                    8,
                    8,
                    250,
                    f"{model['tag']} - {leadtime}h ldt - {target}\nSHAP Values",
                )

                # plt.show()
                if not os.path.exists(os.path.join(results_dir, "shap")):
                    os.makedirs(os.path.join(results_dir, "shap"))
                plt.savefig(
                    f"{os.path.join(results_dir, 'shap')}/shap_{leadtime}h_{target}_{model['tag']}.png",
                    format="png",
                )
                plt.close()

                # Check if the shap_values are numpy arrays or explainer objects
                if isinstance(shap_values[leadtime], shap.Explanation):
                    shap_val_array = shap_values[leadtime].values
                else:
                    shap_val_array = shap_values[leadtime]
                # combined thermodynamic and dynamic features
                temperature = np.sum(shap_val_array[:, [5], j], axis=1)[
                    :,
                    None,
                ]
                pressure = np.sum(shap_val_array[:, [1, 3, 4, 8], j], axis=1)[
                    :,
                    None,
                ]
                wind = np.sum(shap_val_array[:, [0, 2, 7], j], axis=1)[:, None]
                shap.summary_plot(
                    np.concatenate(
                        [
                            temperature,
                            pressure,
                            wind,
                            shap_val_array[:, 6][:, None, j],
                        ],
                        axis=1,
                    ),
                    # shap_data[leadtime],
                    feature_names=np.array(
                        [
                            "T850 (range)",
                            "Pressure Variables",
                            "Wind Variables",
                            "Leadtime",
                        ]
                    ),
                    plot_type="dot",
                    show=False,
                    color_bar=True,
                    alpha=0.75,
                )

                plot_adjuster(
                    4,
                    8,
                    250,
                    f"{model['tag']} - {leadtime}h ldt - {target}\nVariable Grouping SHAP Values",
                )
                plt.savefig(
                    f"{os.path.join(results_dir, 'shap')}/shap_{leadtime}h_{target}_{model['tag']}_vartype.png",
                    format="png",
                )
                plt.close()

                # combined neural weather model features
                newm_shapval = np.sum(shap_val_array[:, [0, 1, 2, 3, 4, 5], j], axis=1)[
                    :, None
                ]
                scalar_shapval = np.sum(shap_val_array[:, [6, 7, 8], j], axis=1)[
                    :, None
                ]
                shap.summary_plot(
                    np.concatenate(
                        [newm_shapval, scalar_shapval],
                        axis=1,
                    ),
                    # shap_data[leadtime],
                    feature_names=np.array(["Neural Weather Model", "IBTrACS Scalars"]),
                    plot_type="dot",
                    show=False,
                    color_bar=True,
                    alpha=0.75,
                )
                plot_adjuster(
                    4,
                    8,
                    250,
                    f"{model['tag']} - {leadtime}h ldt - {target}\nVariable Grouping SHAP Values",
                )
                plt.savefig(
                    f"{os.path.join(results_dir, 'shap')}/shap_{leadtime}h_{target}_{model['tag']}_neural_vs_scalar.png",
                    format="png",
                )
                plt.close()

                # input()


# %%
