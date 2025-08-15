#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:57:33 2023

Script to test handling of a single track

@author: mgomezd1
"""

# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from importlib import reload

# Backend Libraries
import xarray as xr

from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib
from utils import ML_functions as mlf
import torch
import dask

# import cartopy for coastlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

full_data = toolbox.read_hist_track_file(
    tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"
)
results_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)

# %%
if __name__ == "__main__":

    # # check if the context has been set for torch multiprocessing
    # if not torch.multiprocessing.get_start_method(allow_none=True) == "spawn":
    #     torch.multiprocessing.set_start_method("spawn")

    # # print the multiprocessing start method
    # print(torch.multiprocessing.get_start_method(allow_none=True))
    # dask.config.set(scheduler="processes")

    # # Check for GPU availability
    # if torch.cuda.is_available():
    #     calc_device = torch.device("cuda:0")
    # else:
    #     calc_device = torch.device("cpu")

    calc_device = torch.device("cpu")

    ai_model = "panguweather"
    magangle = True

    cache_dir = f"/scratch/mgomezd1/cache/_{ai_model}"

    AI_scaler = None
    fpath = os.path.join(cache_dir, "AI_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading AI scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            AI_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"AI scaler not found at {fpath}")

    base_scaler = None
    fpath = os.path.join(cache_dir, "base_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading base scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            base_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Base scaler not found at {fpath}")

    target_scaler = None
    fpath = os.path.join(cache_dir, "target_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading target scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            target_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Target scaler not found at {fpath}")

    mask_inputs = True
    mask_path = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/mask_dict.pkl"
    with open(mask_path, "rb") as f:
        mask_dict = pickle.load(f)["linear"]

    # Load the MLR model
    models_to_load = [
        # {
        #     "filepath": f"{results_dir+'torch_models/'}MLR_01-27-11h37_epoch-20_panguweather_probabilistic_prob_masked.pt",
        #     "masked": True,
        #     "probabilistic": True,
        #     "tag": "masked MLR",
        #     "results": {},
        #     "deep": False,
        # },
        {
            "filepath": f"{results_dir+'torch_models/'}TorchMLR_03-26-10h27_epoch-20_panguweather_probabilistic_masked.pt",
            "masked": True,
            "probabilistic": True,
            "tag": "MLR (Masked)",
            "results": {},
            "deep": False,
        },
        # {
        #     "filepath": f"{results_dir+'torch_models/'}SimpleANN_03-28-14h49_epoch-20_panguweather_probabilistic_masked.pt",
        #     "masked": True,
        #     "probabilistic": True,
        #     "tag": "ANN (LeakyReLU, M)",
        #     "results": {},
        #     "deep": False,
        # },
        {
            "filepath": f"{results_dir+'torch_models/'}SimpleANN_02-03-08h29_epoch-14_panguweather_probabilistic_masked.pt",
            "masked": True,
            "probabilistic": True,
            "tag": "ANN (LeakyReLU, M)",
            "results": {},
            "deep": False,
        },
        # {
        #     "filepath": f"{results_dir}UNet_9_01-24-21h32_epoch-5_panguweather-probabilisticdout25.pt",
        #     "masked": True,
        #     "probabilistic": True,
        #     "tag": "UNet 168 dropout 25",
        #     "results": [],
        #     "deep": True,
        # },
        # {
        #     "filepath": f"{results_dir+'torch_models/'}UNet_v2_03-31-10h41_epoch-12_panguweather-probabilistic.pt",
        #     "masked": True,
        #     "probabilistic": True,
        #     "tag": "UNetv2 (dout 0.33)",
        #     "results": [],
        #     "deep": True,
        # },
        {
            "filepath": f"{results_dir+'torch_models/'}UNet_v2_06-18-20h11_epoch-6_panguweather-probabilistic.pt",
            "masked": True,
            "probabilistic": True,
            "tag": "UNetv2 (dout 0.33)",
            "results": [],
            "deep": True,
        },
    ]

    for model in models_to_load:
        model["model"] = torch.load(model["filepath"], map_location=calc_device)
        model["model"].eval()
    # %%
    fit = True  # Set to True to fit the models, False to load the models from disk
    if fit:
        plotting_dict = {}
        initial_times = []
        for name, year in [
            ## Validation Storms
            # ("DORIAN", 2019),
            # ("BELNA", 2019),
            # ("VERONICA", 2019),
            # ("KYAAR:KYARR", 2019),
            # ("HALONG", 2019),
            # ("BARBARA", 2019),
            # ("VAYU", 2019),
            # ("JERRY", 2019),
            # ("FLOSSIE", 2019),
            # ("LORNA", 2019),
            # ("OWEN", 2018),
            ## Test Storms
            ("IOTA", 2020),
            ("HEROLD", 2020),
            ("MARIE", 2020),
            ("GONI", 2020),
            ("AMPHAN", 2020),
            ("FERDINAND", 2020),
            ("PAULETTE", 2020),
            # ("ALICIA", 2020),
            ("ELIDA", 2020),
            ("CHAN-HOM", 2020),
            ("NIVAR", 2020),
            ("CLAUDIA", 2020),
        ]:
            print(f"Processing {name} {year}")
            plotting_dict[name] = {}
            data = full_data[full_data.ISO_TIME.dt.year == year]
            storm = data[data.NAME == name]
            track = toolbox.tc_track(
                UID=storm.SID.iloc[0],
                NAME=storm.NAME.iloc[0],
                track=storm[["LAT", "LON"]].to_numpy(),
                timestamps=storm.ISO_TIME.to_numpy(),
                ALT_ID=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")
                ].iloc[0],
                wind=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("WIND")
                ].to_numpy(),
                pres=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("PRES")
                ].to_numpy(),
                datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
                storm_season=storm.SEASON.iloc[0],
                ai_model=ai_model,
            )

            truth_int, truth_time = track.get_ground_truth()
            plotting_dict[name]["IBTrACS"] = (truth_int, truth_time)

            ai_data, base_intensity, target_time, target_leadtime = track.ai.serve()

            if magangle:
                mlf.uv_to_magAngle(data=ai_data, u_idx=0, v_idx=1)

            ai_data = ai_data.compute()
            base_intensity_data = base_intensity.compute()
            target_time = target_time.compute()
            target_leadtime = target_leadtime.compute()
            base_time = target_time - target_leadtime.astype("timedelta64[h]")
            forecast_time = target_time - target_leadtime.astype("timedelta64[h]")

            # get the track timestamps
            track_time = track.timestamps

            # find the index of the forecast time in the track time
            forecast_idx = np.searchsorted(track_time, forecast_time)
            forecast_positions = track.track[forecast_idx]
            forecast_positions = mlf.latlon_to_sincos(forecast_positions)

            # Plot sample masked inputs
            if False:
                # Generate plots of masked inputs for 18, 72, and 120 hour leadtimes

                # Create the colorblind-safe colormap

                # Define the two user-defined colors (as RGB tuples or hex codes)
                color1 = "#3DB7E9FF"  # Blue for low
                color2 = "#D55E00FF"  # Orange for high
                white = "#FFFFFF00"  # White color for diverging colormap

                # Create a custom diverging colormap
                colors = [
                    color1,
                    white,
                    color2,
                ]
                cmap_name = "cblind_diverging"
                diverging_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    cmap_name, colors
                )

                # Sample the appropriate positions
                positions = [0, 0, 0]
                while not np.all(target_leadtime[positions] == [18, 72, 120]):
                    start_pos = np.random.choice(
                        np.where(target_leadtime[: target_leadtime.size - 50] == 6)[0]
                    )
                    positions = [start_pos + 2, start_pos + 5, start_pos + 7]

                    max_val = ai_data[positions][:, 0].max()
                    min_val = ai_data[positions][:, 0].min()
                    if min_val > AI_scaler.mean_[0, 0, 0]:
                        positions = [0, 0, 0]
                    if max_val < AI_scaler.mean_[0, 0, 0]:
                        positions = [0, 0, 0]

                t_base = pd.to_datetime(base_time[positions[0]]).strftime(
                    "%Y-%m-%d-%Hh%M"
                )
                # Code to make visualizations of the masked AI data
                for position in positions:
                    fig, ax = plt.subplots(
                        1,
                        1,
                        figsize=(8, 10),
                        dpi=200,
                        subplot_kw={"projection": ccrs.PlateCarree()},
                    )

                    temp_ai = ai_data[position]
                    # Make the masked inputs
                    mask = mask_dict[target_leadtime[position]]
                    temp_ai = AI_scaler.transform(temp_ai)
                    temp_ai = ai_data[position] * mask
                    temp_ai = AI_scaler.inverse_transform(temp_ai)

                    # retrieve the base position for reference
                    temp_base_pos = track.track[forecast_idx][position]
                    # round to nearest quarter degree
                    plotting_pos = np.round(temp_base_pos / 0.25).astype(int) * 0.25

                    # retrieve the position at target time for reference
                    temp_targettime = target_time[position]
                    temp_target_idx = np.searchsorted(track_time, temp_targettime)
                    temp_target_pos = track.track[temp_target_idx]

                    # calculate the difference in position in quarter degrees
                    diff_pos = temp_target_pos - temp_base_pos
                    diff_pos = np.round(diff_pos / 0.25).astype(int)

                    # calculate the pixel of the true location
                    true_pixel = (121 - diff_pos[0], 121 + diff_pos[1])

                    norm = mpl.colors.TwoSlopeNorm(
                        vmin=0,
                        vcenter=AI_scaler.mean_[0, 0, 0],
                        vmax=max_val,
                    )

                    ax.imshow(
                        temp_ai[0],
                        # cmap="inferno",
                        cmap=diverging_cmap,
                        # vmax=14,
                        norm=norm,
                        extent=[
                            plotting_pos[1] - 30,
                            plotting_pos[1] + 30,
                            plotting_pos[0] - 30,
                            plotting_pos[0] + 30,
                        ],
                        transform=ccrs.PlateCarree(),
                        origin="upper",
                    )

                    ax.add_feature(
                        cfeature.COASTLINE,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=10,
                    )

                    # draw a black plus sign at the true position
                    ax.plot(
                        temp_target_pos[1],
                        temp_target_pos[0],
                        marker="+",
                        color="black",
                        markersize=20,
                        transform=ccrs.PlateCarree(),
                    )
                    # remove ticks
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # set facecolor to transparent
                    ax.set_facecolor("none")
                    fig.set_facecolor("none")

                    # add colorbar at the bottom, m/s, with 4 ticks

                    cbar = fig.colorbar(
                        ax.images[0],
                        ax=ax,
                        orientation="horizontal",
                        pad=0.01,
                        aspect=50,
                        ticks=[0, 5, 10, 15],
                    )
                    text_weight = 28
                    cbar.ax.tick_params(labelsize=text_weight)

                    # set label size
                    cbar.set_label("10m wind magnitude (m/s)", fontsize=text_weight)

                    fig.tight_layout()
                    fig.savefig(
                        f"{results_dir}case_study_plots/{name}_{t_base}_{target_leadtime[position]}_hour_sample.png",
                    )
            #
            print("Starting prediction loop...")
            for model in models_to_load:
                plotting_dict[name][model["tag"]] = {}

            for mode in ["MLR", "deep"]:
                for leadtime in [6, 12, 18, 24, 48, 72, 96, 120, 168]:
                    print(f"Processing {name} {leadtime} hour forecast")
                    bool_idx = target_leadtime == leadtime

                    if sum(bool_idx.astype(int)) == 0:
                        continue

                    temp_ai = ai_data[bool_idx]
                    temp_bi = base_intensity[bool_idx]
                    temp_targettime = target_time[bool_idx]
                    temp_ldt = np.full_like(temp_targettime, leadtime / 168).astype(
                        float
                    )
                    temp_pos = forecast_positions[bool_idx]
                    temp_ai = AI_scaler.transform(temp_ai)
                    temp_bi = base_scaler.transform(temp_bi)

                    temp_mask = mask_dict[leadtime]

                    if mask_inputs:
                        temp_ai = temp_ai * temp_mask

                    if mode == "MLR":
                        temp_maxima = temp_ai.max(axis=(-2, -1))
                        temp_minima = temp_ai.min(axis=(-2, -1))
                        temp_range = temp_maxima - temp_minima

                        temp_inputs = np.vstack(
                            [
                                temp_maxima[:, 0],  # Maximum wind magnitude
                                temp_minima[:, 2],  # Minimum mean sea level pressure
                                temp_range[:, 0],  # Range of wind magnitude
                                temp_range[:, 2],  # Range of mean sea level pressure
                                temp_minima[
                                    :, 3
                                ],  # Minimum geopotential height at 500 hPa
                                temp_range[:, 4],  # Range of temperature at 850 hPa
                                temp_ldt.squeeze(),  # Leadtime
                                temp_bi.T,  # Base intensity
                            ]
                        ).T

                        for model in models_to_load:

                            if not model["deep"]:
                                temp_x = torch.tensor(
                                    temp_inputs, dtype=torch.float32
                                ).to(calc_device)
                            else:
                                print("Deep model, skipping in MLR inputs mode...")

                            with torch.no_grad():
                                if not model["deep"]:
                                    temp_pred = model["model"](temp_x).numpy()
                                    if model["probabilistic"]:
                                        temp_means = temp_pred[:, [0, 2]]
                                        temp_sigma = np.abs(temp_pred[:, [1, 2]])
                                        temp_sigma = target_scaler.inverse_transform(
                                            temp_sigma
                                        )
                                        temp_means = target_scaler.inverse_transform(
                                            temp_means
                                        )
                                        temp_base_adder = base_scaler.inverse_transform(
                                            temp_bi
                                        )
                                        temp_means = temp_means + temp_base_adder
                                        temp_95_lower = temp_means - 1.96 * temp_sigma
                                        temp_95_upper = temp_means + 1.96 * temp_sigma

                            if not model["deep"]:
                                plotting_dict[name][model["tag"]][leadtime] = {
                                    "means": temp_means,
                                    "95_lower": temp_95_lower,
                                    "95_upper": temp_95_upper,
                                    "time": temp_targettime,
                                }

                    elif mode == "deep":
                        scalars = np.vstack(
                            [
                                temp_bi.T,
                                temp_pos.T,
                                np.full(temp_bi.shape[0], leadtime / 168).reshape(
                                    1, -1
                                ),
                            ]
                        ).T

                        for model in models_to_load:
                            if model["deep"]:
                                temp_x = torch.tensor(temp_ai, dtype=torch.float32).to(
                                    calc_device
                                )
                                temp_scalars = torch.tensor(
                                    scalars, dtype=torch.float32
                                ).to(calc_device)
                            else:
                                print("Linear model, skipping in deep inputs mode...")

                            with torch.no_grad():
                                if model["deep"]:
                                    temp_pred = model["model"](
                                        temp_x, temp_scalars
                                    ).numpy()
                                    if model["probabilistic"]:
                                        temp_means = temp_pred[:, [0, 2]]
                                        temp_sigma = np.abs(temp_pred[:, [1, 2]])
                                        temp_sigma = target_scaler.inverse_transform(
                                            temp_sigma
                                        )
                                        temp_means = target_scaler.inverse_transform(
                                            temp_means
                                        )
                                        temp_base_adder = base_scaler.inverse_transform(
                                            temp_bi
                                        )
                                        temp_means = temp_means + temp_base_adder
                                        temp_95_lower = temp_means - 1.96 * temp_sigma
                                        temp_95_upper = temp_means + 1.96 * temp_sigma
                                    else:
                                        "Deterministic model, skipping for now..."
                                        continue
                            if model["deep"]:
                                plotting_dict[name][model["tag"]][leadtime] = {
                                    "means": temp_means,
                                    "95_lower": temp_95_lower,
                                    "95_upper": temp_95_upper,
                                    "time": temp_targettime,
                                }

        np.save(f"{results_dir}case_study_plotting_dict.npy", plotting_dict)
    else:
        plotting_dict = np.load(
            f"{results_dir}case_study_plotting_dict.npy", allow_pickle=True
        ).item()

# %% latex pgfplot data generation

for storm, data in plotting_dict.items():
    truth_int, truth_time = data["IBTrACS"]

    # print out the truth pairs for pgfplots
    print("\n------\nStorm: ", storm, "IBTrACS")
    # Make time labels with YYYY-MM-DD:HH format
    time_labels = [f"{t.astype('datetime64[m]').astype(str)}" for t in truth_time]
    time_labels = [t.replace("T", " ") for t in time_labels]
    # print("Max wind: ")
    # for idx, val in enumerate(truth_int[:, 0]):
    #     print(f"({time_labels[idx]}, {val})", end=" ")
    # print("\nMin pressure: ")
    # for idx, val in enumerate(truth_int[:, 1]):
    #     print(f"({time_labels[idx]}, {val})", end=" ")

    # Create a pandas dataframe with the ibtracs data
    ibtracs_df = pd.DataFrame(
        {
            "time": time_labels,
            "max_wind": truth_int[:, 0],
            "min_pres": truth_int[:, 1],
        }
    )
    # print the dataframe to a csv file
    ibtracs_df.to_csv(
        os.path.join(
            results_dir,
            "case_study_plots",
            f"{storm}_IBTrACS.csv",
        ),
        index=False,
    )

    # find the keys that aren't IBTrACS
    model_keys = [key for key in data.keys() if key != "IBTrACS"]

    leadtime_set = [18, 24, 72, 96, 120]

    for leadtime in leadtime_set:

        for model in model_keys:
            means = data[model][leadtime]["means"]
            lower = data[model][leadtime]["95_lower"]
            upper = data[model][leadtime]["95_upper"]
            time = data[model][leadtime]["time"]

            print("\n------\nStorm: ", storm, "model: ", model, "leadtime: ", leadtime)

            # Make time labels with YYYY-MM-DD:HH:MM format
            time_labels = [f"{t.astype('datetime64[m]').astype(str)}" for t in time]
            time_labels = [t.replace("T", " ") for t in time_labels]

            # create a pandas dataframe with the means, lower, and upper values
            df = pd.DataFrame(
                {
                    "time": time_labels,
                    "mean_max_wind": means[:, 0],
                    "mean_min_pres": means[:, 1],
                    "lower_max_wind": lower[:, 0],
                    "lower_min_pres": lower[:, 1],
                    "upper_max_wind": upper[:, 0],
                    "upper_min_pres": upper[:, 1],
                }
            )

            # print the dataframe to a csv file
            df.to_csv(
                os.path.join(
                    results_dir,
                    "case_study_plots",
                    f"{storm}_{model}_{leadtime}_hour_forecast.csv",
                ),
                index=False,
            )

            # print("Mean max wind: ")
            # for idx, val in enumerate(means[:, 0]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")
            # print("\nUpper 95:")
            # for idx, val in enumerate(upper[:, 0]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")
            # print("\nLower 95:")
            # for idx, val in enumerate(lower[:, 0]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")

            # print("\nMean min pressure: ")
            # for idx, val in enumerate(means[:, 1]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")
            # print("\nUpper 95:")
            # for idx, val in enumerate(upper[:, 1]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")
            # print("\nLower 95:")
            # for idx, val in enumerate(lower[:, 1]):
            #     print(f"({time_labels[idx]}, {val})", end=" ")

    print("\n\n\n\n\n")


# %% Plotting color setup

black = (0 / 255, 0 / 255, 0 / 255)
white = (255 / 255, 255 / 255, 255 / 255)

# Group 1 colors
jazzberry_jam = (159 / 255, 1 / 255, 98 / 255)
jazzberry_creepers = (0 / 255, 159 / 255, 129 / 255)
barbie_pink = (255 / 255, 90 / 255, 175 / 255)
aquamarine = (0 / 255, 252 / 255, 207 / 255)

# Group 2 colors
french_violet = (132 / 255, 0 / 255, 205 / 255)
dodger_blue = (0 / 255, 141 / 255, 249 / 255)
capri = (0 / 255, 194 / 255, 249 / 255)
plum = (255 / 255, 178 / 255, 253 / 255)

# Group 3 colors
carmine = (164 / 255, 1 / 255, 34 / 255)
alizan_crimson = (226 / 255, 1 / 255, 52 / 255)
outrageous_orange = (255 / 255, 110 / 255, 58 / 255)
bright_spark = (255 / 255, 195 / 255, 59 / 255)


# %%
# Color prep for mpl cycler
colors = [
    black,
    jazzberry_jam,
    dodger_blue,
    outrageous_orange,
    plum,
    aquamarine,
]

# make a cyclic iterator from colors using itertools


for storm, data in plotting_dict.items():
    truth_int, truth_time = data["IBTrACS"]

    # find the keys that aren't IBTrACS
    model_keys = [key for key in data.keys() if key != "IBTrACS"]

    leadtime_set = set()
    for model in model_keys:
        if isinstance(data[model], dict):
            leadtime_set.update(data[model].keys())

    color_cycler = itertools.cycle(colors)

    ibtracs_color = next(color_cycler)
    # get colors for each of the models
    model_colors = {}
    for model in model_keys:
        model_colors[model] = next(color_cycler)

    for leadtime in leadtime_set:

        # make prediction plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(truth_time, truth_int[:, 0], color=ibtracs_color, label="IBTrACS")
        ax[1].plot(truth_time, truth_int[:, 1], color=ibtracs_color, label="IBTrACS")

        for model in model_keys:
            means = data[model][leadtime]["means"]
            lower = data[model][leadtime]["95_lower"]
            upper = data[model][leadtime]["95_upper"]
            time = data[model][leadtime]["time"]
            # plot the mean and 95% confidence interval
            ax[0].plot(time, means[:, 0], color=model_colors[model], label=model)
            ax[0].fill_between(
                time, lower[:, 0], upper[:, 0], color=model_colors[model], alpha=0.15
            )
            ax[1].plot(time, means[:, 1], color=model_colors[model], label=model)
            ax[1].fill_between(
                time, lower[:, 1], upper[:, 1], color=model_colors[model], alpha=0.15
            )

        ax[0].set_ylabel("Wind Speed (kt)")
        ax[0].grid()
        ax[0].legend()

        ax[1].set_ylabel("Pressure (hPa)")
        ax[1].grid()
        ax[1].legend()
        # set x label ticks at 45 degree angle
        plt.xticks(rotation=45)
        fig.suptitle(f"{storm} {leadtime} hour forecast")
        ax[1].set_xlabel("Time")

        fig.tight_layout()

        fig_dir = os.path.join(results_dir, "case_study_plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        plt.savefig(
            os.path.join(fig_dir, f"{storm}_{leadtime}_hour_forecast.png"), dpi=200
        )
        plt.show()
        plt.close()


# %%
