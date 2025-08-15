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
import matplotlib.animation as animation
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

    def extract_indices_with_missing(vector, sequence):
        indices_list = []
        seq_len = len(sequence)

        i = 0
        while i < len(vector):
            indices = []
            seq_index = 0

            while seq_index < seq_len and i < len(vector):
                if vector[i] == sequence[seq_index]:
                    indices.append(i)
                    seq_index += 1
                else:
                    indices.append(np.nan)
                    seq_index += 1
                    i -= 1  # Stay on the same element for the next iteration

                i += 1

            if len(indices) == seq_len:
                indices_list.append(indices)

        return indices_list

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
    # mode = "deep"  # "MLR" or "deep"

    cache_dir = f"/scratch/mgomezd1/cache/_{ai_model}"

    magangle = True

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
        {
            "filepath": f"{results_dir+'torch_models/'}UNet_v2_03-31-10h41_epoch-12_panguweather-probabilistic.pt",
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
    fit = True
    if fit:
        plotting_dict = {}
        initial_times = []
        for name, year in [
            ("CLAUDIA", 2020)
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
                ai_data = mlf.uv_to_magAngle(data=ai_data, u_idx=0, v_idx=1)

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

            print("Starting prediction loop...")
            for model in models_to_load:
                plotting_dict[name][model["tag"]] = []

            unique_forecast_times = np.unique(forecast_time)
            plot_info = {}

            plot = True
            if plot:
                # Make a 4 column plot that will show the AI wind magnitude and mslp images
                # the aspect ratio of the images is 1:1
                fig, ax = plt.subplots(2, 4, figsize=(16, 8))

                for axis in ax.flatten():
                    axis.set_box_aspect(1)

                fig.tight_layout()

                def frame_update(frame):
                    # Clear the axes
                    for a in ax.flatten():
                        a.clear()
                    bool_idx = np.isin(forecast_time, unique_forecast_times[frame])
                    temp_ai = ai_data[bool_idx]
                    temp_targettime = target_time[bool_idx]
                    temp_ldt = target_leadtime[bool_idx]

                    bool_filter = np.isin(temp_ldt, [6, 24, 72, 120])
                    temp_ai = temp_ai[bool_filter]
                    temp_targettime = temp_targettime[bool_filter]
                    temp_ldt = temp_ldt[bool_filter]

                    # Find the range of the wind speed and mslp
                    wind_max = np.max(temp_ai[:, 0])
                    wind_min = np.min(temp_ai[:, 0])
                    mslp_max = np.max(temp_ai[:, 2])
                    mslp_min = np.min(temp_ai[:, 2])

                    # prepare for colorbar
                    norm1 = mpl.colors.Normalize(vmin=wind_min, vmax=wind_max)
                    cmap1 = mpl.cm.seismic
                    norm2 = mpl.colors.Normalize(vmin=mslp_min, vmax=mslp_max)
                    cmap2 = mpl.cm.cividis
                    # # Create a colorbar for the wind speed
                    # cbar1 = fig.colorbar(
                    #     mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1),
                    #     ax=ax[0, 0],
                    #     orientation="vertical",
                    #     pad=0.01,
                    # )
                    # cbar1.set_label("Wind Speed (m/s)")
                    # # Create a colorbar for the mslp
                    # cbar2 = fig.colorbar(
                    #     mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                    #     ax=ax[1, 0],
                    #     orientation="vertical",
                    #     pad=0.01,
                    # )
                    # cbar2.set_label("MSLP (hPa)")
                    for i in range(sum(bool_filter.astype(int))):
                        ax[0, i].imshow(
                            temp_ai[i, 0],
                            cmap=cmap1,
                            norm=norm1,
                        )
                        ax[1, i].imshow(
                            temp_ai[i, 2],
                            cmap=cmap2,
                            norm=norm2,
                        )
                        ax[0, i].set_title(f"AI Wind Magnitude ({temp_ldt[i]}h)")
                        ax[1, i].set_title(f"AI MSLP ({temp_ldt[i]}h)")

                    # ax[0, 0].imshow(temp_ai[0, 0], cmap=cmap1, norm=norm1)
                    # ax[0, 1].imshow(temp_ai[1, 0], cmap=cmap1, norm=norm1)
                    # ax[0, 2].imshow(temp_ai[2, 0], cmap=cmap1, norm=norm1)
                    # ax[0, 3].imshow(temp_ai[3, 0], cmap=cmap1, norm=norm1)

                    # ax[1, 0].imshow(temp_ai[0, 2], cmap=cmap2, norm=norm2)
                    # ax[1, 1].imshow(temp_ai[1, 2], cmap=cmap2, norm=norm2)
                    # ax[1, 2].imshow(temp_ai[2, 2], cmap=cmap2, norm=norm2)
                    # ax[1, 3].imshow(temp_ai[3, 2], cmap=cmap2, norm=norm2)
                    # ax[0, 0].set_title("AI Wind Magnitude (6h)")
                    # ax[0, 1].set_title("AI Wind Magnitude (24h)")
                    # ax[0, 2].set_title("AI Wind Magnitude (72h)")
                    # ax[0, 3].set_title("AI Wind Magnitude (120h)")
                    # ax[1, 0].set_title("AI MSLP (6h)")
                    # ax[1, 1].set_title("AI MSLP (24h)")
                    # ax[1, 2].set_title("AI MSLP (72h)")
                    # ax[1, 3].set_title("AI MSLP (120h)")
                    for axis in ax.flatten():
                        axis.set_box_aspect(1)

                anim = animation.FuncAnimation(
                    fig, frame_update, frames=len(unique_forecast_times), interval=400
                )
                fig_dir = os.path.join(results_dir, "case_study_anims")
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                anim.save(
                    os.path.join(fig_dir, f"{name}_ai_images.gif"), writer="pillow"
                )
                plt.close(fig)

            for timestamp in unique_forecast_times:
                # Get the indices of the forecast time in the target time
                bool_idx = np.isin(forecast_time, timestamp)
                if sum(bool_idx.astype(int)) == 0:
                    continue

                temp_ai = ai_data[bool_idx]
                temp_bi = base_intensity[bool_idx]
                temp_ldt = target_leadtime[bool_idx]
                temp_targettime = target_time[bool_idx]
                temp_pos = forecast_positions[bool_idx]
                temp_ai = AI_scaler.transform(temp_ai)
                temp_bi = base_scaler.transform(temp_bi)

                if mask_inputs:
                    # initialize the mask
                    temp_mask = np.ones_like(temp_ai)

                    for idx, ldt in enumerate(temp_ldt):
                        temp_ai[idx] = temp_ai[idx] * mask_dict[ldt]

                temp_maxima = temp_ai.max(axis=(-2, -1))
                temp_minima = temp_ai.min(axis=(-2, -1))
                temp_range = temp_maxima - temp_minima

                temp_lin_in = np.vstack(
                    [
                        temp_maxima[:, 0],  # Maximum wind magnitude
                        temp_minima[:, 2],  # Minimum mean sea level pressure
                        temp_range[:, 0],  # Range of wind magnitude
                        temp_range[:, 2],  # Range of mean sea level pressure
                        temp_minima[:, 3],  # Minimum geopotential height at 500 hPa
                        temp_range[:, 4],  # Range of temperature at 850 hPa
                        temp_ldt.squeeze() / 168,  # Leadtime
                        temp_bi.T,  # Base intensity
                    ]
                ).T

                scalars = np.vstack(
                    [
                        temp_bi.T,
                        temp_pos.T,
                        temp_ldt.squeeze() / 168,
                    ]
                ).T

                with torch.no_grad():
                    for model in models_to_load:
                        if model["tag"] not in plotting_dict[name].keys():
                            plotting_dict[name][model["tag"]] = []

                        if model["deep"]:
                            temp_x = torch.tensor(temp_ai, dtype=torch.float32).to(
                                calc_device
                            )
                            temp_scalars = torch.tensor(
                                scalars, dtype=torch.float32
                            ).to(calc_device)
                            temp_out = model["model"](temp_x, temp_scalars).numpy()
                        else:
                            temp_x = torch.tensor(temp_lin_in, dtype=torch.float32).to(
                                calc_device
                            )
                            temp_out = model["model"](temp_x).numpy()
                        if model["probabilistic"]:
                            temp_means = temp_out[:, [0, 2]]
                            temp_sigma = np.abs(temp_out[:, [1, 2]])
                            temp_sigma = target_scaler.inverse_transform(temp_sigma)
                            temp_means = target_scaler.inverse_transform(temp_means)
                            temp_base_adder = base_scaler.inverse_transform(temp_bi)
                            temp_means = temp_means + temp_base_adder
                            temp_95_lower = temp_means - 1.96 * temp_sigma
                            temp_95_upper = temp_means + 1.96 * temp_sigma
                        else:
                            temp_means = temp_out
                            temp_means = target_scaler.inverse_transform(temp_means)
                            temp_95_lower = temp_out
                            temp_95_upper = temp_out

                        plotting_dict[name][model["tag"]].append(
                            {
                                "means": temp_means,
                                "95_lower": temp_95_lower,
                                "95_upper": temp_95_upper,
                                "time": temp_targettime,
                            }
                        )
        np.save(f"{results_dir}case_study_animation_dict.npy", plotting_dict)
    else:
        plotting_dict = np.load(
            f"{results_dir}case_study_animation_dict.npy", allow_pickle=True
        ).item()


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

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Initialize the plot with the reference timeseries
    (ib_wind_ref,) = ax[0].plot(
        truth_time, truth_int[:, 0], label="IBTrACS", color=ibtracs_color
    )
    (ib_pres_ref,) = ax[1].plot(
        truth_time, truth_int[:, 1], label="IBTrACS", color=ibtracs_color
    )
    # Initialize the plot lines for each model
    lines = {
        model: [
            ax[0].plot([], [], label=model, color=model_colors[model]),
            ax[1].plot([], [], label=model, color=model_colors[model]),
        ]
        for model in model_keys
    }

    fills = {
        model: [
            ax[0].fill_between([], [], [], color=model_colors[model], alpha=0.25),
            ax[1].fill_between([], [], [], color=model_colors[model], alpha=0.25),
        ]
        for model in model_keys
    }

    ax[0].set_ylabel("Wind Speed (kt)")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_ylabel("Pressure (hPa)")
    ax[1].grid()
    ax[1].legend()
    # set x label ticks at 45 degree angle
    plt.xticks(rotation=35)
    fig.suptitle(f"{storm} forecast")
    ax[1].set_xlabel("Time")
    fig.tight_layout()

    num_frames = len(data[model_keys[0]])

    # Update function for animation
    def update(frame):

        artists = []

        ax[0].clear()
        ax[1].clear()

        # plot IBTrACS
        artists.append(
            ax[0].plot(
                truth_time, truth_int[:, 0], color=ibtracs_color, label="IBTrACS"
            )[0]
        )
        artists.append(
            ax[1].plot(
                truth_time, truth_int[:, 1], color=ibtracs_color, label="IBTrACS"
            )[0]
        )
        # Update each model's line with the corresponding timeseries data for the current frame
        for model in model_keys:
            means = data[model][frame]["means"]
            lower = data[model][frame]["95_lower"]
            upper = data[model][frame]["95_upper"]
            time = data[model][frame]["time"]

            # lines[model][1][0].remove()
            # lines[model][0][0] = ax[0].plot(
            #     time, means[:, 0], color=model_colors[model], label=model
            # )
            # lines[model][1][0] = ax[1].plot(
            #     time, means[:, 1], color=model_colors[model], label=model
            # )
            artists.append(
                ax[0].plot(time, means[:, 0], color=model_colors[model], label=model)[0]
            )
            artists.append(
                ax[1].plot(time, means[:, 1], color=model_colors[model], label=model)[0]
            )

            # # Add lines to artists list
            # artists.append(lines[model][0][0])
            # artists.append(lines[model][1][0])

            fills[model][0].remove()
            fills[model][1].remove()
            fills[model][0] = ax[0].fill_between(
                time, lower[:, 0], upper[:, 0], color=model_colors[model], alpha=0.25
            )
            fills[model][1] = ax[1].fill_between(
                time, lower[:, 1], upper[:, 1], color=model_colors[model], alpha=0.25
            )

            # Add fills to artists list
            artists.append(fills[model][0])
            artists.append(fills[model][1])

        # Set the limit of ax[0] to 1.2 times the max of the IBTrACS wind speed
        ax[0].set_ylim(10, 1.5 * np.max(truth_int[:, 0]))
        ## Set the limit of ax[1] to 1.2 times the min of the IBTrACS pressure
        ax[1].set_ylim(np.min(truth_int[:, 1]) - 75, np.max(truth_int[:, 1]) + 75)
        # Set the x limit of both axes to the min and max of the IBTrACS time
        ax[0].set_xlim(np.min(truth_time), np.max(truth_time))

        ax[0].set_ylabel("Wind Speed (kt)")
        ax[0].grid()
        # Make the legend in the top left corner
        ax[0].legend(loc="upper left")

        ax[1].set_ylabel("Pressure (hPa)")
        ax[1].grid()
        # Make the legend in the bottom left corner
        ax[1].legend(loc="lower left")
        # set x label ticks at 45 degree angle
        plt.xticks(rotation=35)
        fig.suptitle(f"{storm} forecast")
        ax[1].set_xlabel("Time")
        # fig.tight_layout()

        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=range(num_frames), blit=True, interval=400
    )

    fig_dir = os.path.join(results_dir, "case_study_anims")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Save the animation as a GIF file
    ani.save(os.path.join(fig_dir, f"{storm}_prediction.gif"), writer="pillow")


#     for leadtime in leadtime_set:

#         # make prediction plot
#         fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
#         ax[0].plot(truth_time, truth_int[:, 0], color=ibtracs_color, label="IBTrACS")
#         ax[1].plot(truth_time, truth_int[:, 1], color=ibtracs_color, label="IBTrACS")

#         for model in model_keys:
#             means = data[model][leadtime]["means"]
#             lower = data[model][leadtime]["95_lower"]
#             upper = data[model][leadtime]["95_upper"]
#             time = data[model][leadtime]["time"]
#             # plot the mean and 95% confidence interval
#             ax[0].plot(time, means[:, 0], color=model_colors[model], label=model)
#             ax[0].fill_between(
#                 time, lower[:, 0], upper[:, 0], color=model_colors[model], alpha=0.25
#             )
#             ax[1].plot(time, means[:, 1], color=model_colors[model], label=model)
#             ax[1].fill_between(
#                 time, lower[:, 1], upper[:, 1], color=model_colors[model], alpha=0.25
#             )

#         plt.savefig(
#             os.path.join(fig_dir, f"{storm}_{leadtime}_hour_forecast.png"), dpi=200
#         )
#         plt.show()
#         plt.close()

# # %%
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# # Sample reference timeseries data
# ref_timeseries = np.linspace(0, 10, 100)

# # Sample data for each model's timeseries
# my_dict = {
#     "model1": [np.sin(ref_timeseries + i) for i in range(5)],
#     "model2": [np.cos(ref_timeseries + i) for i in range(5)],
# }

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Initialize the plot with the reference timeseries
# (line_ref,) = ax.plot(
#     ref_timeseries, np.zeros_like(ref_timeseries), label="Reference", color="black"
# )

# # Initialize the plot lines for each model
# lines = {
#     model: ax.plot(ref_timeseries, np.zeros_like(ref_timeseries), label=model)[0]
#     for model in my_dict
# }

# # Set plot limits and labels
# ax.set_xlim(0, 10)
# ax.set_ylim(-2, 2)
# ax.set_xlabel("Time")
# ax.set_ylabel("Value")
# ax.legend()


# # Update function for animation
# def update(frame):
#     # Update the reference line (if needed)
#     line_ref.set_ydata(np.sin(ref_timeseries + frame / 10.0))

#     # Update each model's line with the corresponding timeseries data for the current frame
#     for model, timeseries_list in my_dict.items():
#         lines[model].set_data(x_data, y_data)

#     return [line_ref] + list(lines.values())


# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True)

# from IPython.display import HTML

# # Display the animation in Jupyter Notebook
# HTML(ani.to_jshtml())

# # Display the animation
# plt.show()


# %%
