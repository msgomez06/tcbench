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
    initial_times = []
    name, year = ("DORIAN", 2019)

    print(f"Processing {name} {year}")
    data = full_data[full_data.ISO_TIME.dt.year == year]
    storm = data[data.NAME == name]
    track = toolbox.tc_track(
        UID=storm.SID.iloc[0],
        NAME=storm.NAME.iloc[0],
        track=storm[["LAT", "LON"]].to_numpy(),
        timestamps=storm.ISO_TIME.to_numpy(),
        ALT_ID=storm[constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")].iloc[
            0
        ],
        wind=storm[constants.ibtracs_cols._track_cols__metadata.get("WIND")].to_numpy(),
        pres=storm[constants.ibtracs_cols._track_cols__metadata.get("PRES")].to_numpy(),
        datadir_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha",
        storm_season=storm.SEASON.iloc[0],
        ai_model="panguweather",
    )

    # %%
    # generate gaussian noise with numpy having the same shape as the track
    # and add it to the track

    def get_noise(source_track):
        track_noise = np.random.normal(0, 10 / 3, source_track.track.shape)
        wind_noise = np.random.normal(0, 5, source_track.wind.shape)
        pressure_noise = np.random.normal(0, 5, source_track.pressure.shape)
        return track_noise, wind_noise, pressure_noise

    num_members = 10

    # make 120 hr prediction windows from the track
    # 6 hr intervals

    initial_times = pd.to_datetime(track.timestamps)
    initial_times = initial_times[np.isin(initial_times.hour, [0, 12])]

    leadtimes = np.arange(6, 5 * 24 + 6, 6).astype("timedelta64[h]")

    valid_times = initial_times.to_numpy()[:, None] + leadtimes

    # tile the valid times to create num_members
    tiled_valid_times = np.tile(valid_times, (num_members, 1, 1))

    # tile the initial times to create num_members * num_leadtimes
    tiled_inits = np.tile(
        initial_times.to_numpy()[:, None], (num_members, 1, leadtimes.shape[0])
    )

    member_ids = np.zeros_like(tiled_inits).astype(int)
    for i in range(num_members):
        member_ids[i] = i

    # Make a dataframe with SID, Initial Time, and Valid Time
    df = pd.DataFrame(
        {
            "SID": [track.uid] * tiled_inits.size,
            "Initial Time": tiled_inits.flatten(),
            "Valid Time": tiled_valid_times.flatten(),
            "ensemble_idx": member_ids.flatten(),
            "wind max": np.zeros(tiled_inits.size),
            "pressure min": np.zeros(tiled_inits.size),
            "lat": np.zeros(tiled_inits.size),
            "lon": np.zeros(tiled_inits.size),
        }
    )

    # %%
    for idx, init_time in enumerate(initial_times):
        val_times = valid_times[idx]
        forecast = np.zeros((val_times.shape[0], 4))

        bool_mask = np.isin(track.timestamps, val_times)
        temp_wind = track.wind[bool_mask]
        temp_pressure = track.pressure[bool_mask]
        temp_track = track.track[bool_mask]
        # if the length of temp_wind is less than the forecast length, pad temP vars with last value
        if len(temp_wind) < forecast.shape[0] and len(temp_wind) > 0:
            temp_wind = np.pad(
                temp_wind, (0, forecast.shape[0] - len(temp_wind)), "edge"
            )
            temp_pressure = np.pad(
                temp_pressure, (0, forecast.shape[0] - len(temp_pressure)), "edge"
            )
            temp_track = np.pad(
                temp_track, ((0, forecast.shape[0] - len(temp_track)), (0, 0)), "edge"
            )
        elif len(temp_wind) == 0:
            # Default to persistence if no data is available
            init_mask = np.isin(
                track.timestamps, init_time.to_numpy().astype("datetime64[ns]")
            )
            temp_wind = np.full(forecast.shape[0], track.wind[init_mask][0])
            temp_pressure = np.full(forecast.shape[0], track.pressure[init_mask][0])
            temp_track = np.full((forecast.shape[0], 2), track.track[init_mask][0])

        forecast[:, 0] = temp_wind
        forecast[:, 1] = temp_pressure
        forecast[:, 2:] = temp_track

        # tile the forecast to create num_members
        forecast = np.tile(forecast, (num_members, 1, 1))

        # add noise to the forecast
        for i in range(num_members):
            track_noise, wind_noise, pressure_noise = get_noise(track)

            if (
                len(wind_noise[bool_mask]) < forecast[i, :, 0].shape[0]
                and len(wind_noise[bool_mask]) > 0
            ):
                # pad the noise with last value
                wind_noise = np.pad(
                    wind_noise[bool_mask],
                    (0, forecast.shape[1] - len(wind_noise[bool_mask])),
                    "edge",
                )
                pressure_noise = np.pad(
                    pressure_noise[bool_mask],
                    (0, forecast.shape[1] - len(pressure_noise[bool_mask])),
                    "edge",
                )
                track_noise = np.pad(
                    track_noise[bool_mask],
                    ((0, forecast.shape[1] - len(track_noise[bool_mask])), (0, 0)),
                    "edge",
                )
            elif len(wind_noise[bool_mask]) == 0:
                # Default to persistent noise
                init_mask = np.isin(
                    track.timestamps, init_time.to_numpy().astype("datetime64[ns]")
                )
                wind_noise = np.full(forecast[i, :, 0].shape, wind_noise[init_mask])
                pressure_noise = np.full(
                    forecast[i, :, 0].shape, pressure_noise[init_mask]
                )
                track_noise = np.full(
                    (forecast[i, :, 0].shape[0], 2), track_noise[init_mask]
                )
            else:
                wind_noise = wind_noise[bool_mask]
                pressure_noise = pressure_noise[bool_mask]
                track_noise = track_noise[bool_mask]

            forecast[i, :, 0] += wind_noise
            forecast[i, :, 1] += pressure_noise
            forecast[i, :, 2:] += track_noise

            # boolean mask to locate data in the dataframe
            bmask = df["Initial Time"] == init_time
            bmask = np.logical_and(bmask, np.isin(df["Valid Time"], val_times))
            bmask = np.logical_and(bmask, df["ensemble_idx"] == i)
            # assign the forecast to the dataframe
            df.loc[bmask, "wind max"] = forecast[i, :, 0]
            df.loc[bmask, "pressure min"] = forecast[i, :, 1]
            df.loc[bmask, "lat"] = forecast[i, :, 2]
            df.loc[bmask, "lon"] = forecast[i, :, 3]

    # save the dataframe to a csv file
    df.to_csv(
        f"{results_dir}synthetic_tracks/{name}_{year}_synthetic_tracks.csv",
        index=False,
    )
    # save the dataframe to a pickle file
    df.to_pickle(
        f"{results_dir}synthetic_tracks/{name}_{year}_synthetic_tracks.pkl",
        index=False,
    )

    # %%
    times = pd.to_datetime(track.timestamps)
    valid_times = times[np.isin(times.hour, [0, 6, 12, 18])]

    start_date = valid_times[0]
    end_date = valid_times[-1]

    num_steps = ((end_date - start_date).to_numpy() / np.timedelta64(6, "h")).astype(
        int
    )
    leadtimes = np.arange(
        -4 * 5 * 6,  # Start 5 days before
        num_steps * 6,  # End at the number of steps * 6 hours
        6,  # 6 hour intervals
    ).astype("timedelta64[h]")
    dates = start_date.to_numpy() + leadtimes

    # Generate 6-hourly leadtimes between the start and end dates

    # # Generate the leadtimes up to 5 days before with 6 hour intervals
    # leadtimes = np.arange(-5 * 24, 0, 6).astype("timedelta64[h]")
    # genesis_dates = start_date.to_numpy() + leadtimes
    # genesis_dates = genesis_dates.astype("datetime64[ns]")

    # valid_times = valid_times.to_numpy()

    # valid_times = np.hstack([genesis_dates, valid_times])
    # valid_times = np.unique(valid_times)

    # # Check that the valid times are continuous
    # assert np.all(
    #     np.diff(valid_times).astype("timedelta64[h]") == np.timedelta64(6, "h")
    # )

    # print(valid_times.shape)

    initial_times.append(dates)

    unique_times = np.unique(np.concatenate(initial_times))
    np.save(f"{results_dir}Initial_Times.npy", unique_times)
