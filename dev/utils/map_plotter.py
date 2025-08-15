# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# data_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha/temp_graphcast/graphcast_predictions_20230101000000_120.nc"
# data_path = (
#     "/scratch/mgomezd1/gcast_outputs/graphcast_predictions_20230521000000_120.nc"
# )


save_dir = "/scratch/mgomezd1/graphcast_map_plots"

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created save directory: {save_dir}")

# ds = xr.open_dataset(data_path)
ds = xr.open_mfdataset(
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/data/ERA5 *.nc"
)

idx = 0

# %%
# add the 10m wind speed as a new variable
ds["V10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
# add the long name and units
ds["V10"].attrs["long_name"] = "10m Wind Speed"
ds["V10"].attrs["units"] = "m s-1"


# %%
var = "V10"
# var = "10m_u_component_of_wind"

# Create a figure with Cartopy and Robinson projection
fig = plt.figure(figsize=(14, 5), dpi=200)
ax = plt.axes(projection=ccrs.Robinson())

# Set extent globally and add coastlines
ax.set_global()
ax.coastlines()

ds[var].isel(
    {
        # "lead-time": idx,
        "valid_time": idx,
        # "batch": 0,
    }
).plot.imshow(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    add_colorbar=False,
)

# make ax title blank
ax.set_title("")

# set facecolor to transparent
ax.set_facecolor("none")
fig.patch.set_facecolor("none")

fig.suptitle("ERA5 10m Wind Speed", fontsize=14)
fig.tight_layout()

# Save the figure
save_path = os.path.join(save_dir, f"{var}_era_map_plot.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

# %%
# do the same for MSLP
var = "msl"
# var = "mean_sea_level_pressure"
# Create a figure with Cartopy and Robinson projection
fig = plt.figure(figsize=(14, 5), dpi=200)
ax = plt.axes(projection=ccrs.Robinson())
# Set extent globally and add coastlines
ax.set_global()
ax.coastlines()
ds[var].isel(
    {
        # "lead-time": idx,
        "valid_time": idx,
        # "batch": 0,
    }
).plot.imshow(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="cividis",
    add_colorbar=False,
)
# make ax title blank
ax.set_title("")
# set facecolor to transparent
ax.set_facecolor("none")
fig.patch.set_facecolor("none")
fig.suptitle("ERA5 Mean Sea Level Pressure", fontsize=14)
# Save the figure
save_path = os.path.join(save_dir, f"{var}_era_map_plot.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

# %% z500
var = "z"
# var = "geopotential"
# Create a figure with Cartopy and Robinson projection
fig = plt.figure(figsize=(14, 5), dpi=200)
ax = plt.axes(projection=ccrs.Robinson())
# Set extent globally and add coastlines
ax.set_global()
ax.coastlines()
ds[var].isel(
    {
        # "lead-time": idx,
        "valid_time": idx,
        # "batch": 0,
    }
).sel(pressure_level=500).plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
    add_colorbar=False,
    levels=20,
)
# make ax title blank
ax.set_title("")
# set facecolor to transparent
ax.set_facecolor("none")
fig.patch.set_facecolor("none")
fig.suptitle("Graphcast Geopotential at 500 hPa", fontsize=14)
# Save the figure
save_path = os.path.join(save_dir, f"{var}_era_map_plot.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

# %% 850 temperature
var = "t"
# var = "temperature"

# Create a figure with Cartopy and Robinson projection
fig = plt.figure(figsize=(14, 5), dpi=200)
ax = plt.axes(projection=ccrs.Robinson())
# Set extent globally and add coastlines
ax.set_global()
ax.coastlines()
ds[var].isel(
    {
        # "lead-time": idx,
        "valid_time": idx,
        # "batch": 0,
    }
).sel({"pressure_level": 850}).plot.imshow(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdYlBu_r",
    add_colorbar=False,
)
# make ax title blank
ax.set_title("")
# set facecolor to transparent
ax.set_facecolor("none")
fig.patch.set_facecolor("none")
fig.suptitle("ERA5 850 hPa Temperature", fontsize=14)
# Save the figure
save_path = os.path.join(save_dir, f"{var}_era_map_plot.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
# %%
