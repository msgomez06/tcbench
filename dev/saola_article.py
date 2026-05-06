"""
Saola combined panel generator
--------------------------------
Creates a single integrated figure for a tropical cyclone case that combines:
 - a map with IBTrACS observations and model tracks
 - two time-series (intensity and MSLP)
 - a bottom row with coarse analysis (CCMP) and deep-learning downscaled wind fields

"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio

from case_studies_paper import (
    read_ibtracs_dir,
    normalize_obs_df,
    load_tcbench_results,
    # plot_case_panel,  # removed as requested
)
from case_studies_paper import normalize_fc_df, _subset_fc_sid, _make_basemap, _plot_model_tracks, _plot_timeseries, _nice_extent
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter

# Unified styling for a single-panel look
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
})

# Enable verbose debug output when True
DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Toggle: enable plotting of ensemble members/envelopes (load ensemble CSVs)
ENSEMBLE_MODE = False

# Global plotting font sizes (increase for readability)
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "legend.fontsize": 15,
})

# -------------------
# Config (edit here)
# -------------------
# Fixed init date for Saola case study
INIT_TIME = "2023-08-31 12:00"   # fixed init for Saola
SID        = "2023234N18128"       
NAME       = "Saola"
YEAR       = 2023

RESULTS_DIR = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/TCBench Results/"
IBTRACS_DIR = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/"

# GeoTIFFs (CCMP = LR, DL = SR)
LR_TIFF = "/work/FAC/FGSE/IDYST/tbeucler/default/sam/tcbench/Saoloa_testing/figure_Typhoon_Saola/LR/CCMP_Wind_Analysis_20230901_V03_merge3_3.tif"
SR_TIFF = "/work/FAC/FGSE/IDYST/tbeucler/default/sam/tcbench/Saoloa_testing/figure_Typhoon_Saola/SR/CCMP_Wind_Analysis_20230901_V03_merge3_3.tif"
STATIONS_TXT = "/work/FAC/FGSE/IDYST/tbeucler/default/sam/tcbench/Saoloa_testing/figure_Typhoon_Saola/station_loc.txt"

END_TIME  = "2023-09-03 12:00"  # cut off at Sep-03 12:00 UTC
OUT_PNG   = "saola_combined.pdf"

# Models to show (robust mapping for probabilistic label)
REQUESTED_MODELS = [
    "AI Post-Processing",  # FourCastNet v2 post-processing (from CSV)
    "TIGGE-IFS",            # Physics-based
    "FourCastNet",          # Deterministic AI
    "GENC",                 # Probabilistic AI
]

# ------------------------------------------------------------------
# Helper functions (I/O, small plotting helpers)
# ------------------------------------------------------------------
def _read_tiff_with_coords(tif_path):
    """Return (u, v, ws), extent, (lons, lats) for 1/2/3-band GeoTIFFs.
       - 3 bands => (u,v,ws)
       - 2 bands => (u,v) and ws = hypot(u,v)
       - 1 band  => (None,None,ws)  (no vectors available)
    """
    import rasterio
    import numpy as np

    with rasterio.open(tif_path) as src:
        count = src.count
        bands = [src.read(i).astype(float) for i in range(1, count + 1)]
        transform = src.transform
        width, height = src.width, src.height
        xmin, ymin, xmax, ymax = src.bounds

    # 1D lon/lat from affine (assumes no rotation; typical for these tiles)
    xs = np.arange(width)
    ys = np.arange(height)
    lons = transform[2] + xs * transform[0]
    lats = transform[5] + ys * transform[4]

    u = v = None
    if count >= 3:
        u, v, ws = bands[0], bands[1], bands[2]
    elif count == 2:
        u, v = bands
        ws = np.hypot(u, v)
    elif count == 1:
        ws = bands[0]
    else:
        raise ValueError(f"{tif_path} has no readable bands")

    return (u, v, ws), (xmin, xmax, ymin, ymax), (lons, lats)


def _plot_downscaling_row(fig, gs_cell, lr_tif, sr_tif, stations_txt=None, cmap="turbo"):
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.PlateCarree()
    subgs = gs_cell.subgridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    ax_l = fig.add_subplot(subgs[0, 0], projection=proj)
    ax_r = fig.add_subplot(subgs[0, 1], projection=proj)

    # Read both TIFFs (robust 1/2/3-band)
    (uL, vL, wsL), extentL, (lonsL, latsL) = _read_tiff_with_coords(lr_tif)
    (uR, vR, wsR), extentR, (lonsR, latsR) = _read_tiff_with_coords(sr_tif)

    def _add_coast(ax):
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e6df", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0b1a3c", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="white", alpha=0.85, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.6)
        try:
            gl.top_labels = gl.right_labels = False
        except Exception:
            pass

    # Shared color scale
    finite_max = np.nanmax([np.nanmax(wsL), np.nanmax(wsR)])
    vmax = 40.0 if np.isfinite(finite_max) and finite_max > 40 else max(30.0, float(finite_max))
    vmin = 0.0

    # Left: CCMP
    _add_coast(ax_l)
    ax_l.set_extent(extentL, crs=proj)
    imL = ax_l.imshow(wsL, extent=extentL, origin="upper", transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    if (uL is not None) and (vL is not None):
        step = 2
        Xl, Yl = np.meshgrid(lonsL, latsL)
        ax_l.quiver(
            Xl[::step, ::step], Yl[::step, ::step],
            uL[::step, ::step], vL[::step, ::step],
            transform=proj, color="k", width=0.0025, scale=700, zorder=3,
        )
    ax_l.text(0.02, 0.98, "(d)  Near Real-Time Coarse Analysis Wind Field", transform=ax_l.transAxes, ha="left", va="top",
              fontsize=16, 
              bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.2"))

    # Right: Deep Learning Wind
    _add_coast(ax_r)
    ax_r.set_extent(extentR, crs=proj)
    imR = ax_r.imshow(wsR, extent=extentR, origin="upper", transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    if (uR is not None) and (vR is not None):
        step = 2
        Xr, Yr = np.meshgrid(lonsR, latsR)
        ax_r.quiver(
            Xr[::step, ::step], Yr[::step, ::step],
            uR[::step, ::step], vR[::step, ::step],
            transform=proj, color="k", width=0.0025, scale=700, zorder=3,
        )

    ax_r.text(0.02, 0.98, "(e) Deep Learning Downscaled Wind Magnitude", transform=ax_r.transAxes, ha="left", va="top",
              fontsize=16, 
              bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.2"))

    # Optional station markers (lat lon per row)
    if stations_txt and os.path.exists(stations_txt):
        try:
            loc = np.loadtxt(stations_txt)
            ax_l.scatter(loc[:, 1], loc[:, 0], s=10, c="red", edgecolor="k", linewidths=0.3, transform=proj, zorder=4)
            ax_r.scatter(loc[:, 1], loc[:, 0], s=10, c="red", edgecolor="k", linewidths=0.3, transform=proj, zorder=4)
        except Exception:
            pass

    # Shared colorbar will be created by the caller; return imR for its mappable
    return (ax_l, ax_r), imR


# ------------------------------------------------------------------
# Initialization selection utilities
# ------------------------------------------------------------------
def _pick_init(fc_df, sid, init_str):
    """Return a dataframe filtered to the requested init if present.
    If the exact init is missing, pick the closest init on that date with max model coverage."""
    df = fc_df.copy()
    # Find time columns
    init_cols = [c for c in ["init_time", "Initial Time", "init", "InitialTime"] if c in df.columns]
    if not init_cols:
        return df  # fallback
    ic = init_cols[0]
    df[ic] = pd.to_datetime(df[ic], errors="coerce", utc=True)

    target = pd.to_datetime(init_str, utc=True)
    # Exact match?
    exact = df[(df["sid"] == sid) & (df[ic] == target)]
    if not exact.empty:
        return exact

    # Same date, closest hour & most models
    same_date = df[(df["sid"] == sid) & (df[ic].dt.date == target.date())]
    if same_date.empty:
        return df[df["sid"] == sid]
    counts = same_date.groupby(ic)["model"].nunique().reset_index(name="n")
    best_init = counts.sort_values(["n", ic], ascending=[False, True]).iloc[0][ic]
    return same_date[same_date[ic] == best_init]


# ------------------------------------------------------------------
# DATA LOADING
# - Read IBTrACS
# - Load TCBench results
# - Append post-processing and probabilistic CSVs (if present)
# ------------------------------------------------------------------
ib_raw = read_ibtracs_dir(IBTRACS_DIR)
obs    = normalize_obs_df(ib_raw)

fc = load_tcbench_results(RESULTS_DIR, year=YEAR, verbose=False)


# --- Integrate FourCastNet v2 post-processing CSV (normalize schema, append) ---
PP_CSV = os.path.join(RESULTS_DIR, "postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU,_M_2023.csv")
if os.path.exists(PP_CSV):
    try:
        _pp = pd.read_csv(PP_CSV)
        # Rename columns to TCBench normalized schema
        _pp = _pp.rename(columns={
            'SID': 'sid',
            'Initial Time': 'init_time',
            'Valid Time': 'valid_time',
            'Lead Time (h)': 'lead_h',
            'wind max': 'vmax_kt',   # assuming units are knots
            'pres min': 'mslp_hpa',
            'ensemble_idx': 'member',
        })
        # Parse datetimes to UTC and coerce
        for _tc in ['init_time', 'valid_time']:
            if _tc in _pp.columns:
                _pp[_tc] = pd.to_datetime(_pp[_tc], errors='coerce', utc=True)
        # Ensure required columns exist
        for req in ['sid', 'init_time', 'valid_time', 'lead_h', 'vmax_kt', 'mslp_hpa']:
            if req not in _pp.columns:
                raise ValueError(f"Post-processing CSV missing required column: {req}")
        # Add model label used downstream and keep member if present
        _pp['model'] = 'AI Post-Processing'
        if 'member' not in _pp.columns:
            _pp['member'] = 0
        # Concatenate into fc
        fc = pd.concat([fc, _pp], ignore_index=True, sort=False)
        debug_print(f"[debug] Appended post-proc rows: {len(_pp)}; fc total rows now: {len(fc)}")
    except Exception as e:
        debug_print("[debug] Failed to append post-proc CSV:", e)
else:
    debug_print("[debug] Post-proc CSV not found at:", PP_CSV)

# --- Integrate GENC ensemble and mean CSVs if present ---
GENC_ENSEMBLE_CSV = os.path.join(RESULTS_DIR, "2023_GENC.csv")
GENC_MEAN_CSV = os.path.join(RESULTS_DIR, "2023_GENC_results.csv")

# Helper to normalize GENC-style CSVs to the fc schema
def _load_and_normalize_genc(path, set_member_nan=False, member_val=None):
    df = pd.read_csv(path)
    # Rename known columns to normalized ones
    colmap = {
        'SID': 'sid',
        'Initial Time': 'init_time',
        'Valid Time': 'valid_time',
        'ensemble_idx': 'member',
        'wind max': 'vmax_kt',
        'pressure min': 'mslp_hpa',
        'lat': 'lat',
        'lon': 'lon',
        'Hour': 'lead_h',
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    # Parse times
    for tc in ['init_time', 'valid_time']:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], errors='coerce', utc=True)
    # If member column missing or requested override
    if 'member' not in df.columns:
        if set_member_nan:
            df['member'] = member_val if member_val is not None else -999
        else:
            df['member'] = 0
    else:
        # coerce to int where possible
        try:
            df['member'] = pd.to_numeric(df['member'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass
    return df

# Append ensemble members (if present) — only if ENSEMBLE_MODE is enabled
if ENSEMBLE_MODE:
    if os.path.exists(GENC_ENSEMBLE_CSV):
        try:
            genc_ens = _load_and_normalize_genc(GENC_ENSEMBLE_CSV)
            # label model
            genc_ens['model'] = 'GENC'
            fc = pd.concat([fc, genc_ens], ignore_index=True, sort=False)
            debug_print(f"[debug] Appended GENC ensemble rows: {len(genc_ens)}; fc total rows now: {len(fc)}")
        except Exception as e:
            debug_print("[debug] Failed to append GENC ensemble CSV:", e)
    else:
        debug_print("[debug] GENC ensemble CSV not found at:", GENC_ENSEMBLE_CSV)
else:
    debug_print("[debug] ENSEMBLE_MODE is False: skipping GENC ensemble CSV loading")

# Append mean results (if present) — tag with a distinct member value
if os.path.exists(GENC_MEAN_CSV):
    try:
        genc_mean = _load_and_normalize_genc(GENC_MEAN_CSV, set_member_nan=True, member_val=-999)
        genc_mean['model'] = 'GENC'
        genc_mean['member'] = -999
        fc = pd.concat([fc, genc_mean], ignore_index=True, sort=False)
        debug_print(f"[debug] Appended GENC mean rows: {len(genc_mean)}; fc total rows now: {len(fc)}")
    except Exception as e:
        debug_print("[debug] Failed to append GENC mean CSV:", e)
else:
    debug_print("[debug] GENC mean CSV not found at:", GENC_MEAN_CSV)

# Filter to the requested init (or the closest same-date init with best coverage)
fc = _pick_init(fc, SID, INIT_TIME)

# Discover available models for this SID/init
avail = list(fc.loc[fc["sid"] == SID, "model"].dropna().unique())

# Also show what load_tcbench_results found for this SID/init
debug_print("[debug] Models available for SID at chosen init:", avail)
try:
    debug_print("[debug] Sample rows for post-proc label in loaded df (first 3):")
    sample_pp = fc[(fc['sid'] == SID) & (fc['model'] == 'AI Post-Processing')].head(3)
    debug_print(sample_pp.to_string(index=False))
except Exception as e:
    debug_print("[debug] Could not sample post-proc from loaded df:", e)

# Map requested labels to what exists
models = []
alias_map = {
    # canonical names used within this script
    'AI Post-Processing': 'AI Post-Processing',
    # filename that is present on disk for FCN v2 post-processing
    'postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU,_M': 'AI Post-Processing',
    # deterministic / physics / probabilistic
    'FourCastNet': 'FourCastNet',
    'TIGGE-IFS': 'TIGGE-IFS',
    'GENC': 'GENC',
    # legacy Pangu label (kept for completeness)
    'PANGU_ANN': 'PANGU_ANN',
}
for wanted in REQUESTED_MODELS:
    m = alias_map.get(wanted, wanted)
    if m in avail:
        models.append(m)

# Unique, keep order
seen = set(); models = [m for m in models if not (m in seen or seen.add(m))]
# Debug: show source CSVs if present in dataframe; otherwise at least show post-proc file
src_cols = [c for c in ['source', 'file', 'filepath', 'path', 'csv_path'] if c in fc.columns]
if src_cols:
    try:
        src_col = src_cols[0]
        df_dbg = fc[(fc['sid'] == SID) & (fc['model'].isin(models))][['model', src_col]].drop_duplicates()
        debug_print('[debug] Source files by model:')
        for m, g in df_dbg.groupby('model'):
            debug_print('  -', m, ':', ', '.join(sorted(g[src_col].astype(str).unique())[:3]))
    except Exception as e:
        debug_print('[debug] Could not list source files by model:', e)
# Always report the post-proc CSV path we attempted to append
debug_print('[debug] Post-proc source CSV:', PP_CSV)
if not models:
    debug_print("[warn] No requested models available at the chosen init; falling back to any available.")
    models = avail

print("Models used on top panel:", models)

# ------------------------------------------------------------------
# PLOTTING: build a single integrated figure (map + time-series + downscaling)
# ------------------------------------------------------------------

# Normalize forecasts and subset for the chosen SID/init
fc_norm = normalize_fc_df(fc)
df_fc_used, chosen_init, models_used = _subset_fc_sid(fc_norm, SID, models, INIT_TIME)
end_cutoff = pd.to_datetime(END_TIME, utc=True) if END_TIME is not None else None

# Observations
df_obs_sid = obs[obs["sid"] == SID].copy()
if df_obs_sid.empty:
    raise ValueError(f"No obs rows for SID {SID}")
df_obs_sid_map = df_obs_sid[df_obs_sid["time"] >= chosen_init].copy()
if end_cutoff is not None:
    df_obs_sid_map = df_obs_sid_map[df_obs_sid_map["time"] <= end_cutoff]
if df_obs_sid_map.empty:
    df_obs_sid_map = df_obs_sid.copy()

# Forecasts cropped for map drawing
df_fc_used_map = df_fc_used.copy()
if "valid_time" in df_fc_used_map.columns:
    df_fc_used_map = df_fc_used_map[df_fc_used_map["valid_time"] >= chosen_init]
    if end_cutoff is not None:
        df_fc_used_map = df_fc_used_map[df_fc_used_map["valid_time"] <= end_cutoff]

# Map extent from obs + forecast track points (tighter view)
obs_lats = df_obs_sid_map["lat"].to_numpy()
obs_lons = df_obs_sid_map["lon"].to_numpy()
if "lat" in df_fc_used_map.columns and "lon" in df_fc_used_map.columns:
    fc_lats = df_fc_used_map["lat"].to_numpy(); fc_lons = df_fc_used_map["lon"].to_numpy()
    fc_lats = fc_lats[~np.isnan(fc_lats)]; fc_lons = fc_lons[~np.isnan(fc_lons)]
else:
    fc_lats = np.array([]); fc_lons = np.array([])
lats = np.concatenate([obs_lats[~np.isnan(obs_lats)], fc_lats])
lons = np.concatenate([obs_lons[~np.isnan(obs_lons)], fc_lons])
extent = _nice_extent(lats, lons, pad_deg=0.7)
xmin, xmax, ymin, ymax = extent
extent = (108.0, 118.0, ymin, ymax)

# Build top panel using the common helper, then reposition axes to make
# the map exactly the same height as the combined right-side plots.
fig, ax_map, ax_vmax, ax_mslp, ax_cbar, proj = _make_basemap(extent)
# Allow map to freely resize (avoid fixed aspect interfering with set_position)
try:
    ax_map.set_aspect('auto', adjustable='box')
except Exception:
    try:
        ax_map.set_aspect('auto')
        ax_map.set_adjustable('box')
    except Exception:
        pass
fig.set_size_inches(16, 18, forward=True)
if ax_cbar is not None:
    ax_cbar.set_visible(False)

# Draw model tracks (right column lead_step default of 6 hours)
model_handles = _plot_model_tracks(ax_map, df_fc_used_map, models_used, ccrs.PlateCarree(), lead_step=6)

# Draw IBTrACS in monochrome (remove wind-speed coloring distraction)
df_line = df_obs_sid_map.dropna(subset=["lat","lon"]).copy()
ax_map.plot(df_line["lon"], df_line["lat"], transform=ccrs.PlateCarree(), color="k", linewidth=1.6, zorder=4)
ax_map.scatter(df_line["lon"], df_line["lat"], transform=ccrs.PlateCarree(), s=14, facecolor="white", edgecolor="k", linewidths=0.8, zorder=5)

# Single, larger legend on the map only
ib_handle = Line2D([0],[0], marker="o", linestyle="-", color="k", markersize=7, label="Observations")

# Remap model legend labels to requested terminology
legend_label_map = {
    'TIGGE-IFS': 'Physics-based',
    'FourCastNet': 'Deterministic AI',
    'AI Post-Processing': 'AI Post-Processing',
    'GENC': 'Probabilistic AI',
}
if model_handles:
    for h in model_handles:
        try:
            lbl = h.get_label()
            if lbl in legend_label_map:
                h.set_label(legend_label_map[lbl])
        except Exception:
            pass

# Make AI Post-Processing share the same color as Deterministic AI (FourCastNet)
# and use a dashed linestyle so the two are distinguishable.
det_color = None
pp_labels = {
    'AI Post-Processing',
    'ANN (postprocess)',
    'Postprocess',
    'postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU,_M',
    'postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU_M',
    'FourCastNetV2_ANN',
}
if model_handles:
    # find the Deterministic AI color from its legend handle
    for h in model_handles:
        if h.get_label() == "Deterministic AI":
            try:
                det_color = h.get_color()
            except Exception:
                det_color = None
            break
    # apply color + dashed linestyle to AI Post-Processing
    if det_color is not None:
        for h in model_handles:
            if h.get_label() == "AI Post-Processing":
                try:
                    h.set_color(det_color)
                except Exception:
                    pass
                try:
                    h.set_linestyle('--')
                except Exception:
                    pass
        # Also update any map lines for the post-processing label
        try:
            for ln in ax_map.get_lines():
                if ln.get_label() in pp_labels:
                    ln.set_color(det_color)
                    ln.set_linestyle('--')
        except Exception:
            pass
# Build ordered legend: Observations, Physics-based, Deterministic AI, Probabilistic AI, AI Post-Processing
label_order = [
    "Observations",
    "Physics-based",
    "Deterministic AI",
    "Probabilistic AI",
    "AI Post-Processing",
]
handle_map = {}
handle_map["Observations"] = ib_handle
if model_handles:
    for h in model_handles:
        handle_map[h.get_label()] = h
handles = [handle_map[lbl] for lbl in label_order if lbl in handle_map]
if handles:
    ax_map.legend(handles=handles, loc="lower right", frameon=True, fontsize=13)

# Subtitles for uniform look across all panels
_subtitle_bbox = dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.2")
ax_map.text(0.02, 0.98, "(a)  Tropical Cyclone Track Forecast", transform=ax_map.transAxes,
            ha="left", va="top", fontsize=16,  bbox=_subtitle_bbox)


# Plot time series
_plot_timeseries(ax_vmax, df_obs_sid, df_fc_used, models_used,
                 var_obs="wind_kts", var_fc="vmax_kt", ylabel="Max wind (kt)",
                 start_time=chosen_init, end_time=end_cutoff)
ax_vmax.text(0.98, 0.98, "(b)  Intensity Forecast", transform=ax_vmax.transAxes,
             ha="right", va="top", fontsize=16,  bbox=_subtitle_bbox)
_plot_timeseries(ax_mslp, df_obs_sid, df_fc_used, models_used,
                 var_obs="mslp_hpa", var_fc="mslp_hpa", ylabel="Sea-level pressure (hPa)",
                 start_time=chosen_init, end_time=end_cutoff)
ax_mslp.text(0.02, 0.98, "(c)  Minimum Sea Level Pressure", transform=ax_mslp.transAxes,
             ha="left", va="top", fontsize=16,  bbox=_subtitle_bbox)

# Remove legends from time-series (only want the map legend)
for ax in (ax_vmax, ax_mslp):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# Harmonize time-series styling:
#  - AI Post-Processing shares the color of Deterministic AI
#  - AI Post-Processing uses a dashed linestyle so it is distinguishable
def _harmonize_ts(ax_list, det_color_hint=None):
    det_labels = {"Deterministic AI", "FourCastNet"}
    pp_labels = {
        "AI Post-Processing",
        "ANN (postprocess)",
        "Postprocess",
        "postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU,_M",
        "postprocessing_fourcastnetv2_0shot_ANN_LeakyReLU_M",
        "FourCastNetV2_ANN",
    }
    # Find deterministic AI color from plotted lines if possible
    det_col = None
    for ax in ax_list:
        for ln in ax.get_lines():
            if ln.get_label() in det_labels:
                try:
                    det_col = ln.get_color()
                except Exception:
                    det_col = None
                if det_col is not None:
                    break
        if det_col is not None:
            break
    if det_col is None:
        det_col = det_color_hint  # fall back to map legend color if available
    if det_col is None:
        return  # nothing to do
    # Apply color + dashed linestyle to post-processing lines
    for ax in ax_list:
        for ln in ax.get_lines():
            if ln.get_label() in pp_labels:
                try:
                    ln.set_color(det_col)
                    ln.set_linestyle("--")
                except Exception:
                    pass

_harmonize_ts([ax_vmax, ax_mslp], det_color_hint=det_color)

# --- Probabilistic ensemble shading for GENC (if available) ---
if ENSEMBLE_MODE:
    try:
        if 'model' in df_fc_used.columns and 'GENC' in df_fc_used['model'].unique():
            df_genc = df_fc_used[df_fc_used['model'] == 'GENC'].copy()
            if not df_genc.empty and 'member' in df_genc.columns:
                # Normalize time column
                if 'valid_time' in df_genc.columns:
                    df_genc['valid_time'] = pd.to_datetime(df_genc['valid_time'], utc=True)
                    # Group by valid_time and compute envelope
                    g = df_genc.groupby('valid_time')
                    if 'vmax_kt' in df_genc.columns:
                        vmax_min = g['vmax_kt'].min()
                        vmax_max = g['vmax_kt'].max()
                    else:
                        vmax_min = vmax_max = None
                    if 'mslp_hpa' in df_genc.columns:
                        mslp_min = g['mslp_hpa'].min()
                        mslp_max = g['mslp_hpa'].max()
                    else:
                        mslp_min = mslp_max = None
                    # find color for Probabilistic AI from model_handles (if available)
                    prob_color = None
                    try:
                        if model_handles:
                            for h in model_handles:
                                if h.get_label() == 'Probabilistic AI':
                                    try:
                                        prob_color = h.get_color()
                                    except Exception:
                                        prob_color = None
                                    break
                    except Exception:
                        prob_color = None
                    # Only draw if we have a color and data
                    if prob_color is not None:
                        if vmax_min is not None and len(vmax_min):
                            times = vmax_min.index.to_pydatetime()
                            ax_vmax.fill_between(times, vmax_min.values, vmax_max.values, color=prob_color, alpha=0.15, zorder=1)
                        if mslp_min is not None and len(mslp_min):
                            times_p = mslp_min.index.to_pydatetime()
                            ax_mslp.fill_between(times_p, mslp_min.values, mslp_max.values, color=prob_color, alpha=0.15, zorder=1)
    except Exception:
        pass

# --- Ensemble envelope for AI Post-Processing (if available) ---
if ENSEMBLE_MODE:
    try:
        if 'model' in df_fc_used.columns and 'AI Post-Processing' in df_fc_used['model'].unique():
            df_pp = df_fc_used[df_fc_used['model'] == 'AI Post-Processing'].copy()
            if not df_pp.empty and 'member' in df_pp.columns:
                # Ensure valid_time is datetime
                if 'valid_time' in df_pp.columns:
                    df_pp['valid_time'] = pd.to_datetime(df_pp['valid_time'], utc=True)
                    gpp = df_pp.groupby('valid_time')
                    # compute min/max across members for vmax and mslp
                    vmax_min_pp = gpp['vmax_kt'].min() if 'vmax_kt' in df_pp.columns else None
                    vmax_max_pp = gpp['vmax_kt'].max() if 'vmax_kt' in df_pp.columns else None
                    mslp_min_pp = gpp['mslp_hpa'].min() if 'mslp_hpa' in df_pp.columns else None
                    mslp_max_pp = gpp['mslp_hpa'].max() if 'mslp_hpa' in df_pp.columns else None
                    # determine color: prefer the recolored det_color if set, otherwise inspect model_handles
                    pp_color = None
                    try:
                        if det_color is not None:
                            pp_color = det_color
                        elif model_handles:
                            for h in model_handles:
                                if h.get_label() == 'AI Post-Processing' or h.get_label() == 'Deterministic AI':
                                    try:
                                        pp_color = h.get_color()
                                    except Exception:
                                        pp_color = None
                                    break
                    except Exception:
                        pp_color = None
                    # draw shading on time-series if color and data present
                    if pp_color is not None:
                        if vmax_min_pp is not None and len(vmax_min_pp):
                            tvals = vmax_min_pp.index.to_pydatetime()
                            ax_vmax.fill_between(tvals, vmax_min_pp.values, vmax_max_pp.values, color=pp_color, alpha=0.15, zorder=1)
                        if mslp_min_pp is not None and len(mslp_min_pp):
                            tvals_p = mslp_min_pp.index.to_pydatetime()
                            ax_mslp.fill_between(tvals_p, mslp_min_pp.values, mslp_max_pp.values, color=pp_color, alpha=0.15, zorder=1)
    except Exception:
        pass

# Determine right-bound and share ticks
if end_cutoff is not None:
    right_bound = end_cutoff
else:
    end_obs = df_obs_sid[df_obs_sid["time"] >= chosen_init]["time"].max()
    end_fc_col = df_fc_used.get("valid_time")
    end_fc_max = end_fc_col.max() if end_fc_col is not None else None
    right_bound = max([t for t in [end_obs, end_fc_max] if pd.notna(t)]) if (pd.notna(end_obs) or end_fc_max is not None) else (chosen_init + pd.Timedelta(hours=120))
for ax in (ax_vmax, ax_mslp):
    ax.set_xlim(left=chosen_init, right=right_bound)
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    if "left" in ax.spines: ax.spines["left"].set_visible(False)

# Shared 12-hour ticks and custom labels on bottom axis
tick_times = pd.date_range(start=pd.to_datetime(chosen_init), end=pd.to_datetime(right_bound), freq="12H", tz="UTC")
tick_nums = mdates.date2num(tick_times.to_pydatetime())
for ax in (ax_vmax, ax_mslp):
    ax.xaxis.set_major_locator(FixedLocator(tick_nums))
# Formatter: same on both, but labels shown on top for ax_vmax and bottom for ax_mslp
def _fmt_12h_or_date(x, pos):
    dt = mdates.num2date(x); hour = dt.hour
    if hour == 12: return "12:00"
    if hour == 0: return dt.strftime("%b-%d")
    return ""
ax_mslp.xaxis.set_major_formatter(FuncFormatter(_fmt_12h_or_date))
ax_vmax.xaxis.set_major_formatter(FuncFormatter(_fmt_12h_or_date))
# Show ticks/labels on top of panel b (ax_vmax)
ax_vmax.xaxis.set_ticks_position('top')
ax_vmax.xaxis.set_label_position('top')
ax_vmax.tick_params(labelbottom=False, labeltop=True)
# Ensure bottom panel keeps labels on bottom
ax_mslp.xaxis.set_ticks_position('bottom')
ax_mslp.tick_params(labelbottom=True)
# --- Layout: make the map the same height as (ax_vmax + ax_mslp) ---
# Use symmetrical column widths so the map and right column have the same horizontal extent
outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 2.2], hspace=0.12)
top_pos = outer[0, 0].get_position(fig)
LEFT, RIGHT = 0.08, 0.92
gap = 0.04
# Use symmetrical column widths so the map and right column have the same horizontal extent
col_width = (RIGHT - LEFT - gap) / 2.0

# Stack right axes in the right column with equal width to the map
right_x0 = LEFT + col_width + gap
ax_vmax.set_position([right_x0, top_pos.y0 + top_pos.height*0.52, col_width, top_pos.height*0.44])
ax_mslp.set_position([right_x0, top_pos.y0,                      col_width, top_pos.height*0.44])

# Now set the map to span exactly the combined height of the two right axes, with the same width
y0 = ax_mslp.get_position().y0
y1 = ax_vmax.get_position().y0 + ax_vmax.get_position().height
map_height = y1 - y0
ax_map.set_position([LEFT, y0, col_width, map_height])

"""
# --- Init time vertical marker (shared across both time-series) ---
# Draw a red dashed line at the first tick (chosen_init) on both axes
for ax in (ax_vmax, ax_mslp):
    ax.axvline(x=chosen_init, color='red', linestyle='--', linewidth=1.2, zorder=6)

# Place a red math-style "$t_{init}$" label centered between the two plots, aligned to x=chosen_init
# Convert x in data coords (ax_mslp) to figure coords for precise horizontal placement
x_disp, _ = ax_mslp.transData.transform((mdates.date2num(pd.to_datetime(chosen_init)), 0))
x_fig = fig.transFigure.inverted().transform((x_disp, 0))[0]
pos_v = ax_vmax.get_position(); pos_m = ax_mslp.get_position()
y_mid = pos_m.y1 + (pos_v.y0 - pos_m.y1) * 0.5
fig.text(x_fig, y_mid, r"$t_{\mathrm{init}}$", color='red', ha='center', va='center', fontsize=12, fontweight='bold')
# Add centered valid-time label between the two plots
right_pos = ax_mslp.get_position()
x_center = right_pos.x0 + right_pos.width * 0.5
fig.text(x_center, y_mid, r"$\leftarrow\ t_{\mathrm{valid}}\ \rightarrow$", ha='center', va='center', fontsize=12, fontweight='bold')
"""
# --- Bottom row: downscaling pair ---
(ax_ccmp, ax_dl), im = _plot_downscaling_row(fig, outer[1, 0], LR_TIFF, SR_TIFF, stations_txt=STATIONS_TXT)
# Match top-panel gap and column widths to keep perfect vertical alignment
pL = ax_ccmp.get_position(); pR = ax_dl.get_position()
gap_lr = gap
half = col_width
ax_ccmp.set_position([LEFT, pL.y0, half, pL.height])
ax_dl.set_position([LEFT + half + gap_lr, pR.y0, half, pR.height])

# Synchronize right margins across all right-column axes using ax_dl as reference
right_limit = ax_dl.get_position().x1
for ax in (ax_vmax, ax_mslp):
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, right_limit - pos.x0, pos.height])


# Centered reference-time label between the two plots (no init vertical marker)
# Place a centered "← REFERENCE (TARGET) TIME →" label between the Intensity and MSLP plots
pos_v = ax_vmax.get_position(); pos_m = ax_mslp.get_position()
y_mid = pos_m.y1 + (pos_v.y0 - pos_m.y1) * 0.5
right_pos = ax_mslp.get_position()
x_center = right_pos.x0 + right_pos.width * 0.5
fig.text(x_center, y_mid, u"\u2190 Reference (Target) Time \u2192", ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Shared colorbar aligned to the same LEFT and right_limit
cbar_left = LEFT
cbar_right = right_limit
cbar_width = cbar_right - cbar_left
cax = fig.add_axes([cbar_left, 0.06, cbar_width, 0.02])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Wind Speed (m/s)")

 # No suptitle (figure intended to be embedded/annotated externally)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=220)
plt.close(fig)

print(f"Saved → {Path(OUT_PNG).resolve()}")
# %%
