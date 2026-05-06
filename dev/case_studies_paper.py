# %%
"""
Case-study panel plots for TCBench storms (clean IMT presentation version)

What you get:
- A single function `plot_case_panel(...)` that produces a publication-ready
  panel for a given SID (e.g., Otis 2023294N09264):
    • Left: map with IBTrACS track (colored by wind) and model tracks
    • Right-top: Max wind speed (kt) vs valid time (obs in black, models colored)
    • Right-bottom: Sea-level pressure (hPa) vs valid time (obs in black, models colored)
- Utilities to normalize both IBTrACS-like observation tables and TCBench
  result CSVs, including ensemble aggregation.
- Graceful fallbacks when intensity/pressure are missing in the forecast files.

Expected columns (flexible, aliases handled):
  Observations: sid, time, lat, lon, (wind), (pressure)
  Forecasts:    sid, model, init_time, lead_h, lat, lon, (vmax/wind), (mslp/pressure), (member)

Author: TCBench / IMT presentation edition
"""
from __future__ import annotations

from pathlib import Path
import re
import os
import json
from typing import Iterable, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------------------------------------------
# Color management
# --------------------------------------------------------------------------------------
FIXED_MODEL_COLORS = {
    "GENC": "#E5C300",
    "PANGU": "#FF6F61",
    "TIGGE-GEFS": "#1F77B4",
    "TIGGE-IFS": "#FF7F0E",
    "AIFS": "#2CA02C",
    "FourCastNet": "#9467BD",
    "FGN": "#17BECF",
}
_tab10 = list(mpl.cm.get_cmap("tab10").colors)
_tab20 = list(mpl.cm.get_cmap("tab20").colors)
_FALLBACK_PALETTE = _tab10 + _tab20
_DEF_COLOR_CACHE: dict[str, tuple] = {}


def _model_color(name: str):
    if name in FIXED_MODEL_COLORS:
        return FIXED_MODEL_COLORS[name]
    if name in _DEF_COLOR_CACHE:
        return _DEF_COLOR_CACHE[name]
    idx = len(_DEF_COLOR_CACHE) % len(_FALLBACK_PALETTE)
    _DEF_COLOR_CACHE[name] = _FALLBACK_PALETTE[idx]
    return _DEF_COLOR_CACHE[name]


# --------------------------------------------------------------------------------------
# Column helpers & normalization
# --------------------------------------------------------------------------------------


def _pick(
    df: pd.DataFrame, candidates: Sequence[str], required: bool = True
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")
    return None


def normalize_obs_df(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Return obs with canonical columns: sid, time(UTC), lat, lon, wind_kts, mslp_hpa."""
    obs = obs_df.copy()

    sid = _pick(obs, ["sid", "SID"])
    t = _pick(obs, ["time", "ISO_TIME", "iso_time"])
    la = _pick(obs, ["lat", "LAT"])
    lo = _pick(obs, ["lon", "LON"])
    wi = _pick(
        obs,
        ["wind_kts", "usa_wind", "USA_WIND", "WMO_WIND", "wind_max", "VMAX", "VMAX_KT"],
        required=False,
    )
    pr = _pick(
        obs,
        [
            "mslp_hpa",
            "mslp",
            "PRES",
            "usa_pres",
            "USA_PRES",
            "WMO_PRES",
            "central_pressure",
            "MIN_MSLP",
        ],
        required=False,
    )

    obs = obs.rename(columns={sid: "sid", t: "time", la: "lat", lo: "lon"})
    if wi:
        obs = obs.rename(columns={wi: "wind_kts"})
    if pr:
        obs = obs.rename(columns={pr: "mslp_hpa"})

    obs["time"] = pd.to_datetime(obs["time"], errors="coerce", utc=True)
    for c in ["lat", "lon", "wind_kts", "mslp_hpa"]:
        if c in obs.columns:
            obs[c] = pd.to_numeric(obs[c], errors="coerce")
    obs["lon"] = ((obs["lon"] + 180) % 360) - 180

    # Ensure presence of value columns
    if "wind_kts" not in obs:
        obs["wind_kts"] = np.nan
    if "mslp_hpa" not in obs:
        obs["mslp_hpa"] = np.nan

    # Convert Pa -> hPa for obs if needed
    if "mslp_hpa" in obs.columns:
        med = np.nanmedian(pd.to_numeric(obs["mslp_hpa"], errors="coerce"))
        if med > 2000:
            obs["mslp_hpa"] = pd.to_numeric(obs["mslp_hpa"], errors="coerce") / 100.0

    return obs.sort_values(["sid", "time"])[
        ["sid", "time", "lat", "lon", "wind_kts", "mslp_hpa"]
    ]


def normalize_fc_df(fc_df: pd.DataFrame) -> pd.DataFrame:
    """Return fc with canonical columns: sid, model, init_time, valid_time, lead_h, lat, lon, vmax_kt, mslp_hpa, (member)."""
    fc = fc_df.copy()

    sid = _pick(fc, ["sid", "SID"])
    la = _pick(fc, ["lat", "LAT"], required=False)
    lo = _pick(fc, ["lon", "LON"], required=False)

    init = _pick(
        fc,
        ["init_time", "Initial Time", "INIT_TIME", "t0", "init", "INIT"],
        required=False,
    )
    valid = _pick(
        fc, ["valid_time", "Valid Time", "time", "ISO_TIME", "VT"], required=False
    )
    lead = _pick(
        fc,
        ["lead_h", "lead_hours", "Hour", "tau", "LEAD_H", "LEAD", "Lead Time (h)"],
        required=False,
    )
    mdl = _pick(fc, ["model", "name", "MODEL"], required=False)
    mem = _pick(
        fc,
        [
            "member",
            "MEMBER",
            "ens_member",
            "ensemble",
            "ENS",
            "mem",
            "member_id",
            "ensemble_idx",
        ],
        required=False,
    )

    vmax = _pick(
        fc,
        ["vmax_kt", "vmax", "VMAX", "MAXWIND", "wind_kts", "wind", "WIND"],
        required=False,
    )
    mslp = _pick(
        fc,
        ["mslp_hpa", "mslp", "MSLP", "PRES", "pressure", "min_mslp", "MIN_MSLP"],
        required=False,
    )

    ren = {sid: "sid", la: "lat", lo: "lon"}
    if init:
        ren[init] = "init_time"
    if valid:
        ren[valid] = "valid_time"
    if lead:
        ren[lead] = "lead_h"
    if mdl:
        ren[mdl] = "model"
    if mem:
        ren[mem] = "member"
    if vmax:
        ren[vmax] = "vmax_kt"
    if mslp:
        ren[mslp] = "mslp_hpa"
    fc = fc.rename(columns=ren)

    # --- Heuristics: if vmax_kt/mslp_hpa were not found via explicit aliases,
    # try to infer from column names (avoiding score/error columns).
    import re

    if "vmax_kt" not in fc.columns:
        wind_like = [c for c in fc.columns if re.search(r"vmax|wind", c, re.IGNORECASE)]
        wind_like = [
            c
            for c in wind_like
            if not re.search(r"crps|ae|se|dpe|error|ri", c, re.IGNORECASE)
        ]
        if wind_like:
            fc = fc.rename(columns={wind_like[0]: "vmax_kt"})
    if "mslp_hpa" not in fc.columns:
        pres_like = [
            c
            for c in fc.columns
            if re.search(r"mslp|pres|pressure|pmin|min[_ ]?mslp", c, re.IGNORECASE)
        ]
        pres_like = [
            c
            for c in pres_like
            if not re.search(r"crps|ae|se|dpe|error|ri", c, re.IGNORECASE)
        ]
        if pres_like:
            fc = fc.rename(columns={pres_like[0]: "mslp_hpa"})

    fc["sid"] = fc["sid"].astype(str)
    for c in ["lat", "lon", "lead_h", "vmax_kt", "mslp_hpa"]:
        if c in fc.columns:
            fc[c] = pd.to_numeric(fc[c], errors="coerce")
    # Convert Pa -> hPa if needed
    if "mslp_hpa" in fc.columns:
        med = np.nanmedian(pd.to_numeric(fc["mslp_hpa"], errors="coerce"))
        if med > 2000:  # values look like Pa
            fc["mslp_hpa"] = pd.to_numeric(fc["mslp_hpa"], errors="coerce") / 100.0
    if "lon" in fc.columns:
        fc["lon"] = ((fc["lon"] + 180) % 360) - 180

    if "init_time" in fc:
        fc["init_time"] = pd.to_datetime(fc["init_time"], errors="coerce", utc=True)
    if "valid_time" in fc:
        fc["valid_time"] = pd.to_datetime(fc["valid_time"], errors="coerce", utc=True)

    # Derive lead or valid time if missing
    if ("lead_h" not in fc or fc["lead_h"].isna().all()) and {
        "init_time",
        "valid_time",
    } <= set(fc.columns):
        fc["lead_h"] = (
            (fc["valid_time"] - fc["init_time"]).dt.total_seconds() / 3600.0
        ).round()
    if "valid_time" not in fc and {"init_time", "lead_h"} <= set(fc.columns):
        fc["valid_time"] = fc["init_time"] + pd.to_timedelta(fc["lead_h"], unit="h")

    if "model" not in fc:
        fc["model"] = "(unknown)"
    if "member" in fc:
        fc["member"] = fc["member"].astype(str)

    keep = ["sid", "model", "init_time", "valid_time", "lead_h"]
    if "lat" in fc.columns:
        keep.append("lat")
    if "lon" in fc.columns:
        keep.append("lon")
    if "vmax_kt" in fc:
        keep.append("vmax_kt")
    if "mslp_hpa" in fc:
        keep.append("mslp_hpa")
    if "member" in fc:
        keep.append("member")

    fc = fc.copy()[keep]
    return fc


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def load_tcbench_results(
    results_dir: str | Path,
    year: int = 2023,
    only_models: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load per-model CSVs like `2023_aifs.csv`, skipping *_RI and *results.*"""
    p = Path(results_dir)
    if not p.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    frames: List[pd.DataFrame] = []
    skip_re = re.compile(r"(_RI|results)", re.IGNORECASE)

    # Gather candidate CSVs: standard ("{year}_*.csv") and postprocessing-like ("*_YEAR.csv")
    cand_files = set(p.glob(f"{year}_*.csv")) | set(p.glob(f"*_{year}.csv"))
    for f in sorted(cand_files):
        name = f.name
        if skip_re.search(name):
            if verbose:
                print(f"[skip] {name} (RI/results)")
            continue
        stem = f.stem
        # Determine a model_key from the filename for bookkeeping
        if name.startswith(f"{year}_"):
            model_key = stem.split("_", 1)[1]
        elif name.endswith(f"_{year}.csv"):
            model_key = stem[: -(len(f"_{year}"))]  # strip trailing _YEAR
        else:
            model_key = stem

        # Human-friendly display & a simple filter key used when only_models is provided
        disp = None
        filter_key = model_key
        if model_key.lower().startswith("postprocessing"):
            up = name.upper()
            low = name.lower()
            # Special naming for Pangu postprocessing variants
            if ("pangu" in low) or ("panguweather" in low):
                if "ANN" in up:
                    disp = "PANGU_ANN"
                    filter_key = "pangu_ann"
                elif "UNET" in up:
                    disp = "PANGU_UNET"
                    filter_key = "pangu_unet"
                elif "MLR" in up:
                    disp = "PANGU_MLR"
                    filter_key = "pangu_mlr"
                else:
                    disp = "PANGU_POSTPROC"
                    filter_key = "pangu_postproc"
            else:
                if "ANN" in up:
                    disp = "ANN (postprocess)"
                    filter_key = "ann"
                elif "RF" in up or "RANDOMFOREST" in up:
                    disp = "RF (postprocess)"
                    filter_key = "rf"
                else:
                    disp = "Postprocess"
                    filter_key = "postprocess"

        # Respect only_models if provided (match against multiple keys)
        if only_models:
            wanted = {m.lower() for m in only_models}
            keys_here = {model_key.lower(), filter_key.lower()}
            if disp:
                keys_here.add(disp.lower())
            if keys_here.isdisjoint(wanted):
                continue
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e1:
            # Try the conservative streaming fallback
            if verbose:
                print(f"[warn] {name}: {e1}\n       attempting fallback reader...")
            try:
                df = _read_csv_fallback(f, year=year, verbose=verbose)
            except Exception as e2:
                if verbose:
                    try:
                        cols = list(pd.read_csv(f, nrows=0).columns)
                    except Exception:
                        cols = []
                    print(f"[skip] {name}: fallback failed: {e2}\n       columns: {cols}")
                continue
            if df.empty:
                if verbose:
                    print(f"[skip] {name}: fallback produced 0 rows after filtering")
                continue
        # Stamp model then normalize
        df["model"] = disp or model_key
        try:
            norm = normalize_fc_df(df)
        except Exception as e3:
            if verbose:
                print(f"[skip] {name}: normalize failed: {e3}")
            continue
        frames.append(norm)
        if verbose:
            print(f"[ok]   {name}: {len(norm):,} rows")

    if not frames:
        raise RuntimeError("No usable model CSVs were found.")

    fc = pd.concat(frames, ignore_index=True)

    # Pretty-name map
    pretty = {
        "fcnet": "FourCastNet",
        "aifs": "AIFS",
        "pangu": "Pangu",
        "genc": "GENC",
        "weatherlab_FNV3": "FNV3",
        "TIGGE_IFS": "TIGGE-IFS",
        "TIGGE_GEFS": "TIGGE-GEFS",
    }
    fc["model"] = fc["model"].map(lambda m: pretty.get(m, m))
    return fc


# --------------------------------------------------------------------------------------
# --- Fallback reader for environments where pandas+NumPy compatibility raises
#     "module 'numpy' has no attribute 'matrix'" during read/normalize.

def _read_csv_fallback(
    csv_path: str | Path,
    year: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Stream a CSV in chunks with conservative settings and minimal inference.

    - Uses engine='python', dtype=str, on_bad_lines='skip'.
    - Optionally filters rows to a given *calendar year* using the most common
      datetime-like columns if present, to keep memory reasonable.
    """
    p = Path(csv_path)
    use_cols = None  # let pandas decide; we handle missing columns downstream
    frames: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(
            p,
            engine="python",
            dtype=str,
            chunksize=200_000,
            on_bad_lines="skip",
            low_memory=True,
        ):
            # Optional year filter on common time columns
            if year is not None:
                for tcol in ("Initial Time", "Valid Time", "init_time", "valid_time", "ISO_TIME"):
                    if tcol in chunk.columns:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            dt = pd.to_datetime(chunk[tcol], errors="coerce", utc=True)
                        chunk = chunk.loc[dt.dt.year == int(year)]
                        break  # filter using the first matching column
            if not chunk.empty:
                frames.append(chunk)
    except Exception as e:
        raise RuntimeError(f"fallback reader failed: {e}") from e

    if not frames:
        # Return an empty DF with no columns; caller will decide to skip
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    # Light canonicalization: ensure SID column is stringy if present
    for c in ("SID", "sid"):
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out

# --------------------------------------------------------------------------------------
# Targeted loader(s)
# --------------------------------------------------------------------------------------


def load_specific_results(
    csv_path: str | Path, display_name: Optional[str] = None, verbose: bool = True
) -> pd.DataFrame:
    """Load and normalize a single result CSV, stamping the given display name.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV containing per-lead forecasts for one model (any schema handled by `normalize_fc_df`).
    display_name : str | None
        Name to use in plots/legends (e.g., "ANN (postprocess)"). Defaults to file stem.
    verbose : bool
        Print a small summary upon success.
    """
    f = Path(csv_path)
    if not f.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    try:
        raw = pd.read_csv(f, low_memory=False)
        model_name = display_name or f.stem
        raw["model"] = model_name
        norm = normalize_fc_df(raw)
        if verbose:
            print(f"[ok] {f.name}: {len(norm):,} rows as '{model_name}'")
        return norm
    except Exception as e:
        try:
            cols = list(pd.read_csv(f, nrows=0).columns)
        except Exception:
            cols = []
        raise RuntimeError(f"Failed to load {f.name}: {e}. Columns: {cols}")


# --------------------------------------------------------------------------------------
# Lightweight IBTrACS reader (CSV)
# --------------------------------------------------------------------------------------

def read_ibtracs_dir(tracks_path: str | Path) -> pd.DataFrame:
    """Read IBTrACS CSV files from a directory (or a single CSV), return one DataFrame.

    This avoids importing the heavy `utils.toolbox` module which pulls dask/numba/scipy
    and can trigger environment recursion errors in some setups.

    Parameters
    ----------
    tracks_path : str | Path
        Either a directory containing one or more `*.csv` / `*.csv.gz` files, or
        a direct path to a single CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw IBTrACS table as concatenated DataFrame. Column normalization is handled
        later by `normalize_obs_df` in this module.
    """
    p = Path(tracks_path)
    if not p.exists():
        raise FileNotFoundError(f"IBTrACS path not found: {tracks_path}")

    files: list[Path]
    if p.is_file():
        if p.suffix.lower() not in {".csv", ".gz"} and not p.name.endswith(".csv.gz"):
            raise ValueError(f"Unsupported file type for IBTrACS: {p.name}")
        files = [p]
    else:
        files = sorted(list(p.glob("*.csv")) + list(p.glob("*.csv.gz")))
        if not files:
            raise FileNotFoundError(
                f"No IBTrACS CSV files found in: {tracks_path} (looked for *.csv, *.csv.gz)"
            )

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception as e:
            print(f"[warn] Could not read {f.name}: {e}")
            continue
    if not frames:
        raise RuntimeError("Failed to read any IBTrACS CSV files.")

    ib = pd.concat(frames, ignore_index=True)

    # Light parsing so downstream filters are fast & robust
    # Ensure these common columns exist in a parseable form if present
    time_cols = [c for c in ["ISO_TIME", "time", "iso_time"] if c in ib.columns]
    for tcol in time_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ib[tcol] = pd.to_datetime(ib[tcol], errors="coerce", utc=True)

    # Numeric coercions for frequent fields (safe if missing)
    for c in [
        "LAT", "lat", "LON", "lon",
        "USA_WIND", "WMO_WIND", "WIND", "wind_kts", "usa_wind", "WMO_PRES",
        "USA_PRES", "PRES", "mslp", "mslp_hpa",
    ]:
        if c in ib.columns:
            ib[c] = pd.to_numeric(ib[c], errors="coerce")

    if "SID" in ib.columns:
        ib["SID"] = ib["SID"].astype(str)

    return ib


def load_ann_postprocess(csv_path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Convenience wrapper for ANN postprocessing model."""
    return load_specific_results(
        csv_path, display_name="ANN (postprocess)", verbose=verbose
    )


# --------------------------------------------------------------------------------------
# Linestyle helper for postprocessing models
# --------------------------------------------------------------------------------------
def _model_linestyle(name: str):
    n = (name or "").lower()
    if any(tok in n for tok in ["postprocess", "postproc", "ann", "unet", "mlr"]):
        return (0, (3, 2))  # dashed
    return "-"

# --- Model grouping for legend --------------------------------------------------------
def _model_group(name: str) -> str:
    """Return one of {'Physical', 'AI', 'Postprocess'} based on model name."""
    n = (name or "").lower()
    # Post-processing variants (e.g., PANGU_ANN / _UNET / _MLR, generic 'postprocess')
    if _is_postproc(name):
        return "Postprocess"
    # Common AI models
    if any(tok in n for tok in ["fourcast", "pangu", "aifs", "fgn", "ai","fnv","genc"]):
        return "AI"
    # Default bucket
    return "Physical"


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------


def _nice_extent(
    lats: np.ndarray, lons: np.ndarray, pad_deg: float = 5.0
) -> Tuple[float, float, float, float]:
    return (
        float(np.nanmin(lons) - pad_deg),
        float(np.nanmax(lons) + pad_deg),
        float(np.nanmin(lats) - pad_deg),
        float(np.nanmax(lats) + pad_deg),
    )


def _make_basemap(extent=None):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 9), dpi=160)
    # 3 rows x 2 cols: map spans left column rows 0–1; right column has two stacked time series; bottom row = colorbar
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.2, 1.0],
        height_ratios=[1.0, 1.0, 0.07],
        wspace=0.18,
        hspace=0.25,
    )

    ax_map = fig.add_subplot(gs[0:2, 0], projection=proj)
    ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e6df")
    ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0b1a3c")
    ax_map.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.6,
        edgecolor="white",
        alpha=0.85,
    )
    ax_map.gridlines(draw_labels=True, linewidth=0.2, alpha=0.6)
    if extent:
        ax_map.set_extent(extent, crs=ccrs.PlateCarree())

    ax_vmax = fig.add_subplot(gs[0, 1])
    ax_mslp = fig.add_subplot(gs[1, 1], sharex=ax_vmax)
    ax_cbar = fig.add_subplot(gs[2, :])
    return fig, ax_map, ax_vmax, ax_mslp, ax_cbar, proj


def _aggregate_by_lead(df: pd.DataFrame, lead_step: Optional[int] = 6) -> pd.DataFrame:
    df = df.copy()
    df["lead_h"] = pd.to_numeric(df["lead_h"], errors="coerce").round().astype(int)
    if lead_step is not None:
        df = df[df["lead_h"] % int(lead_step) == 0]
    # Average across members/duplicates per lead
    num_cols = [c for c in ["lat", "lon", "vmax_kt", "mslp_hpa"] if c in df.columns]
    grouped = df.groupby("lead_h", as_index=False)[num_cols].mean(numeric_only=True)
    # stamp metadata from the group
    grouped["model"] = df["model"].iloc[0]
    grouped["init_time"] = df["init_time"].iloc[0]
    grouped["valid_time"] = grouped["init_time"] + pd.to_timedelta(
        grouped["lead_h"], unit="h"
    )
    return grouped


def _choose_init(
    df_fc_sid: pd.DataFrame,
    which_models: Optional[Sequence[str]],
    requested: Optional[str],
) -> pd.Timestamp:
    if requested is not None:
        return pd.to_datetime(requested, utc=True)
    if which_models is None:
        which_models = list(df_fc_sid["model"].dropna().unique())
    counts = (
        df_fc_sid[df_fc_sid["model"].isin(which_models)]
        .groupby("init_time")["model"]
        .nunique()
    )
    if counts.empty:
        return pd.to_datetime(df_fc_sid["init_time"].min(), utc=True)
    max_models = counts.max()
    candidate_times = counts[counts == max_models].index
    return pd.to_datetime(max(candidate_times))


def _subset_fc_sid(
    df_fc: pd.DataFrame,
    sid: str,
    models: Optional[Sequence[str]],
    init_time: Optional[str],
):
    sub = df_fc[df_fc["sid"] == sid].copy()
    if sub.empty:
        raise ValueError(f"No forecast rows for SID {sid}")
    chosen = _choose_init(sub, models, init_time)
    if models is None:
        models = list(sub["model"].dropna().unique())
    have = sub.groupby("model")["init_time"].apply(lambda s: chosen in set(s))
    used_models = [m for m in models if have.get(m, False)]
    used = sub[(sub["init_time"] == chosen) & (sub["model"].isin(used_models))].copy()
    return used, chosen, used_models


def _plot_obs_track(ax, obs_sid: pd.DataFrame, proj, cmap: str = "turbo"):
    lats = obs_sid["lat"].to_numpy()
    lons = obs_sid["lon"].to_numpy()
    winds = obs_sid["wind_kts"].to_numpy()
    if np.isnan(winds).all():  # gentle gradient if no winds
        winds = np.linspace(50, 100, len(winds))
    ax.plot(lons, lats, transform=proj, color="w", alpha=0.35, lw=2.0, zorder=3)
    sc = ax.scatter(
        lons,
        lats,
        c=winds,
        s=30,
        cmap=cmap,
        vmin=float(np.nanmin(winds)),
        vmax=float(np.nanmax(winds)),
        transform=proj,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    # start/end markers
    ax.scatter(
        lons[0],
        lats[0],
        marker="^",
        s=80,
        color="white",
        transform=proj,
        zorder=6,
        edgecolor="k",
        linewidth=0.6,
    )
    ax.scatter(
        lons[-1],
        lats[-1],
        marker="*",
        s=120,
        color="white",
        transform=proj,
        zorder=6,
        edgecolor="k",
        linewidth=0.6,
    )
    return sc



# --------------------------------------------------------------------------------------
# Convenience: export three single-panel figures for one init
#   1) Physical models only
#   2) Physical + AI models
#   3) Physical + AI + Postprocess models
# Returns the list of saved file paths
# --------------------------------------------------------------------------------------

def export_case_panel_tiers(
    sid: str,
    name: str,
    obs_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    init_time: str,
    out_prefix: str = "case_panel",
    end_time: str | None = None,
    tiles: str | None = None,
    tiles_zoom: int | None = None,
    map_pad_deg: float = 1.0,
    wind_cmap: str = "turbo",
) -> list[str]:
    """Generate three static panels for the same init, gradually adding model groups.

    Parameters
    ----------
    sid, name : str
        Storm ID and display name.
    obs_df, fc_df : DataFrame
        Raw tables; they will be normalized internally by the plotting call.
    init_time : str
        UTC init timestamp like "2023-10-24 00:00".
    out_prefix : str
        Prefix (path stem) for the output PNGs. We will write
        ``{out_prefix}_phys.png``, ``{out_prefix}_phys_ai.png``,
        and ``{out_prefix}_phys_ai_post.png``.
    end_time : str | None
        UTC cutoff for map and timeseries domains.
    tiles, tiles_zoom, map_pad_deg, wind_cmap :
        Passed through to :func:`plot_case_panel` for consistent styling.

    Returns
    -------
    list[str]
        File paths of the three generated figures (only paths for which models
        existed are returned).
    """
    # Determine available models for this SID, preserving the appearance order
    fc_norm = normalize_fc_df(fc_df)
    df_sid = fc_norm[fc_norm["sid"] == str(sid)].copy()
    if df_sid.empty:
        raise ValueError(f"No forecast rows for SID {sid}")

    # Unique models while preserving order of first appearance
    seen = set()
    all_models = [m for m in df_sid["model"].tolist() if not (m in seen or seen.add(m))]

    # Partition by group
    physical = [m for m in all_models if _model_group(m) == "Physical"]
    ai_models = [m for m in all_models if _model_group(m) == "AI"]
    postproc = [m for m in all_models if _model_group(m) == "Postprocess"]

    tiers: list[tuple[str, list[str]]] = [
        ("phys", physical),
        ("phys_ai", physical + ai_models),
        ("phys_ai_post", physical + ai_models + postproc),
    ]

    out_paths: list[str] = []
    base = Path(out_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    for suffix, models in tiers:
        # Skip empty tiers gracefully (e.g., no physical models in this dataset)
        if not models:
            continue
        out_png = str(base.with_name(base.stem + f"_{suffix}").with_suffix(".png"))
        # Call the existing single-panel plotter with the filtered model list
        plot_case_panel(
            sid=sid,
            name=name,
            obs_df=obs_df,
            fc_df=fc_df,
            models=models,
            init_time=init_time,
            end_time=end_time,
            wind_cmap=wind_cmap,
            out_png=out_png,
            show=False,
            tiles=tiles,
            tiles_zoom=tiles_zoom,
            map_pad_deg=map_pad_deg,
        )
        out_paths.append(out_png)

    return out_paths

def _plot_model_tracks(
    ax, df_fc_used: pd.DataFrame, models_used: Sequence[str], proj, lead_step: int = 6
):
    handles: List[Line2D] = []
    for m in models_used:
        sub = df_fc_used[(df_fc_used["model"] == m)].sort_values("lead_h")
        color = _model_color(m)
        ls = _model_linestyle(m)
        if sub.empty:
            # still add legend handle so model appears in legend
            handles.append(Line2D([0], [0], color=color, lw=2.8, label=m, linestyle=ls))
            continue
        sub = _aggregate_by_lead(sub, lead_step=lead_step)
        has_track = (
            ("lat" in sub.columns)
            and ("lon" in sub.columns)
            and sub[["lat", "lon"]].notna().all(axis=1).any()
        )
        if has_track:
            ax.plot(
                sub["lon"],
                sub["lat"],
                transform=proj,
                color=color,
                lw=2.0,
                alpha=0.95,
                zorder=3.2,
                linestyle=ls,
            )
            ax.scatter(
                sub["lon"],
                sub["lat"],
                s=14,
                color=color,
                alpha=0.9,
                transform=proj,
                zorder=3.3,
            )
            # mark first/last
            first = sub.dropna(subset=["lat", "lon"]).iloc[0]
            last = sub.dropna(subset=["lat", "lon"]).iloc[-1]
            ax.scatter(
                [first["lon"]],
                [first["lat"]],
                marker="s",
                s=28,
                color=color,
                edgecolor="k",
                linewidth=0.3,
                transform=proj,
                zorder=3.4,
            )
            ax.scatter(
                [last["lon"]],
                [last["lat"]],
                marker="v",
                s=36,
                color=color,
                edgecolor="k",
                linewidth=0.3,
                transform=proj,
                zorder=3.4,
            )
        # Always add a legend handle
        handles.append(Line2D([0], [0], color=color, lw=2.8, label=m, linestyle=ls))
    return handles


def _plot_timeseries(
    ax,
    obs_sid: pd.DataFrame,
    df_fc_used: pd.DataFrame,
    models_used: Sequence[str],
    var_obs: str,
    var_fc: str,
    ylabel: str,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
):
    # Trim obs to start/end window
    obs_plot = obs_sid
    if start_time is not None and "time" in obs_plot.columns:
        obs_plot = obs_plot[obs_plot["time"] >= start_time]
    if end_time is not None and "time" in obs_plot.columns:
        obs_plot = obs_plot[obs_plot["time"] <= end_time]
    if var_obs in obs_plot:
        ax.plot(
            obs_plot["time"], obs_plot[var_obs], "k-", lw=2.6, label="Obs (IBTrACS)"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Obs unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    # Model series (mean per lead)
    has_any = False
    for m in models_used:
        sub = df_fc_used[df_fc_used["model"] == m].copy()
        if var_fc not in sub.columns or sub[var_fc].isna().all():
            continue
        sub = _aggregate_by_lead(sub, lead_step=None)
        if start_time is not None and "valid_time" in sub.columns:
            sub = sub[sub["valid_time"] >= start_time]
        if end_time is not None and "valid_time" in sub.columns:
            sub = sub[sub["valid_time"] <= end_time]
        if sub.empty:
            continue
        ls = _model_linestyle(m)
        ax.plot(
            sub["valid_time"],
            sub[var_fc],
            lw=1.8,
            alpha=0.9,
            label=m,
            color=_model_color(m),
            linestyle=ls,
        )
        has_any = True

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if has_any:
        ax.legend(loc="best", fontsize=8, frameon=True)


# --- Optional map tiles background (requires internet). Falls back silently if unavailable.
try:
    import cartopy.io.img_tiles as cimgt
except Exception:  # pragma: no cover
    cimgt = None


def _estimate_zoom_from_extent(extent):
    # crude zoom heuristic based on lon/lat span
    lon_span = max(1e-6, extent[1] - extent[0])
    lat_span = max(1e-6, extent[3] - extent[2])
    span = max(lon_span, lat_span)
    if span > 60:
        return 3
    if span > 30:
        return 4
    if span > 15:
        return 5
    if span > 8:
        return 6
    if span > 4:
        return 7
    return 8


def _add_tiles_background(ax, extent, provider: str = "esri", zoom: int | None = None):
    if cimgt is None:
        return False
    try:
        if provider == "esri":
            tiles = cimgt.QuadtreeTiles(
                desired_tile_form="RGB",
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            )
        elif provider == "osm":
            tiles = cimgt.QuadtreeTiles(
                desired_tile_form="RGB",
                url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            )
        else:  # stamen-terrain as a light alternative
            tiles = cimgt.Stamen("terrain-background")
        # ensure correct mode
        try:
            tiles.desired_tile_form = "RGB"
        except Exception:
            pass
        if zoom is None:
            zoom = _estimate_zoom_from_extent(extent)
        ax.add_image(tiles, zoom)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Diagnostics helper
# --------------------------------------------------------------------------------------


def _diagnose_vars(df_fc_used: pd.DataFrame, models_used: Sequence[str]) -> str:
    rows = ["Model availability (per-lead mean series):"]
    for m in models_used:
        sub = df_fc_used[df_fc_used["model"] == m]
        has_w = ("vmax_kt" in sub.columns) and sub["vmax_kt"].notna().any()
        has_p = ("mslp_hpa" in sub.columns) and sub["mslp_hpa"].notna().any()
        rows.append(
            f" - {m:12s} | wind: {'yes' if has_w else 'no '} | mslp: {'yes' if has_p else 'no '}"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------------------
# Interactive HTML (Plotly) export
# --------------------------------------------------------------------------------------


def _mpl_to_plotly_color(c):
    """Convert a Matplotlib color spec to a Plotly-friendly string."""
    try:
        if isinstance(c, str):
            # hex or named
            return c
        r, g, b = mpl.colors.to_rgb(c)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    except Exception:
        return None


def _is_postproc(name: str) -> bool:
    n = (name or "").lower()
    return any(
        tok in n for tok in ["postprocess", "postproc", "ann", "unet", "mlr"]
    )  # noqa: E501


def build_interactive_panel_html(
    sid: str,
    name: str,
    obs_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    init_time: Optional[str] = None,
    end_time: Optional[str] = None,
    html_path: str = "panel_interactive.html",
    map_lead_step: Optional[int] = 6,
    init_times: Optional[Sequence[str]] = None,
    map_style: str = "satellite-streets",
    mapbox_token: Optional[str] = None,
):
    """Export a single, self-contained interactive HTML panel (Plotly) with:
    - **Satellite map** (Mapbox) when a token is available; graceful fallback to static geo map.
    - **Init selector** (buttons) for up to 4 init times.

    Parameters
    ----------
    init_times : list[str] | None
        Explicit list of inits (UTC ISO strings) to include in the selector. If None,
        we auto-pick up to the **last 4** available inits for the SID + model set.
    map_style : str
        Mapbox style (e.g., "satellite", "satellite-streets", "outdoors"). Default is
        "satellite-streets".
    mapbox_token : str | None
        Mapbox access token. If None, uses env vars `MAPBOX_TOKEN` or `PLOTLY_MAPBOX_TOKEN`.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Plotly is required for build_interactive_panel_html(). Install plotly>=5.0"
        ) from e

    # Normalize inputs and subset
    obs = normalize_obs_df(obs_df)
    fc = normalize_fc_df(fc_df)

    obs_sid = obs[obs["sid"] == sid].copy()
    if obs_sid.empty:
        raise ValueError(f"No obs rows for SID {sid}")

    avail_models = sorted(fc.loc[fc["sid"] == sid, "model"].dropna().unique())
    if models is None:
        models = avail_models
    else:
        models = [m for m in models if m in avail_models]
        if not models:
            raise ValueError("None of the requested models are available for this SID.")

    # Determine candidate init times (up to 4)
    df_sid = fc[fc["sid"] == sid]
    unique_inits = (
        sorted(pd.to_datetime(df_sid["init_time"].dropna().unique())) if not df_sid.empty else []
    )
    if init_times is None:
        # Default: pick up to the last 4 inits where at least one requested model is present
        present = df_sid[df_sid["model"].isin(models)]
        inits_filtered = (
            sorted(pd.to_datetime(present["init_time"].dropna().unique())) if not present.empty else []
        )
        init_list = inits_filtered[-4:] if len(inits_filtered) > 4 else inits_filtered
    else:
        init_list = [pd.to_datetime(t, utc=True) for t in init_times]

    if not init_list:
        # Fallback to a single chosen init
        init_list = [
            _choose_init(df_sid, models, init_time)
        ]
    # Ensure UTC tz
    init_list = [pd.to_datetime(t, utc=True) for t in init_list]

    # If init_time explicitly provided, put it first if present
    if init_time is not None:
        it = pd.to_datetime(init_time, utc=True)
        if it in init_list:
            init_list = [it] + [t for t in init_list if t != it]

    # Enforce max 4 inits
    init_list = init_list[:4]

    end_cut = pd.to_datetime(end_time, utc=True) if end_time is not None else None

    # --- Helpers ---------------------------------------------------------------
    def _agg(sub: pd.DataFrame, step: Optional[int]):
        if sub.empty:
            return sub
        sub = sub.copy()
        sub["lead_h"] = pd.to_numeric(sub["lead_h"], errors="coerce").round()
        keep = [c for c in ["lat", "lon", "vmax_kt", "mslp_hpa"] if c in sub.columns]
        if step is not None:
            sub = sub[sub["lead_h"] % int(step) == 0]
        out = sub.groupby("lead_h", as_index=False)[keep].mean(numeric_only=True)
        out["model"] = sub["model"].iloc[0]
        out["init_time"] = sub["init_time"].iloc[0]
        out["valid_time"] = out["init_time"] + pd.to_timedelta(out["lead_h"], unit="h")
        return out

    def _center_zoom(lat_min, lat_max, lon_min, lon_max):
        clat = float((lat_min + lat_max) / 2.0)
        clon = float((lon_min + lon_max) / 2.0)
        span = max(abs(lat_max - lat_min), abs(lon_max - lon_min), 1e-6)
        # crude mapping span(deg) -> zoom
        if span > 60:
            zoom = 3
        elif span > 30:
            zoom = 4
        elif span > 15:
            zoom = 5
        elif span > 8:
            zoom = 6
        elif span > 4:
            zoom = 7
        else:
            zoom = 8
        return clat, clon, zoom

    # Decide mapping backend (Mapbox satellite if token is available)
    token = mapbox_token or os.environ.get("MAPBOX_TOKEN") or os.environ.get("PLOTLY_MAPBOX_TOKEN")
    use_mapbox = token is not None

    # Base figure with two XY panels on the right; left is a blank domain where mapbox will sit
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "domain"}, {"type": "xy"}]],
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    # Fix right-side axes titles; ranges set per-init later
    fig.update_yaxes(title_text="Sea-level pressure (hPa)", row=1, col=2)
    fig.update_yaxes(title_text="Max wind (kt)", row=2, col=2)
    fig.update_xaxes(title_text="Valid time (UTC)", row=1, col=2)
    fig.update_xaxes(title_text="Valid time (UTC)", row=2, col=2)

    # Mapbox placement on the left column domain
    if use_mapbox:
        fig.update_layout(
            mapbox=dict(
                style=map_style,
                domain=dict(x=[0.0, 0.60], y=[0.0, 1.0]),
                accesstoken=token,
            )
        )
    else:
        # No token: we'll fallback to Scattergeo *behind* using a white background and set geo domain
        fig.update_layout(
            geo=dict(domain=dict(x=[0.0, 0.60], y=[0.0, 1.0]), showcountries=True, showland=True,
                     landcolor="#e8e6df", showocean=True, oceancolor="#0b1a3c" )
        )

    # Build per-init traces for the first init; others go into frames
    frames = []

    def _build_for_init(chosen_init: pd.Timestamp):
        # Crop obs to window
        obs_win = obs_sid[obs_sid["time"] >= chosen_init]
        if end_cut is not None:
            obs_win = obs_win[obs_win["time"] <= end_cut]
        if obs_win.empty:
            obs_win = obs_sid.copy()

        # --- Build customdata for OBS hover (mslp, lat, lon, time_str)
        if "mslp_hpa" in obs_win.columns:
            _obs_mslp = obs_win["mslp_hpa"].to_numpy()
        else:
            _obs_mslp = np.full(len(obs_win), np.nan)
        _obs_time_str = obs_win["time"].dt.strftime("%Y-%m-%d %H:%MZ").to_numpy()
        _obs_cd = np.column_stack([
            _obs_mslp,
            obs_win["lat"].to_numpy(),
            obs_win["lon"].to_numpy(),
            _obs_time_str,
        ])

        # Forecasts for chosen init & models
        fc_used = fc[(fc["sid"] == sid) & (fc["init_time"] == chosen_init) & (fc["model"].isin(models))].copy()
        if not fc_used.empty and "valid_time" in fc_used.columns:
            fc_used = fc_used[fc_used["valid_time"] >= chosen_init]
            if end_cut is not None:
                fc_used = fc_used[fc_used["valid_time"] <= end_cut]

        # Aggregate
        fc_map_list, fc_ts_list = [], []
        for m in models:
            sub = fc_used[fc_used["model"] == m]
            if sub.empty:
                continue
            m_map = _agg(sub, map_lead_step)
            m_map["model"] = m
            fc_map_list.append(m_map)
            m_ts = _agg(sub, None)
            m_ts["model"] = m
            fc_ts_list.append(m_ts)
        fc_map = pd.concat(fc_map_list, ignore_index=True) if fc_map_list else pd.DataFrame()
        fc_ts = pd.concat(fc_ts_list, ignore_index=True) if fc_ts_list else pd.DataFrame()

        # Determine color scale bounds (combine obs wind + model vmax if present)
        _vals = []
        if "wind_kts" in obs_win:
            _vals.append(obs_win["wind_kts"].to_numpy())
        if not fc_map.empty and "vmax_kt" in fc_map:
            _vals.append(fc_map["vmax_kt"].to_numpy())
        if _vals:
            _all = np.concatenate([np.asarray(v) for v in _vals])
            _cmin = float(np.nanmin(_all))
            _cmax = float(np.nanmax(_all))
        else:
            _cmin, _cmax = 0.0, 100.0

        # Extent and map center/zoom
        pad = 5.0
        lats = obs_win["lat"].to_numpy()
        lons = obs_win["lon"].to_numpy()
        if not fc_map.empty and {"lat", "lon"}.issubset(fc_map.columns):
            lats = np.r_[lats, fc_map["lat"].to_numpy()]
            lons = np.r_[lons, fc_map["lon"].to_numpy()]
        lat_min, lat_max = float(np.nanmin(lats) - pad), float(np.nanmax(lats) + pad)
        lon_min, lon_max = float(np.nanmin(lons) - pad), float(np.nanmax(lons) + pad)
        clat, clon, zoom = _center_zoom(lat_min, lat_max, lon_min, lon_max)

        # Create traces
        traces = []
        if use_mapbox:
            # IBTrACS on mapbox: white markers/lines, no wind coloring, no colorbar, no legend
            traces.append(
                go.Scattermapbox(
                    lon=obs_win["lon"], lat=obs_win["lat"],
                    mode="markers+lines",
                    marker=dict(size=6, color="white", line=dict(color="black", width=0.5)),
                    line=dict(color="white", width=2),
                    name="IBTrACS (obs)",
                    showlegend=False,
                    customdata=_obs_cd,
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "MSLP: %{customdata[0]:.0f} hPa<br>"
                        "Lat: %{customdata[1]:.2f}°, Lon: %{customdata[2]:.2f}°"
                        "<extra></extra>"
                    ),
                )
            )
            if not fc_map.empty and {"lat", "lon"}.issubset(fc_map.columns):
                for m in models:
                    tr = fc_map[fc_map["model"] == m]
                    if tr.empty or tr[["lat", "lon"]].isna().all(axis=None):
                        continue
                    _vt_str = (
                        tr["valid_time"].dt.strftime("%Y-%m-%d %H:%MZ").to_numpy()
                        if "valid_time" in tr.columns else np.array([""] * len(tr))
                    )
                    _lead = tr["lead_h"].to_numpy() if "lead_h" in tr.columns else np.full(len(tr), np.nan)
                    _pmin = tr["mslp_hpa"].to_numpy() if "mslp_hpa" in tr.columns else np.full(len(tr), np.nan)
                    _cd = np.column_stack([_lead, _pmin, _vt_str])

                    traces.append(
                        go.Scattermapbox(
                            lon=tr["lon"], lat=tr["lat"], mode="lines+markers",
                            line=dict(color=_mpl_to_plotly_color(_model_color(m)) or None,
                                      width=2,
                                      dash=("dash" if _is_postproc(m) else "solid")),
                            marker=dict(size=6, color=_mpl_to_plotly_color(_model_color(m)) or None),
                            name=m, meta=m, showlegend=False,
                            customdata=_cd,
                            hovertemplate=(
                                "<b>%{customdata[2]}</b><br>"
                                "Model: %{meta}<br>"
                                "Lead: %{customdata[0]:.0f} h<br>"
                                "MSLP: %{customdata[1]:.0f} hPa"
                                "<extra></extra>"
                            ),
                        )
                    )
        else:
            # Fallback: geo with white obs, model tracks by model color, no wind coloring, no colorbar, no legend
            traces.append(
                go.Scattergeo(
                    lon=obs_win["lon"], lat=obs_win["lat"], mode="markers+lines",
                    marker=dict(size=6, color="white", line=dict(color="black", width=0.5)),
                    line=dict(color="white", width=2), name="IBTrACS (obs)",
                    showlegend=False,
                    customdata=_obs_cd,
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "MSLP: %{customdata[0]:.0f} hPa<br>"
                        "Lat: %{customdata[1]:.2f}°, Lon: %{customdata[2]:.2f}°"
                        "<extra></extra>"
                    ),
                )
            )
            if not fc_map.empty and {"lat", "lon"}.issubset(fc_map.columns):
                for m in models:
                    tr = fc_map[fc_map["model"] == m]
                    if tr.empty or tr[["lat", "lon"]].isna().all(axis=None):
                        continue
                    _vt_str = (
                        tr["valid_time"].dt.strftime("%Y-%m-%d %H:%MZ").to_numpy()
                        if "valid_time" in tr.columns else np.array([""] * len(tr))
                    )
                    _lead = tr["lead_h"].to_numpy() if "lead_h" in tr.columns else np.full(len(tr), np.nan)
                    _pmin = tr["mslp_hpa"].to_numpy() if "mslp_hpa" in tr.columns else np.full(len(tr), np.nan)
                    _cd = np.column_stack([_lead, _pmin, _vt_str])

                    traces.append(
                        go.Scattergeo(
                            lon=tr["lon"], lat=tr["lat"], mode="lines+markers",
                            line=dict(color=_mpl_to_plotly_color(_model_color(m)) or None,
                                      width=2,
                                      dash=("dash" if _is_postproc(m) else "solid")),
                            marker=dict(size=6, color=_mpl_to_plotly_color(_model_color(m)) or None),
                            name=m, meta=m, showlegend=False,
                            customdata=_cd,
                            hovertemplate=(
                                "<b>%{customdata[2]}</b><br>"
                                "Model: %{meta}<br>"
                                "Lead: %{customdata[0]:.0f} h<br>"
                                "MSLP: %{customdata[1]:.0f} hPa"
                                "<extra></extra>"
                            ),
                        )
                    )

        # MSLP on top (x2/y2)
        traces.append(
            go.Scatter(
                x=obs_win["time"], y=obs_win["mslp_hpa"], mode="lines",
                line=dict(color="black", width=3), name="Obs (IBTrACS)",
                xaxis="x2", yaxis="y2", showlegend=True,
            )
        )
        if not fc_ts.empty and "mslp_hpa" in fc_ts:
            for m in models:
                ts = fc_ts[(fc_ts["model"] == m) & fc_ts["mslp_hpa"].notna()]
                if ts.empty:
                    continue
                traces.append(
                    go.Scatter(
                        x=ts["valid_time"], y=ts["mslp_hpa"], mode="lines",
                        line=dict(color=_mpl_to_plotly_color(_model_color(m)) or None,
                                  width=2, dash=("dash" if _is_postproc(m) else "solid")),
                        name=m, xaxis="x2", yaxis="y2", showlegend=False,
                    )
                )
        # Wind on bottom (x3/y3)
        traces.append(
            go.Scatter(
                x=obs_win["time"], y=obs_win["wind_kts"], mode="lines",
                line=dict(color="black", width=3), name="Obs (IBTrACS)",
                xaxis="x3", yaxis="y3", showlegend=False,
            )
        )
        if not fc_ts.empty and "vmax_kt" in fc_ts:
            for m in models:
                ts = fc_ts[(fc_ts["model"] == m) & fc_ts["vmax_kt"].notna()]
                if ts.empty:
                    continue
                traces.append(
                    go.Scatter(
                        x=ts["valid_time"], y=ts["vmax_kt"], mode="lines",
                        line=dict(color=_mpl_to_plotly_color(_model_color(m)) or None,
                                  width=2, dash=("dash" if _is_postproc(m) else "solid")),
                        name=m, xaxis="x3", yaxis="y3", showlegend=True,
                    )
                )

        # Axis ranges and map layout updates
        right_end = end_cut if end_cut is not None else (obs_win["time"].max() if not obs_win.empty else chosen_init)
        layout_updates = dict(
            title=(
                f"<b>{name}</b> <span style='font-size:0.9em;'>({sid})</span> — "
                f"<span style='font-size:0.9em;'>init {pd.to_datetime(chosen_init).strftime('%Y-%m-%d %HZ')}</span>"
            ),
            xaxis2=dict(range=[chosen_init, right_end]),
            xaxis3=dict(range=[chosen_init, right_end]),
        )
        if use_mapbox:
            layout_updates.update(dict(mapbox=dict(center=dict(lat=clat, lon=clon), zoom=zoom, style=map_style)))
        else:
            layout_updates.update(dict(geo=dict(lonaxis_range=[lon_min, lon_max], lataxis_range=[lat_min, lat_max])))
        return traces, layout_updates

    # Build base data for the first init
    first_init = init_list[0]
    base_traces, base_layout = _build_for_init(first_init)
    fig.add_traces(base_traces)

    # Configure domains for right panels explicitly
    fig.update_layout(
        xaxis2=dict(domain=[0.64, 1.0], anchor="y2"),
        yaxis2=dict(domain=[0.55, 1.0], anchor="x2"),
        xaxis3=dict(domain=[0.64, 1.0], anchor="y3"),
        yaxis3=dict(domain=[0.05, 0.50], anchor="x3"),
    )
    fig.update_layout(**base_layout)

    # Build frames for the remaining inits
    for it in init_list[1:]:
        fr_traces, fr_layout = _build_for_init(it)
        frames.append(go.Frame(data=fr_traces, name=str(it), layout=go.Layout(**fr_layout)))
    fig.frames = frames

    # Buttons to switch init
    buttons = []
    for it in init_list:
        label = pd.to_datetime(it).strftime("%Y-%m-%d %HZ")
        buttons.append(
            dict(
                label=label,
                method="animate",
                args=[
                    [str(it)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.65, y=1.08,
                xanchor="left", yanchor="bottom",
                buttons=buttons,
                pad={"r": 6, "t": 6},
                showactive=True,
            )
        ],
        annotations=[
            dict(
                text="<b>Init time:</b>",
                x=0.62, y=1.095,
                xref="paper", yref="paper",
                showarrow=False,
                align="left",
            )
        ],
        margin=dict(l=10, r=10, t=90, b=50),
        template="plotly_white",
    )

    # Export a single self-contained HTML for slides (works offline)
    pio.write_html(fig, html_path, include_plotlyjs=True, full_html=True)
    return html_path


# --------------------------------------------------------------------------------------
# Interactive HTML (Folium ESRI map + Plotly time series; no tokens required)
# --------------------------------------------------------------------------------------

def _hex_from_scalar(v: float, vmin: float, vmax: float, cmap: str = "turbo") -> str:
    try:
        cm = mpl.cm.get_cmap(cmap)
        x = 0.5 if not np.isfinite(v) else (v - vmin) / max(1e-9, (vmax - vmin))
        r, g, b, _ = cm(np.clip(x, 0, 1))
        return mpl.colors.to_hex((r, g, b))
    except Exception:
        return "#3388ff"


def build_interactive_panel_html_esri(
    sid: str,
    name: str,
    obs_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    init_times: Optional[Sequence[str]] = None,
    end_time: Optional[str] = None,
    html_path: str = "panel_interactive_esri.html",
    map_zoom_pad_deg: float = 5.0,
    map_height_px: int = 500,
    cmap_name: str = "turbo",
):
    """Create an interactive HTML with an **ESRI satellite** map (Leaflet/Folium)
    on the left and **Plotly** wind/MSLP time series on the right. No Mapbox token needed.

    - Left: ESRI World Imagery (satellite). Per-init layers; markers colored by wind.
    - Right: Plotly time series with an init selector that synchronizes with the map.

    Notes
    -----
    * This intentionally avoids Mapbox. It requires internet to fetch ESRI tiles at view time.
    * The HTML is self-contained in terms of JS libs (Leaflet/Plotly are referenced via Folium+inline).  
      If your environment blocks CDN access, open with internet enabled.
    """
    # Lazy imports
    try:
        import folium
        from folium import FeatureGroup
        from branca.element import MacroElement, Figure
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as e:
        raise ImportError("This function requires folium and plotly.") from e

    obs = normalize_obs_df(obs_df)
    fc = normalize_fc_df(fc_df)

    obs_sid = obs[obs["sid"] == sid].copy()
    if obs_sid.empty:
        raise ValueError(f"No obs rows for SID {sid}")

    df_sid = fc[fc["sid"] == sid].copy()
    if df_sid.empty:
        raise ValueError(f"No forecast rows for SID {sid}")

    # Determine model list
    avail_models = sorted(df_sid["model"].dropna().unique())
    if models is None:
        models = avail_models
    else:
        models = [m for m in models if m in avail_models]
        if not models:
            raise ValueError("None of the requested models are available for this SID.")

    # Determine init list (up to four, like the Plotly exporter)
    if init_times is None:
        inits_filtered = sorted(pd.to_datetime(df_sid[df_sid["model"].isin(models)]["init_time"].dropna().unique()))
        init_list = inits_filtered[-4:] if len(inits_filtered) > 4 else inits_filtered
    else:
        init_list = [pd.to_datetime(t, utc=True) for t in init_times]
    if not init_list:
        init_list = [pd.to_datetime(df_sid["init_time"].min(), utc=True)]

    end_cut = pd.to_datetime(end_time, utc=True) if end_time is not None else None

    # Helper: aggregate per-lead
    def _agg(sub: pd.DataFrame, step: Optional[int]):
        if sub.empty:
            return sub
        sub = sub.copy()
        sub["lead_h"] = pd.to_numeric(sub["lead_h"], errors="coerce").round()
        keep = [c for c in ["lat", "lon", "vmax_kt", "mslp_hpa"] if c in sub.columns]
        if step is not None:
            sub = sub[sub["lead_h"] % int(step) == 0]
        out = sub.groupby("lead_h", as_index=False)[keep].mean(numeric_only=True)
        out["model"] = sub["model"].iloc[0]
        out["init_time"] = sub["init_time"].iloc[0]
        out["valid_time"] = out["init_time"] + pd.to_timedelta(out["lead_h"], unit="h")
        return out

    # Build datasets per init
    init_payload = {}
    global_lat, global_lon = [], []
    global_winds = []

    for it in init_list:
        # Crop obs
        obs_win = obs_sid[obs_sid["time"] >= it]
        if end_cut is not None:
            obs_win = obs_win[obs_win["time"] <= end_cut]
        if obs_win.empty:
            obs_win = obs_sid.copy()

        # Forecasts at this init
        df_used = df_sid[(df_sid["init_time"] == it) & (df_sid["model"].isin(models))].copy()
        if not df_used.empty and "valid_time" in df_used.columns:
            df_used = df_used[df_used["valid_time"] >= it]
            if end_cut is not None:
                df_used = df_used[df_used["valid_time"] <= end_cut]

        # Aggregate for map and timeseries
        fc_map_list, fc_ts_list = [], []
        for m in models:
            sub = df_used[df_used["model"] == m]
            if sub.empty:
                continue
            m_map = _agg(sub, 6)
            m_map["model"] = m
            fc_map_list.append(m_map)
            m_ts = _agg(sub, None)
            m_ts["model"] = m
            fc_ts_list.append(m_ts)
        fc_map = pd.concat(fc_map_list, ignore_index=True) if fc_map_list else pd.DataFrame()
        fc_ts = pd.concat(fc_ts_list, ignore_index=True) if fc_ts_list else pd.DataFrame()

        init_payload[it] = dict(obs=obs_win, fc_map=fc_map, fc_ts=fc_ts)

        # NaN-safe: only extend with finite coordinates
        _olat = np.asarray(obs_win["lat"], dtype=float)
        _olon = np.asarray(obs_win["lon"], dtype=float)
        global_lat.extend(_olat[np.isfinite(_olat)].tolist())
        global_lon.extend(_olon[np.isfinite(_olon)].tolist())
        if not fc_map.empty and {"lat", "lon"}.issubset(fc_map.columns):
            _flat = np.asarray(fc_map["lat"], dtype=float)
            _flon = np.asarray(fc_map["lon"], dtype=float)
            global_lat.extend(_flat[np.isfinite(_flat)].tolist())
            global_lon.extend(_flon[np.isfinite(_flon)].tolist())
        if "wind_kts" in obs_win:
            global_winds.extend(obs_win["wind_kts"].tolist())
        if not fc_map.empty and "vmax_kt" in fc_map:
            global_winds.extend(fc_map["vmax_kt"].tolist())

    # Map center/zoom from global extent
    lat_min, lat_max = float(np.nanmin(global_lat) - map_zoom_pad_deg), float(np.nanmax(global_lat) + map_zoom_pad_deg)
    lon_min, lon_max = float(np.nanmin(global_lon) - map_zoom_pad_deg), float(np.nanmax(global_lon) + map_zoom_pad_deg)
    if not global_lat or not global_lon:
        # Fallback center over eastern Pacific if no valid coords
        clat, clon = 15.0, -100.0
    else:
        clat, clon = (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0

    # Color scale bounds
    if global_winds:
        cmin, cmax = float(np.nanmin(global_winds)), float(np.nanmax(global_winds))
    else:
        cmin, cmax = 0.0, 100.0

    # Build Folium map with ESRI base
    tiles_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m = folium.Map(location=[clat, clon], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer(tiles=tiles_url, attr="Esri WorldImagery", name="ESRI Satellite", overlay=False, control=False).add_to(m)

    # Per-init overlay groups
    group_names = []
    model_layer_map = {}
    for it in init_list:
        label = pd.to_datetime(it).strftime("%Y-%m-%d %HZ")
        grp = FeatureGroup(name=f"Init {label}", show=(it == init_list[0]))
        model_layer_map[str(it)] = {}
        payload = init_payload[it]
        obs_win, fc_map = payload["obs"], payload["fc_map"]

        # Obs polyline (white) and markers (white, no wind coloring)
        olats, olons = obs_win["lat"].to_list(), obs_win["lon"].to_list()
        obs_coords = [
            (float(la), float(lo))
            for la, lo in zip(olats, olons)
            if np.isfinite(la) and np.isfinite(lo)
        ]
        if len(obs_coords) >= 2:
            folium.PolyLine(obs_coords, color="#ffffff", weight=2, opacity=0.7).add_to(grp)
        for _, r in obs_win.iterrows():
            la, lo = float(r.get("lat", np.nan)), float(r.get("lon", np.nan))
            if not (np.isfinite(la) and np.isfinite(lo)):
                continue
            folium.CircleMarker(
                location=(la, lo),
                radius=4,
                color="#222",
                weight=0.5,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.9,
                tooltip=(
                    f"<b>{pd.to_datetime(r['time']).strftime('%Y-%m-%d %H:%MZ')}</b><br>"
                    f"MSLP: {r.get('mslp_hpa', np.nan):.0f} hPa<br>"
                    f"Wind: {r.get('wind_kts', np.nan):.0f} kt"
                ),
            ).add_to(grp)

        # Models: one sub-layer per model so we can toggle via our custom legend
        if not fc_map.empty and {"lat", "lon"}.issubset(fc_map.columns):
            for mname in models:
                tr = fc_map[fc_map["model"] == mname].copy()
                if tr.empty:
                    continue
                tr = tr.dropna(subset=["lat", "lon"])  # remove NaNs for mapping
                if tr.empty:
                    continue

                mcol = _mpl_to_plotly_color(_model_color(mname)) or "#3388ff"
                dash = "6,4" if _is_postproc(mname) else None

                subgrp = FeatureGroup(name=f"{label} — {mname}")  # one sublayer per model

                coords = [
                    (float(la), float(lo))
                    for la, lo in zip(tr["lat"].to_list(), tr["lon"].to_list())
                    if np.isfinite(la) and np.isfinite(lo)
                ]
                if len(coords) >= 2:
                    folium.PolyLine(
                        coords,
                        color=mcol,
                        weight=2,
                        opacity=0.9,
                        dash_array=dash,
                    ).add_to(subgrp)

                for _, rr in tr.iterrows():
                    la, lo = float(rr.get("lat", np.nan)), float(rr.get("lon", np.nan))
                    if not (np.isfinite(la) and np.isfinite(lo)):
                        continue
                    lead = float(rr.get("lead_h", np.nan))
                    pmin = float(rr.get("mslp_hpa", np.nan))
                    wmax = float(rr.get("vmax_kt", np.nan))
                    folium.CircleMarker(
                        location=(la, lo),
                        radius=3,
                        color="#111",
                        weight=0.4,
                        fill=True,
                        fill_color=mcol,
                        fill_opacity=0.9,
                        tooltip=(
                            f"<b>{pd.to_datetime(rr['valid_time']).strftime('%Y-%m-%d %H:%MZ')}</b><br>"
                            f"Model: {mname}<br>"
                            f"Lead: {lead:.0f} h<br>"
                            f"MSLP: {pmin:.0f} hPa<br>"
                            f"Wind: {wmax:.0f} kt"
                        ),
                    ).add_to(subgrp)

                subgrp.add_to(grp)
                model_layer_map[str(it)][mname] = subgrp.get_name()  # remember JS var

        # Add the per-init group to the map and register its JS name
        grp.add_to(m)
        group_names.append(grp.get_name())

    # --- Build Plotly time series payloads (no frames; replot per init) -----------------
    from plotly.utils import PlotlyJSONEncoder

    def _build_ts_for_init(it: pd.Timestamp):
        payload = init_payload[it]
        obs_win, fc_ts = payload["obs"], payload["fc_ts"]
        data = []

        # MSLP (top) — obs (no legend to avoid duplicates)
        data.append(
            go.Scatter(
                x=obs_win["time"],
                y=obs_win["mslp_hpa"],
                mode="lines",
                line=dict(color="black", width=3),
                name="Obs (IBTrACS)",
                showlegend=False,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%MZ}</b><br>Obs (IBTrACS)<br>MSLP: %{y:.0f} hPa<extra></extra>"
                ),
                xaxis="x", yaxis="y",
                meta="mslp",
            )
        )
        # MSLP models (top) — legend appears through wind traces only
        for mname in models:
            ts = fc_ts[(fc_ts["model"] == mname)]
            if ts.empty:
                continue
            ts = ts.copy()
            if "mslp_hpa" in ts.columns:
                ts["mslp_hpa"] = pd.to_numeric(ts["mslp_hpa"], errors="coerce")
            if ("mslp_hpa" in ts.columns) and ts["mslp_hpa"].notna().any():
                col = _mpl_to_plotly_color(_model_color(mname)) or None
                dash = "dash" if _is_postproc(mname) else "solid"
                data.append(
                    go.Scatter(
                        x=ts["valid_time"], y=ts["mslp_hpa"], mode="lines",
                        line=dict(color=col, width=2, dash=dash), name=mname,
                        legendgroup=mname, showlegend=False,
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d %H:%MZ}</b><br>Model: %{fullData.name}<br>MSLP: %{y:.0f} hPa<extra></extra>"
                        ),
                        xaxis="x", yaxis="y",
                        meta="mslp",
                    )
                )

        # Wind (bottom) — obs (no legend)
        data.append(
            go.Scatter(
                x=obs_win["time"],
                y=obs_win["wind_kts"],
                mode="lines",
                line=dict(color="black", width=3),
                name="Obs (IBTrACS)",
                showlegend=False,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%MZ}</b><br>Obs (IBTrACS)<br>Wind: %{y:.0f} kt<extra></extra>"
                ),
                xaxis="x2", yaxis="y2",
                meta="wind",
            )
        )
        # Wind models (bottom) — ONLY models that actually have wind values for this init
        for mname in models:
            ts = fc_ts[(fc_ts["model"] == mname)]
            if ts.empty:
                continue
            ts = ts.copy()
            if "vmax_kt" in ts.columns:
                ts["vmax_kt"] = pd.to_numeric(ts["vmax_kt"], errors="coerce")
            if ("vmax_kt" not in ts.columns) or ts["vmax_kt"].isna().all():
                continue  # skip models without wind for this init/time window
            col = _mpl_to_plotly_color(_model_color(mname)) or None
            dash = "dash" if _is_postproc(mname) else "solid"
            data.append(
                go.Scatter(
                    x=ts["valid_time"], y=ts["vmax_kt"], mode="lines",
                    line=dict(color=col, width=2, dash=dash), name=mname,
                    legendgroup=mname, showlegend=True,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d %H:%MZ}</b><br>Model: %{fullData.name}<br>Wind: %{y:.0f} kt<extra></extra>"
                    ),
                    xaxis="x2", yaxis="y2", meta="wind",
                )
            )

        # Per-init layout (title + x ranges for both subplots)
        right_end = (
            end_cut if end_cut is not None else (obs_win["time"].max() if not obs_win.empty else it)
        )
        layout_updates = dict(
            title=(
                f"<b>{name}</b> <span style='font-size:0.9em;'>({sid})</span> — "
                f"<span style='font-size:0.9em;'>init {pd.to_datetime(it).strftime('%Y-%m-%d %HZ')}</span>"
            ),
            xaxis=dict(range=[it, right_end], title_text="Valid time (UTC)"),
            xaxis2=dict(range=[it, right_end], title_text="Valid time (UTC)"),
            yaxis=dict(title_text="Sea-level pressure (hPa)", side="right", automargin=True, title_standoff=6),
            yaxis2=dict(title_text="Max wind (kt)", side="right", automargin=True, title_standoff=6),
        )
        return data, layout_updates

    TS_PAYLOADS = {}
    for it in init_list:
        traces, lay = _build_ts_for_init(it)
        TS_PAYLOADS[str(it)] = {
            "data": [t.to_plotly_json() for t in traces],
            "layout": lay,
        }

    TS_BASE_LAYOUT = {
        "height": map_height_px,
        "margin": {"l": 10, "r": 10, "t": 60, "b": 110},  # extra bottom room for legend
        "template": "plotly_white",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,                # place legend below the wind subplot
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.85)",
            "tracegroupgap": 6,
            "groupclick": "togglegroup",
        },
        # Domains emulate a 2x1 subplot grid (top MSLP, bottom wind)
        "xaxis":  {"domain": [0.0, 1.0], "anchor": "y",  "title": {"text": "Valid time (UTC)"}, "automargin": True},
        "yaxis":  {"domain": [0.55, 1.0], "anchor": "x", "side": "right", "automargin": True,
                   "title": {"text": "Sea-level pressure (hPa)"}, "title_standoff": 6},
        "xaxis2": {"domain": [0.0, 1.0], "anchor": "y2", "title": {"text": "Valid time (UTC)"}, "automargin": True},
        "yaxis2": {"domain": [0.05, 0.50], "anchor": "x2", "side": "right", "automargin": True,
                   "title": {"text": "Max wind (kt)"}, "title_standoff": 6},
    }

    # Empty container; JS will draw with Plotly.newPlot
    ts_div = "<div id='ts_div'></div>"

    # JSON blobs for JS
    TS_PAYLOADS_JSON = json.dumps(TS_PAYLOADS, cls=PlotlyJSONEncoder)
    TS_BASE_LAYOUT_JSON = json.dumps(TS_BASE_LAYOUT, cls=PlotlyJSONEncoder)

    # Render Folium map to HTML components
    map_html = m.get_root().render()
    # HTML legend overlay (checkboxes) — one entry per model
    groups_order = ["Physical", "AI", "Postprocess"]
    group_entries: dict[str, list[str]] = {g: [] for g in groups_order}

    for mname in models:
        grp = _model_group(mname)
        if grp not in group_entries:
            group_entries[grp] = []
        col = _mpl_to_plotly_color(_model_color(mname)) or "#3388ff"
        swatch = f"<span style='width:12px;height:12px;background:{col};border:1px solid #111;display:inline-block;'></span>"
        entry = (
            f"<label style='display:flex;align-items:center;gap:6px;margin:2px 0;'>"
            f"<input type='checkbox' class='modchk' data-model='{mname}' data-group='{grp}' checked>"
            f"{swatch}<span style='white-space:nowrap'>{mname}</span></label>"
        )
        group_entries[grp].append(entry)

    legend_sections: list[str] = []
    for grp in groups_order:
        entries = group_entries.get(grp, [])
        if not entries:
            continue
        legend_sections.append(
            "<div class='legend-group'>"
            f"<label class='grphead'><input type='checkbox' class='grpchk' data-group='{grp}' checked>"
            f"<span style='font-weight:600'>{grp}</span></label>"
            + "<div class='legend-items'>" + "".join(entries) + "</div>"
            + "</div>"
        )

    legend_html = ("<div id='mapLegend' class='legend'>"
                   "<div style='font-weight:600;margin-bottom:4px;'>Models</div>"
                   + "".join(legend_sections) + "</div>")
    # Compose a two-column responsive layout with a synced init selector
    it0 = init_list[0]
    select_options = "".join(
        f"<option value='{str(it)}' {'selected' if (it==it0) else ''}>{pd.to_datetime(it).strftime('%Y-%m-%d %HZ')}</option>"
        for it in init_list
    )

    # Build a mapping { str(init_timestamp): folium_js_var_name }
    layer_map = {str(it): group_names[i] for i, it in enumerate(init_list)}
    map_var = m.get_name()  # Folium global map variable name

    # JS: robust layer toggling and Plotly sync
    js_sync = f"""
    <script>
    var INIT_LAYER_MAP = {json.dumps({str(k): v for k, v in layer_map.items()})};
    var MODEL_LAYER_MAP = {json.dumps({str(k): v for k, v in model_layer_map.items()})};
    var MAP_VAR_NAME = {json.dumps(map_var)};
    var MODELS = {json.dumps(list(models))};
    var CURRENT_INIT = {json.dumps(str(it0))};

    // Time-series datasets per init and a shared base layout
    var TS_DATA = {TS_PAYLOADS_JSON};
    var TS_BASE = {TS_BASE_LAYOUT_JSON};

    function setInitLayer(tsStr) {{
      var mapObj = window[MAP_VAR_NAME];
      if (!mapObj) return;
      Object.values(INIT_LAYER_MAP).forEach(function(varName) {{
        var layer = window[varName];
        try {{ if (mapObj.hasLayer(layer)) mapObj.removeLayer(layer); }} catch(e) {{}}
      }});
      var wanted = INIT_LAYER_MAP[tsStr];
      if (wanted && window[wanted]) {{
        try {{ mapObj.addLayer(window[wanted]); }} catch(e) {{}}
      }}
    }}

    function applyModelVis() {{
      // Map: toggle sublayers for CURRENT_INIT according to checkboxes
      var mapObj = window[MAP_VAR_NAME];
      if (mapObj) {{
        var checks = document.querySelectorAll('.modchk');
        var visByModel = {{}};
        checks.forEach(function(cb) {{ visByModel[cb.dataset.model] = cb.checked; }});
        var mp = MODEL_LAYER_MAP[CURRENT_INIT] || {{}};
        Object.keys(mp).forEach(function(mname) {{
          var subVar = mp[mname];
          var layer = window[subVar];
          var on = !!visByModel[mname];
          if (!layer) return;
          try {{ if (on && !mapObj.hasLayer(layer)) mapObj.addLayer(layer); }} catch(e) {{}}
          try {{ if (!on && mapObj.hasLayer(layer)) mapObj.removeLayer(layer); }} catch(e) {{}}
        }});
      }}

      // Plotly: toggle model traces by name (both panels). Only wind traces carry legend entries.
      if (window.Plotly) {{
        var gd = document.getElementById('ts_div');
        if (!gd || !gd.data) return;
        var checks = document.querySelectorAll('.modchk');
        var visByModel = {{}}; checks.forEach(function(cb) {{ visByModel[cb.dataset.model] = cb.checked; }});
        var idxs = [], vis = [];
        for (var i=0;i<gd.data.length;i++) {{
          var nm = gd.data[i].name || '';
          if (MODELS.indexOf(nm) >= 0) {{
            idxs.push(i);
            vis.push(!!visByModel[nm]);
          }}
        }}
        if (idxs.length) {{ Plotly.restyle('ts_div', {{visible: vis}}, idxs); }}
      }}
    }}

    function plotInit(val) {{
      var ds = TS_DATA[val];
      if (!ds) return;
      var lay = JSON.parse(JSON.stringify(TS_BASE));
      if (ds.layout) {{
        if (ds.layout.title) lay.title = ds.layout.title;
        ['xaxis','xaxis2','yaxis','yaxis2'].forEach(function(ax) {{
          if (ds.layout[ax]) {{
            lay[ax] = Object.assign({{}}, lay[ax] || {{}}, ds.layout[ax]);
          }}
        }});
        if (ds.layout.legend) lay.legend = Object.assign({{}}, lay.legend || {{}}, ds.layout.legend);
        if (ds.layout.margin) lay.margin = Object.assign({{}}, lay.margin || {{}}, ds.layout.margin);
      }}
      // Draw fresh figure (ensures legend reflects only models with wind at this init)
      Plotly.newPlot('ts_div', ds.data, lay, {{displayModeBar: true}}).then(function() {{
        applyModelVis();
      }});
    }}

    function onInitChange(val) {{
      CURRENT_INIT = val;
      setInitLayer(val);
      if (window.Plotly) {{
        plotInit(val); // fully refresh the time-series for the selected init
      }} else {{
        applyModelVis();
      }}
    }}

    document.addEventListener('DOMContentLoaded', function() {{
      var sel = document.getElementById('initSelect');
      if (window.Plotly) {{ plotInit(CURRENT_INIT); }}
      if (sel) {{ sel.addEventListener('change', function() {{ onInitChange(this.value);}}); }}

      // Per-model toggles
      document.querySelectorAll('.modchk').forEach(function(cb) {{
          cb.addEventListener('change', applyModelVis);
      }});

      // Group master checkboxes toggle all models in their group
      document.querySelectorAll('.grpchk').forEach(function(cb) {{
          cb.addEventListener('change', function() {{
          var grp = this.dataset.group;
          var on = this.checked;
          document.querySelectorAll('.modchk[data-group="' + grp + '"]').forEach(function(mcb){{
              mcb.checked = on;
          }});
          applyModelVis();
      }});
      }});
    }});
    </script>
    """

    # Final HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset='utf-8'/>
    <title>{name} ({sid}) — Interactive ESRI Map + Time Series</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
    body {{ margin:0; font-family:Arial, sans-serif; }}
    .bar {{ display:flex; align-items:center; gap:10px; padding:8px 12px; border-bottom:1px solid #ddd; }}
    .grid {{ display:grid; grid-template-columns: 60% 40%; grid-template-rows: auto; height: {map_height_px}px; }}
    #mapcol {{ position:relative; }}
    #plotcol {{ padding:6px; }}
    .title {{ font-size:16px; font-weight:600; }}
    .label {{ font-weight:600; }}
    .legend {{ position:absolute; right:10px; top:10px;
            background:rgba(255,255,255,0.92); padding:8px 10px;
            border:1px solid #ccc; border-radius:4px;
            max-height:85%; overflow:auto; font-size:12px; z-index:2000; pointer-events:auto; }}
    .legend-group {{ margin-top:6px; border-top:1px solid #eee; padding-top:4px; }}
    .legend-group:first-of-type {{ border-top:none; padding-top:0; }}
    .grphead {{ display:flex; align-items:center; gap:6px; margin:2px 0; }}
    .legend-items {{ margin-left:16px; }}
    .leaflet-control-layers {{ display: none !important; }}
    </style>
    </head>
    <body>
    <div class='bar'>
        <span class='title'>{name} (<span style='opacity:0.7'>{sid}</span>)</span>
        <span class='label'>Init time:</span>
        <select id='initSelect'>{select_options}</select>
    </div>
    <div class='grid'>
        <div id='mapcol'>
        {map_html}
        {legend_html}
        </div>
        <div id='plotcol'>
        {ts_div}
        </div>
    </div>
    {js_sync}
    </body>
    </html>
    """

    # Write HTML file and return path
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def plot_case_panel(
    sid: str,
    name: str,
    obs_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    init_time: Optional[str] = None,
    lead_step: int = 6,
    wind_cmap: str = "turbo",
    out_png: Optional[str] = None,
    show: bool = True,
    tiles: str | None = None,
    tiles_zoom: int | None = None,
    map_pad_deg: float = 1.0,
    end_time: Optional[str] = None,
):
    """Create a clean case-study panel (map + intensity + pressure).

    Parameters
    ----------
    sid, name : str
        Storm ID and display name (e.g., ("2023294N09264", "Otis")).
    obs_df : DataFrame
        IBTrACS-like observations; flexible schema handled by `normalize_obs_df`.
    fc_df : DataFrame
        Forecast results across models; flexible schema handled by `normalize_fc_df`.
    models : list[str] | None
        Explicit subset of models; None = all available for chosen init.
    init_time : str | None
        ISO string of desired init. None = choose latest init with max model overlap.
    lead_step : int
        Subsample model track points by lead when drawing the map (default 6 h).
    wind_cmap : str
        Matplotlib colormap name for obs track coloring by wind.
    out_png : str | None
        If provided, save the panel to this path.
    show : bool
        Whether to show the figure interactively.
    tiles : {"esri","osm","stamen"} | None
        Optional web tiles background (satellite/imagery). Requires internet. Default None.
    tiles_zoom : int | None
        Zoom level for tiles; if None a heuristic is used.
    map_pad_deg : float
        Degrees of padding added around the min/max track coordinates when computing
        the map extent. Smaller values zoom in more tightly (default 3°; previously 5°).
    end_time : str | None
        Optional UTC cutoff for plots (map + time series). Example: "2023-10-27 12:00".
    """
    # Normalize inputs
    obs = normalize_obs_df(obs_df)
    df_obs_sid = obs[obs["sid"] == sid]
    if df_obs_sid.empty:
        raise ValueError(f"No obs rows for SID {sid}")

    fc_norm = normalize_fc_df(fc_df)
    df_fc_used, chosen_init, models_used = _subset_fc_sid(
        fc_norm, sid, models, init_time
    )

    # Optional cutoff time (UTC)
    end_cutoff = pd.to_datetime(end_time, utc=True) if end_time is not None else None

    # Crop obs for the map: start at init time; end at cutoff if provided
    df_obs_sid_map = df_obs_sid[df_obs_sid["time"] >= chosen_init].copy()
    if end_cutoff is not None:
        df_obs_sid_map = df_obs_sid_map[df_obs_sid_map["time"] <= end_cutoff]
    if df_obs_sid_map.empty:
        df_obs_sid_map = df_obs_sid.copy()

    # Time-cropped forecasts for map extent / track drawing
    df_fc_used_map = df_fc_used.copy()
    if "valid_time" in df_fc_used_map.columns:
        df_fc_used_map = df_fc_used_map[df_fc_used_map["valid_time"] >= chosen_init]
        if end_cutoff is not None:
            df_fc_used_map = df_fc_used_map[df_fc_used_map["valid_time"] <= end_cutoff]

    # Figure and axes
    # Map extent from obs + any available forecast tracks
    obs_lats = df_obs_sid_map["lat"].to_numpy()
    obs_lons = df_obs_sid_map["lon"].to_numpy()
    if "lat" in df_fc_used_map.columns and "lon" in df_fc_used_map.columns:
        fc_lats = df_fc_used_map["lat"].to_numpy()
        fc_lons = df_fc_used_map["lon"].to_numpy()
        fc_lats = fc_lats[~np.isnan(fc_lats)]
        fc_lons = fc_lons[~np.isnan(fc_lons)]
    else:
        fc_lats = np.array([])
        fc_lons = np.array([])
    lats = np.concatenate([obs_lats[~np.isnan(obs_lats)], fc_lats])
    lons = np.concatenate([obs_lons[~np.isnan(obs_lons)], fc_lons])
    extent = _nice_extent(lats, lons, pad_deg=float(map_pad_deg))

    fig, ax_map, ax_vmax, ax_mslp, ax_cbar, proj = _make_basemap(extent)

    # Optional satellite/tiles background (requires internet). Silent if it fails.
    if tiles is not None:
        _add_tiles_background(ax_map, extent, provider=tiles, zoom=tiles_zoom)

    # Draw map layers and legend
    model_handles = _plot_model_tracks(
        ax_map, df_fc_used_map, models_used, ccrs.PlateCarree(), lead_step=lead_step
    )
    sc = _plot_obs_track(ax_map, df_obs_sid_map, ccrs.PlateCarree(), cmap=wind_cmap)
    # Build combined legend: IBTrACS + models
    ib_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        color="k",
        markersize=6,
        label="IBTrACS (obs)",
    )
    handles = [ib_handle] + (model_handles or [])
    if handles:
        ax_map.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)

    # Timeseries right column
    _plot_timeseries(
        ax_vmax,
        df_obs_sid,
        df_fc_used,
        models_used,
        var_obs="wind_kts",
        var_fc="vmax_kt",
        ylabel="Max wind (kt)",
        start_time=chosen_init,
        end_time=end_cutoff,
    )
    _plot_timeseries(
        ax_mslp,
        df_obs_sid,
        df_fc_used,
        models_used,
        var_obs="mslp_hpa",
        var_fc="mslp_hpa",
        ylabel="Sea-level pressure (hPa)",
        start_time=chosen_init,
        end_time=end_cutoff,
    )

    # Shared x formatting for right column (ConciseDateFormatter)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in (ax_vmax, ax_mslp):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        #ax.set_xlabel("Valid time (UTC)")
    
    # Put the time-series y-axes on the right so they don't clash with the map
    for ax in (ax_vmax, ax_mslp):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        # optional clean-up: hide the left spine
        if "left" in ax.spines:
            ax.spines["left"].set_visible(False)

    if end_cutoff is not None:
        right_bound = end_cutoff
    else:
        end_obs = df_obs_sid[df_obs_sid["time"] >= chosen_init]["time"].max()
        end_fc = df_fc_used.get("valid_time")
        end_fc = end_fc.max() if end_fc is not None else None
        right_bound = (
            max([t for t in [end_obs, end_fc] if pd.notna(t)])
            if (pd.notna(end_obs) or end_fc is not None)
            else (chosen_init + pd.Timedelta(hours=120))
        )
    for ax in (ax_vmax, ax_mslp):
        ax.set_xlim(left=chosen_init, right=right_bound)

    # Print a short availability summary to aid debugging
    try:
        print(_diagnose_vars(df_fc_used, models_used))
    except Exception:
        pass

    # Colorbar for wind
    cb = plt.colorbar(sc, cax=ax_cbar, orientation="horizontal")
    cb.set_label("Obs 1‑min sustained wind (kt)")

    # Title
    init_str = pd.to_datetime(chosen_init).strftime("%Y-%m-%d %HZ")
    fig.suptitle(f"{name} ({sid}) — init {init_str}", fontsize=16, y=0.98)

    if out_png:
        fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Helper: Render panels for a range of inits around a center
# --------------------------------------------------------------------------------------
def plot_case_panel_range(
    sid: str,
    name: str,
    obs_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    center_init: Optional[str] = None,
    n_before: int = 1,
    n_after: int = 1,
    lead_step: int = 6,
    wind_cmap: str = "turbo",
    tiles: str | None = None,
    tiles_zoom: int | None = None,
    out_dir: str | Path | None = None,
    basename: Optional[str] = None,
    show: bool = False,
    end_time: Optional[str] = None,
) -> list[pd.Timestamp]:
    """Render multiple panels for consecutive init times around a center init.

    Parameters
    ----------
    center_init : str | None
        If None, the function uses the same strategy as `plot_case_panel` to pick the
        center (latest init with maximum model overlap). Otherwise, use the provided UTC
        init string.
    n_before, n_after : int
        Number of *available* init times to include before/after the center.
    out_dir : path | None
        Directory to save PNGs. Defaults to current directory.
    basename : str | None
        Basename for files (defaults to f"panel_{name}_{sid}"). The init time stamp
        is appended automatically.
    show : bool
        Whether to display each panel (default False; saves faster in headless runs).
    end_time : str | None
        Optional UTC cutoff applied to every panel.

    Returns
    -------
    List of init timestamps that were plotted (UTC).
    """
    # Normalize once, then select SID
    fc_norm = normalize_fc_df(fc_df)
    df_sid = fc_norm[fc_norm["sid"] == sid].copy()
    if df_sid.empty:
        raise ValueError(f"No forecast rows for SID {sid}")

    # Determine center init
    if center_init is None:
        center_ts = _choose_init(df_sid, models, None)
    else:
        center_ts = pd.to_datetime(center_init, utc=True)

    # Collect unique inits available for the models of interest
    if models is None:
        models = sorted(df_sid["model"].dropna().unique())
    df_sid = df_sid[df_sid["model"].isin(models)]
    inits_unique = sorted(pd.to_datetime(df_sid["init_time"].dropna().unique()))
    if not inits_unique:
        raise ValueError("No init_time values available after filtering.")

    # Find index of center; if not present exactly, pick nearest earlier one
    # (common when center_ts was chosen by overlap but a model list was restricted).
    try:
        idx_center = inits_unique.index(center_ts)
    except ValueError:
        # pick the latest init before center_ts, or the earliest if none before
        earlier = [t for t in inits_unique if t <= center_ts]
        idx_center = len(earlier) - 1 if earlier else 0

    i0 = max(0, idx_center - n_before)
    i1 = min(len(inits_unique) - 1, idx_center + n_after)
    selected = inits_unique[i0 : i1 + 1]

    out_dir = Path(out_dir) if out_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or f"panel_{name}_{sid}"

    rendered: list[pd.Timestamp] = []
    for it in selected:
        stamp = pd.to_datetime(it).strftime("%Y%m%d%HZ")
        out_png = out_dir / f"{base}_init{stamp}.png"
        plot_case_panel(
            sid=sid,
            name=name,
            obs_df=obs_df,
            fc_df=fc_df,
            models=models,
            init_time=str(it),
            lead_step=lead_step,
            wind_cmap=wind_cmap,
            out_png=str(out_png),
            show=show,
            tiles=tiles,
            tiles_zoom=tiles_zoom,
            end_time=end_time,
        )
        rendered.append(it)
    return rendered


# --------------------------------------------------------------------------------------
# Example usage (run this file directly)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # You are expected to have an `ibtracs` DataFrame available in scope when running
    # from a notebook/script that imports this module. For a quick CLI test, import
    # this file and pass obs_df and fc_df explicitly.
    #
    # Example:
    #   from case_studies_paper import plot_case_panel, load_tcbench_results
    #   fc = load_tcbench_results("/path/to/TCBench Results", year=2023)
    #   plot_case_panel("2023294N09264", "Otis", ibtracs, fc, models=None, init_time=None,
    #                   lead_step=6, out_png="panel_Otis_2023294N09264.png", show=True)
    #   plot_case_panel_range("2023294N09264", "Otis", ibtracs, fc,
    #                         models=None, center_init=None, n_before=1, n_after=2,
    #                         tiles=None, out_dir="./panels", show=False)
    pass

# %%


# --------------------------------------------------------------------------------------
# Convenience loader: Only Pangu postprocessing models
# --------------------------------------------------------------------------------------
def load_pangu_postprocessing(
    results_dir: str | Path, year: int = 2023, verbose: bool = True
) -> pd.DataFrame:
    """Load only the Pangu postprocessing triplet (ANN/UNET/MLR) with nice names.

    This function relies on the directory scanner and filters by model names, so it
    will be fast compared to loading everything.
    """
    return load_tcbench_results(
        results_dir,
        year=year,
        only_models=["pangu_ann", "pangu_unet", "pangu_mlr"],
        verbose=verbose,
    )
