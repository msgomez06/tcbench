# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import argparse

from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib
import metrics


def evaluate_tracks_RI(
    ibtracs_folder: str,
    results_folder: str,
    year: int = 2023,
    RI_thresh: float = 34.0,
    RI_window: int = 24,
    keep_intensification: bool = False,
    select_files: list[str] | None = None,
    recompute: bool = False,
    verbose: bool = True,
):
    """Compute RI labels for track CSVs and save `<name>_RI.csv` (works for any model type)."""
    # Read IBTrACS and filter by year
    ibtracs = toolbox.read_hist_track_file(tracks_path=ibtracs_folder)
    ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == year]

    # Precompute IBTrACS intensification and RI flag at 6h grid
    ibtracs = ibtracs[np.isin(ibtracs["ISO_TIME"].dt.hour, [0, 6, 12, 18])].copy()
    ibtracs["USA_WIND"] = (
        ibtracs["USA_WIND"].astype(str).str.strip().replace("", np.nan).astype(float)
    )
    ibtracs["RI"] = pd.Series(
        pd.array([pd.NA] * len(ibtracs), dtype="boolean"), index=ibtracs.index
    )
    ibtracs["intensification"] = pd.Series(
        np.full(len(ibtracs), np.nan), index=ibtracs.index, dtype="float"
    )

    for SID in ibtracs["SID"].unique():
        mask_sid = ibtracs["SID"] == SID
        df_sid = ibtracs.loc[mask_sid].copy()
        ref_sid = df_sid.copy()
        orig_idx = ref_sid.index
        df_sid["ref_time"] = pd.to_datetime(df_sid["ISO_TIME"]) - pd.Timedelta(
            hours=RI_window
        )
        df_sid = df_sid.merge(
            ref_sid.rename(
                columns={"ISO_TIME": "ref_time", "USA_WIND": "ref_intensity"}
            )[["ref_time", "ref_intensity"]],
            on="ref_time",
            how="left",
        )
        df_sid["intensification"] = df_sid["USA_WIND"] - df_sid["ref_intensity"]
        df_sid["RI"] = df_sid["intensification"] >= RI_thresh
        df_sid = df_sid.set_index(orig_idx)
        ibtracs.loc[df_sid.index, "RI"] = df_sid["RI"].astype("boolean")
        ibtracs.loc[df_sid.index, "intensification"] = pd.to_numeric(
            df_sid["intensification"], errors="coerce"
        )

    # Discover target files
    if select_files is None:
        all_files = [f for f in os.listdir(results_folder) if f.endswith(".csv")]
        # consider any track-like csv that is not already an RI or results file
        track_files = [
            f
            for f in all_files
            if ("results" not in f.lower())
            and (not f.lower().endswith("_ri.csv"))
            and ("ibtracs" not in f.lower())
        ]
    else:
        track_files = [f for f in select_files if f.endswith(".csv")]

    out_map: dict[str, str] = {}

    for track_file in track_files:
        base = os.path.splitext(track_file)[0]
        out_csv = os.path.join(results_folder, f"{base}_RI.csv")
        if (not recompute) and os.path.exists(out_csv):
            if verbose:
                print(f"Skipping {track_file} (exists): {os.path.basename(out_csv)}")
            out_map[base] = out_csv
            continue

        track_df = pd.read_csv(os.path.join(results_folder, track_file))

        # Detect ensemble column
        probabilistic = False
        ensemble_col = None
        for col in track_df.columns:
            if "ensemble" in str(col).lower():
                ensemble_col = col
                probabilistic = True
                break

        # Ensure datetime and indexability
        track_df["Initial Time"] = pd.to_datetime(
            track_df["Initial Time"], errors="coerce"
        )
        track_df["Valid Time"] = pd.to_datetime(track_df["Valid Time"], errors="coerce")

        # Initialize RI columns if missing
        if "RI" not in track_df.columns:
            track_df["RI"] = pd.Series(
                pd.array([pd.NA] * len(track_df), dtype="boolean"), index=track_df.index
            )
        else:
            try:
                track_df["RI"] = track_df["RI"].astype("boolean")
            except Exception:
                track_df["RI"] = pd.Series(
                    pd.array([pd.NA] * len(track_df), dtype="boolean"),
                    index=track_df.index,
                )
        if keep_intensification and "intensification" not in track_df.columns:
            track_df["intensification"] = np.nan

        for SID in track_df["SID"].dropna().unique():
            mask_sid = track_df["SID"] == SID
            df_sid = track_df.loc[mask_sid].copy()
            ib_ref = ibtracs[ibtracs["SID"] == SID].copy()

            ref_sid = df_sid.copy()
            orig_idx = df_sid.index
            df_sid["ref_time"] = df_sid["Valid Time"] - pd.Timedelta(hours=RI_window)

            # Build a temporary key for (init, tau) match
            ref_sid["temp_index"] = (
                ref_sid["Initial Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                + " tau "
                + (
                    (
                        (
                            ref_sid["Valid Time"] - ref_sid["Initial Time"]
                        ).dt.total_seconds()
                        // 3600
                    )
                    .astype(int)
                    .astype(str)
                )
            )
            df_sid["temp_index"] = (
                df_sid["Initial Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                + " tau "
                + (
                    (
                        (df_sid["ref_time"] - df_sid["Initial Time"]).dt.total_seconds()
                        // 3600
                    )
                    .astype(int)
                    .astype(str)
                )
            )

            if probabilistic and (ensemble_col in df_sid.columns):
                ref_sid["temp_index"] = (
                    ref_sid["temp_index"]
                    + " ens_idx "
                    + ref_sid[ensemble_col].astype(str)
                )
                df_sid["temp_index"] = (
                    df_sid["temp_index"]
                    + " ens_idx "
                    + df_sid[ensemble_col].astype(str)
                )

            ref_sid["orig_idx"] = ref_sid.index

            # Local self-merge to get wind at ref_time from same file when available
            df_sid = df_sid.merge(
                ref_sid[["temp_index", "wind max"]].rename(
                    columns={"wind max": "wind max ref"}
                ),
                on="temp_index",
                how="left",
            )

            # For those rows where ref_time == Initial Time, fall back to IBTrACS wind
            fallback_mask = df_sid["Initial Time"] == df_sid["ref_time"]
            if fallback_mask.any():
                ib_finder = df_sid.loc[fallback_mask, ["Valid Time"]].merge(
                    ib_ref.rename(columns={"ISO_TIME": "Valid Time"})[
                        ["Valid Time", "USA_WIND"]
                    ],
                    on="Valid Time",
                    how="left",
                )
                ref_values = (
                    ib_finder["USA_WIND"]
                    .astype(str)
                    .str.strip()
                    .replace("", np.nan)
                    .astype(float)
                )
                df_sid.loc[fallback_mask, "wind max ref"] = ref_values.values

            # Compute intensification and flag
            df_sid["intensification"] = df_sid["wind max"] - df_sid["wind max ref"]
            df_sid["RI"] = df_sid["intensification"] >= RI_thresh

            # Map back to original indices
            df_sid["temp_index"] = (
                df_sid["Initial Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                + " tau "
                + (
                    (
                        (
                            df_sid["Valid Time"] - df_sid["Initial Time"]
                        ).dt.total_seconds()
                        // 3600
                    )
                    .astype(int)
                    .astype(str)
                )
            )
            if probabilistic and (ensemble_col in df_sid.columns):
                df_sid["temp_index"] = (
                    df_sid["temp_index"]
                    + " ens_idx "
                    + df_sid[ensemble_col].astype(str)
                )

            ref_sid = ref_sid[["temp_index", "orig_idx"]]
            df_sid = df_sid.merge(ref_sid, on="temp_index", how="left")
            df_sid.dropna(subset=["orig_idx"], inplace=True)
            df_sid = df_sid.set_index("orig_idx")
            df_sid.index.name = None

            track_df.loc[df_sid.index, "RI"] = df_sid["RI"].astype("boolean")
            if keep_intensification:
                track_df.loc[df_sid.index, "intensification"] = df_sid[
                    "intensification"
                ]

        # Save
        track_df.to_csv(out_csv, index=False)
        out_map[base] = out_csv
        if verbose:
            print(f"Saved RI file: {out_csv}")

    return out_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RI labels for track CSVs and save <name>_RI.csv"
    )
    parser.add_argument(
        "--ibtracs_folder",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/default/milton/TCBench Results",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--RI_thresh", type=float, default=34.0)
    parser.add_argument("--RI_window", type=int, default=24)
    parser.add_argument("--keep_intensification", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    evaluate_tracks_RI(
        ibtracs_folder=args.ibtracs_folder,
        results_folder=args.results_folder,
        year=args.year,
        RI_thresh=args.RI_thresh,
        RI_window=args.RI_window,
        keep_intensification=args.keep_intensification,
        select_files=None,
        recompute=args.recompute,
        verbose=True,
    )
