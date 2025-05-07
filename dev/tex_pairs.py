# %% Imports
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import toolbox

# %%
results_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/"
)

# %% Training Curves


# Panguweather MLR
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


PM_mlr_files = {
    "probabilistic": load_pickle(
        results_path
        + "TorchMLR_losses_03-26-10h27_panguweather_probabilistic_masked.pkl"
    ),
    "deterministic": load_pickle(
        results_path + "MLR_TorchMLR_losses_12-12-15h15_panguweather_deterministic.pkl"
    ),
}

PU_mlr_files = {
    "probabilistic": load_pickle(
        results_path
        + "TorchMLR_losses_04-02-15h06_panguweather_probabilistic_unmasked.pkl"
    ),
    "deterministic": load_pickle(
        results_path
        + "TorchMLR_losses_04-02-14h25_panguweather_deterministic_unmasked.pkl"
    ),
}

PU_ann_files = {
    "deterministic_unmasked": load_pickle(
        results_path
        + "SimpleANN_losses_04-02-14h22_panguweather_deterministic_unmasked.pkl"
    ),
    "probabilistic_unmasked": load_pickle(
        results_path
        + "SimpleANN_losses_01-31-11h47_panguweather_probabilistic_unmasked.pkl"
    ),
}

PM_ann_files = {
    "probabilistic_masked": load_pickle(
        results_path
        + "SimpleANN_losses_02-03-08h29_panguweather_probabilistic_masked.pkl"
    ),
    "deterministic_masked": load_pickle(
        results_path
        + "SimpleANN_losses_04-02-14h16_panguweather_deterministic_masked.pkl"
    ),
}

PM_cnn_files = {
    "deterministic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_04-02-10h47_panguweather_deterministic_[32,64,128]_masked.pkl"
    ),
    "probabilistic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_04-02-13h23_panguweather_probabilistic_[32,64,128]_masked.pkl"
    ),
}

PU_cnn_files = {
    "deterministic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_04-02-11h56_panguweather_deterministic_[32,64,128]_unmasked.pkl"
    ),
    "probabilistic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_04-02-14h23_panguweather_probabilistic_[32,64,128]_unmasked.pkl"
    ),
}

pangu_unet_files = {
    "probabilistic": load_pickle(
        results_path
        + "CNN_UNet_losses_01-14-16h09_panguweather_probabilistic_[32,64,128].pkl"
    )
}

fm_mlr_files = {
    "probabilistic_masked": load_pickle(
        results_path
        + "TorchMLR_losses_01-31-12h20_fourcastnetv2_probabilistic_masked.pkl"
    ),
    "deterministic_masked": load_pickle(
        results_path
        + "TorchMLR_losses_04-02-14h36_fourcastnetv2_deterministic_masked.pkl"
    ),
}

fu_mlr_files = {
    "probabilistic_unmasked": load_pickle(
        results_path
        + "TorchMLR_losses_01-31-12h05_fourcastnetv2_probabilistic_unmasked.pkl"
    ),
    "deterministic_unmasked": load_pickle(
        results_path
        + "TorchMLR_losses_04-02-14h54_fourcastnetv2_deterministic_unmasked.pkl"
    ),
}

fm_ann_files = {
    "probabilistic_masked": load_pickle(
        results_path
        + "SimpleANN_losses_02-03-08h36_fourcastnetv2_probabilistic_masked.pkl"
    ),
    "deterministic_masked": load_pickle(
        results_path
        + "SimpleANN_losses_04-02-14h43_fourcastnetv2_deterministic_masked.pkl"
    ),
}
fu_ann_files = {
    "deterministic_unmasked": load_pickle(
        results_path
        + "SimpleANN_losses_04-02-14h46_fourcastnetv2_deterministic_unmasked.pkl"
    ),
    "probabilistic_unmasked": load_pickle(
        results_path
        + "SimpleANN_losses_01-31-11h42_fourcastnetv2_probabilistic_unmasked.pkl"
    ),
}

era5_mlr_masked = {
    "deterministic_mlr": load_pickle(
        results_path + "TorchMLR_losses_04-10-14h11_ERA5_deterministic_masked.pkl"
    ),
    "probabilistic_mlr": load_pickle(
        results_path + "TorchMLR_losses_04-10-14h58_ERA5_probabilistic_masked.pkl"
    ),
}

era5_mlr_unmasked = {
    "deterministic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-13h59_ERA5_deterministic_unmasked.pkl"
    ),
    "probabilistic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-14h46_ERA5_probabilistic_unmasked.pkl"
    ),
}

era5_ann_masked = {
    "deterministic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-14h19_ERA5_deterministic_masked.pkl"
    ),
    "probabilistic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-14h24_ERA5_probabilistic_masked.pkl"
    ),
}

era5_ann_unmasked = {
    "deterministic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-13h52_ERA5_deterministic_unmasked.pkl"
    ),
    "probabilistic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-14h46_ERA5_probabilistic_unmasked.pkl"
    ),
}


# %%
for filedict, name in zip(
    [
        PM_mlr_files,
        PU_mlr_files,
        PU_ann_files,
        PM_ann_files,
        PU_cnn_files,
        PM_cnn_files,
        pangu_unet_files,
        fm_mlr_files,
        fu_mlr_files,
        fm_ann_files,
        fu_ann_files,
        era5_mlr_masked,
        era5_mlr_unmasked,
        era5_ann_masked,
        era5_ann_unmasked,
    ],
    [
        "PanMask (MLR)",
        "PanUnmask (MLR)",
        "PanUnmask (ANN)",
        "PanMask (ANN)",
        "Pangu_unmask CNN",
        "Pangu_mask CNN",
        "Pangu UNet",
        "fourMask (MLR)",
        "fourUnmask (MLR)",
        "fourMask (ANN)",
        "fourUnmask (ANN)",
        "ERA5 Masked (MLR)",
        "ERA5 Unmasked (MLR)",
        "ERA5 Masked (ANN)",
        "ERA5 Unmasked (ANN)",
    ],
):
    for key, scores in filedict.items():
        print(f"{name} {key}")
        if False:
            for scoretype, score in scores.items():
                print(f"{name} {key} {scoretype}")

                out_pairs = ""
                for idx, score in enumerate(scores[scoretype]):
                    out_pairs += f"({idx},{score}) "
                print(out_pairs)

        keys = list(scores.keys())

        # find the key corresponding to validation loss
        val_loss_key = [key for key in keys if "val" in key][0]
        train_loss_key = [key for key in keys if "train" in key][0]

        min_idx = np.argmin(scores[val_loss_key])
        # print out minimum training loss and validation loss at min_idx
        print(f"min training loss: {scores[train_loss_key][min_idx]}")
        print(f"min validation loss: {scores[val_loss_key][min_idx]}")
        print(f"min idx: {min_idx}\n")


# %%


# %%
