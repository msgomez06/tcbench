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

# PM_cnn_files = {
#     "deterministic": load_pickle(
#         results_path
#         + "CNN_Regularized_CNN_losses_04-02-10h47_panguweather_deterministic_[32,64,128]_masked.pkl"
#     ),
#     "probabilistic": load_pickle(
#         results_path
#         + "CNN_Regularized_CNN_losses_04-02-13h23_panguweather_probabilistic_[32,64,128]_masked.pkl"
#     ),
# }

# PU_cnn_files = {
#     "deterministic": load_pickle(
#         results_path
#         + "CNN_Regularized_CNN_losses_04-02-11h56_panguweather_deterministic_[32,64,128]_unmasked.pkl"
#     ),
#     "probabilistic": load_pickle(
#         results_path
#         + "CNN_Regularized_CNN_losses_04-02-14h23_panguweather_probabilistic_[32,64,128]_unmasked.pkl"
#     ),
# }
PM_cnn_files = {
    "deterministic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-12h56_panguweather_deterministic_[32,64,128].pkl"
    ),
    "probabilistic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-17h13_panguweather_probabilistic_[32,64,128].pkl"
    ),
}

PU_cnn_files = {
    "deterministic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-15h08_panguweather_deterministic_[32,64,128].pkl"
    ),
    "probabilistic": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-18h08_panguweather_probabilistic_[32,64,128].pkl"
    ),
}
PM_unet_files = {
    "probabilistic": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-18-20h11_panguweather_probabilistic_[32,64,128].pkl"
    ),
    "deterministic": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-18-16h30_panguweather_deterministic_[32,64,128].pkl"
    ),
}
PU_unet_files = {
    "probabilistic": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-18-21h50_panguweather_probabilistic_[32,64,128].pkl"
    ),
    "deterministic": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-18-18h33_panguweather_deterministic_[32,64,128].pkl"
    ),
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
fm_cnn_files = {
    "probabilistic_masked": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-23h12_fourcastnetv2_probabilistic_[32,64,128].pkl"
    ),
    "deterministic_masked": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-19h04_fourcastnetv2_deterministic_[32,64,128].pkl"
    ),
}
fu_cnn_files = {
    "probabilistic_unmasked": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-26-00h08_fourcastnetv2_probabilistic_[32,64,128].pkl"
    ),
    "deterministic_unmasked": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_06-25-21h09_fourcastnetv2_deterministic_[32,64,128].pkl"
    ),
}

fm_unet_files = {
    "probabilistic_masked": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-24-16h26_fourcastnetv2_probabilistic_[32,64,128].pkl"
    ),
    "deterministic_masked": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-25-09h49_fourcastnetv2_deterministic_[32,64,128].pkl"
    ),
}
fu_unet_files = {
    "probabilistic_unmasked": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-24-17h54_fourcastnetv2_probabilistic_[32,64,128].pkl"
    ),
    "deterministic_unmasked": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_06-25-12h00_fourcastnetv2_deterministic_[32,64,128].pkl"
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
    "deterministic_mlr": load_pickle(
        results_path + "TorchMLR_losses_04-10-14h01_ERA5_deterministic_unmasked.pkl"
    ),
    "probabilistic_mlr": load_pickle(
        results_path + "TorchMLR_losses_04-10-14h38_ERA5_probabilistic_unmasked.pkl"
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
        results_path + "SimpleANN_losses_04-10-13h59_ERA5_deterministic_unmasked.pkl"
    ),
    "probabilistic_ann": load_pickle(
        results_path + "SimpleANN_losses_04-10-14h46_ERA5_probabilistic_unmasked.pkl"
    ),
}
era5_cnn_masked = {
    "deterministic_cnn": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_07-23-15h40_ERA5_deterministic_[32,64,128].pkl"
    ),
    "probabilistic_cnn": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_07-24-12h09_ERA5_probabilistic_[32,64,128].pkl"
    ),
}
era5_cnn_unmasked = {
    "deterministic_cnn": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_07-24-10h57_ERA5_deterministic_[32,64,128].pkl"
    ),
    "probabilistic_cnn": load_pickle(
        results_path
        + "CNN_Regularized_CNN_losses_07-24-13h36_ERA5_probabilistic_[32,64,128].pkl"
    ),
}
era5_unet_masked = {
    "deterministic_unet": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_07-24-15h13_ERA5_deterministic_[32,64,128].pkl"
    ),
    "probabilistic_unet": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_07-24-16h26_ERA5_probabilistic_[32,64,128].pkl"
    ),
}
era5_unet_unmasked = {
    "deterministic_unet": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_07-24-15h48_ERA5_deterministic_[32,64,128].pkl"
    ),
    "probabilistic_unet": load_pickle(
        results_path
        + "CNN_UNet_v2_losses_07-24-17h01_ERA5_probabilistic_[32,64,128].pkl"
    ),
}


# %%
for filedict, name in zip(
    [
        PM_mlr_files,
        PU_mlr_files,
        PU_ann_files,
        PM_ann_files,
        PM_cnn_files,
        PU_cnn_files,
        PM_unet_files,
        PU_unet_files,
        fm_mlr_files,
        fu_mlr_files,
        fm_ann_files,
        fu_ann_files,
        fm_cnn_files,
        fu_cnn_files,
        fm_unet_files,
        fu_unet_files,
        era5_mlr_masked,
        era5_mlr_unmasked,
        era5_ann_masked,
        era5_ann_unmasked,
        era5_cnn_masked,
        era5_cnn_unmasked,
        era5_unet_masked,
        era5_unet_unmasked,
    ],
    [
        "PanMask (MLR)",
        "PanUnmask (MLR)",
        "PanUnmask (ANN)",
        "PanMask (ANN)",
        "Pangu_mask CNN",
        "Pangu_unmask CNN",
        "Pangu UNet (masked)",
        "Pangu UNet (unmasked)",
        "fourMask (MLR)",
        "fourUnmask (MLR)",
        "fourMask (ANN)",
        "fourUnmask (ANN)",
        "fourMask CNN",
        "fourUnmask CNN",
        "fourMask UNet",
        "fourUnmask UNet",
        "ERA5 Masked (MLR)",
        "ERA5 Unmasked (MLR)",
        "ERA5 Masked (ANN)",
        "ERA5 Unmasked (ANN)",
        "ERA5 Masked (CNN)",
        "ERA5 Unmasked (CNN)",
        "ERA5 Masked (UNet)",
        "ERA5 Unmasked (UNet)",
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
