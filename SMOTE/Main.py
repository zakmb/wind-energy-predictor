import os

from Utils import (
    identify_event_groups,
    load_and_denormalise,
    normalise_and_save,
    plot_validation,
)
from Models import (
    CombinedSMOTE,
    ExtrapolatingSMOTE,
    IntervalSMOTE,
    PhysicsAwareSMOTE,
)

DATA_DIR = "../Data/Processed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_original.csv")
SCALE_FILE = os.path.join(DATA_DIR, "scaling_params.csv")
OUT_BASE = "../Data/SMOTE"

FEATURES = [
    "ERA5_10mWS",
    "ERA5_10mWD_Sin",
    "ERA5_10mWD_Cos",
    "ERA5_100mWS",
    "ERA5_100mWD_Sin",
    "ERA5_100mWD_Cos",
    "AirDensity",
    "Temperature",
    "Pressure",
    "ERA5_WH",
    "WindSpeed_Hub",
    "ActivePower",
]

# Configuration Constants
N_SAMPLES = 4000
NEIGHBORS = 3
METHODS_TO_RUN = ["interval", "extrapolate", "physics", "combined"]


def main():
    df_real, scales = load_and_denormalise(TRAIN_FILE, SCALE_FILE)
    df_real = identify_event_groups(df_real)

    for method in METHODS_TO_RUN:
        out_dir = os.path.join(OUT_BASE, method.capitalize())
        out_file = os.path.join(out_dir, f"{method}_smote.csv")

        # instantiate the model
        if method == "interval":
            smote = IntervalSMOTE(N_SAMPLES, NEIGHBORS, FEATURES)
        elif method == "extrapolate":
            smote = ExtrapolatingSMOTE(N_SAMPLES, NEIGHBORS, FEATURES)
        elif method == "physics":
            smote = PhysicsAwareSMOTE(N_SAMPLES, NEIGHBORS, FEATURES)
        else:
            smote = CombinedSMOTE(N_SAMPLES, NEIGHBORS, FEATURES)

        # generate
        df_syn = smote.generate(df_real, "_EventLabel")

        # validate + save
        plot_validation(df_real, df_syn, method, out_dir)
        normalise_and_save(df_syn, scales, out_file)


if __name__ == "__main__":
    main()
