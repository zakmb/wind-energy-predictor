import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from consts import (
    BIG_CUT_OUT_WIND,
    CUT_OUT_WIND,
    LULL_DURATION_HOURS,
    LULL_SPEED,
    PHYSICS_LIMITS,
    RAMP_THRESHOLD_KW,
    RATED_POWER_KW,
    CUT_OUT_SPEED_MPS,
)

INPUT_FILE_PATH = "../Data/Raw/InitialData.csv"
OUTPUT_DIR = "../Data/Processed/"
PLOT_DIR = os.path.join(OUTPUT_DIR, "Plots")

OUTPUT_TRAIN_ORIG = os.path.join(OUTPUT_DIR, "train_original.csv")
OUTPUT_TRAIN_BAL = os.path.join(OUTPUT_DIR, "train_balanced.csv")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.csv")
OUTPUT_SCALING = os.path.join(OUTPUT_DIR, "scaling_params.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

FORECAST_COLS = [
    "StartTime",
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
    "_EventLabel",
    "_EventID",
]

LULL_STEPS = int((LULL_DURATION_HOURS * 60) / 10)


def load_and_clean_data():
    df = pd.read_csv(INPUT_FILE_PATH)

    col_map = {
        "StartTime": "StartTime",
        "WindSpeed_mps_Mean": "WindSpeed_Hub",
        "Power_kW_Mean": "ActivePower",
        "Pitch_Deg_Mean": "PitchAngle",
        "temp_2_Mean": "Temperature",
        "ERA5_MSLP_hPa": "Pressure",
        "ERA5_SWH_m": "ERA5_WH",
        "ERA5_10mWS_mps": "ERA5_10mWS",
        "ERA5_10mWD_deg": "ERA5_10mWD",
        "ERA5_100mWS_mps": "ERA5_100mWS",
        "ERA5_100mWD_deg": "ERA5_100mWD",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)

    df = df.sort_values("StartTime", kind="stable").reset_index(drop=True)

    df = detect_label_id_events(df)

    for col, (min_v, max_v) in PHYSICS_LIMITS.items():
        if col in df.columns:
            out_of_bounds = (df[col] < min_v) | (df[col] > max_v)
            df = df[~out_of_bounds]
            if col not in ["WindSpeed_Hub", "ActivePower", "PitchAngle"]:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df["AirDensity"] = (df["Pressure"] * 100) / (287.058 * (df["Temperature"] + 273.15))

    mask_maint = (
        (df["WindSpeed_Hub"] > 5.0)
        & (df["WindSpeed_Hub"] < 20.0)
        & (df["ActivePower"] <= 10.0)
    )
    df = df[~mask_maint]

    for raw_dir in ["ERA5_10mWD", "ERA5_100mWD"]:
        df[f"{raw_dir}_Sin"] = np.sin(np.deg2rad(df[raw_dir]))
        df[f"{raw_dir}_Cos"] = np.cos(np.deg2rad(df[raw_dir]))

    return df.dropna(subset=["WindSpeed_Hub", "ActivePower"]).reset_index(drop=True)


def detect_label_id_events(df, max_hours=12):
    mask_cut = df["WindSpeed_Hub"] > CUT_OUT_WIND
    mask_big = df["WindSpeed_Hub"] > BIG_CUT_OUT_WIND

    diffs = df["ActivePower"].diff().fillna(0)
    mask_ramp = diffs.abs() > RAMP_THRESHOLD_KW

    is_low_wind = df["WindSpeed_Hub"] < LULL_SPEED

    low_wind_block_id = (is_low_wind != is_low_wind.shift()).cumsum()

    df["block_id"] = low_wind_block_id
    block_durations = (
        df[is_low_wind]
        .groupby("block_id")["StartTime"]
        .transform(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    )

    mask_lull = is_low_wind & (block_durations >= 10.0)

    df["_EventLabel"] = "Normal"
    df.loc[mask_lull, "_EventLabel"] = "Lull"
    df.loc[mask_ramp, "_EventLabel"] = "Ramp"
    df.loc[mask_cut, "_EventLabel"] = "CutOut"
    df.loc[mask_big, "_EventLabel"] = "BigCutOut"

    event_ids = np.zeros(len(df), dtype=int)
    current_id = 0
    block_start_time = df["StartTime"].iloc[0]
    current_label = df["_EventLabel"].iloc[0]

    for i in range(len(df)):
        row_time = df["StartTime"].iloc[i]
        row_label = df["_EventLabel"].iloc[i]
        hours_elapsed = (row_time - block_start_time).total_seconds() / 3600

        if row_label != current_label or hours_elapsed >= max_hours:
            current_id += 1
            block_start_time = row_time
            current_label = row_label

        event_ids[i] = current_id

    df["_EventID"] = event_ids
    df.drop(columns=["block_id"], inplace=True, errors="ignore")
    return df


def process_data(df):
    test_ids = []
    test_ids.extend(df[df["_EventLabel"] == "BigCutOut"]["_EventID"].unique())
    test_ids.extend(get_last_n_ids(df, "CutOut", 4))
    test_ids.extend(get_last_n_ids(df, "Lull", 10))
    test_ids.extend(get_last_n_ids(df, "Ramp", 300))

    test_ids = list(dict.fromkeys(test_ids))

    normal_df = df[df["_EventLabel"] == "Normal"]
    split_pt = int(len(normal_df) * 0.8)
    split_time = normal_df.iloc[split_pt]["StartTime"]

    test_mask = (df["_EventID"].isin(test_ids)) | (
        (df["_EventLabel"] == "Normal") & (df["StartTime"] >= split_time)
    )

    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    generate_plots(train_df)

    scale_cols = [
        c
        for c in train_df.columns
        if c in FORECAST_COLS and c not in ["_EventLabel", "_EventID", "StartTime"]
    ]
    scaling_params = train_df[scale_cols].agg(["min", "max"]).transpose()
    scaling_params.to_csv(OUTPUT_SCALING)

    for c in scale_cols:
        denominator = (
            scaling_params.loc[c, "max"] - scaling_params.loc[c, "min"]
        ) + 1e-8
        train_df[c] = (train_df[c] - scaling_params.loc[c, "min"]) / denominator
        test_df[c] = (test_df[c] - scaling_params.loc[c, "min"]) / denominator
        test_df[c] = test_df[c].clip(0.0, 1.0)

    train_df[FORECAST_COLS].to_csv(OUTPUT_TRAIN_ORIG, index=False)

    test_df[FORECAST_COLS].sort_values(["StartTime"], kind="stable").to_csv(
        OUTPUT_TEST, index=False
    )

    dfs_to_add = [train_df[FORECAST_COLS]]
    current_max_id = train_df["_EventID"].max()

    multipliers = [("CutOut", 3000), ("Lull", 8), ("Ramp", 8)]

    for label, count in multipliers:
        label_ids = train_df[train_df["_EventLabel"] == label]["_EventID"].unique()
        for _ in range(count):
            for eid in label_ids:
                current_max_id += 1
                new_block = train_df[train_df["_EventID"] == eid][FORECAST_COLS].copy()
                new_block["_EventID"] = current_max_id
                dfs_to_add.append(new_block)

    train_bal = (
        pd.concat(dfs_to_add)
        .sort_values(["_EventID", "StartTime"], kind="stable")
        .reset_index(drop=True)
    )
    train_bal.to_csv(OUTPUT_TRAIN_BAL, index=False)


def get_last_n_ids(df, label, n):
    ids = df[df["_EventLabel"] == label]["_EventID"].unique()
    return list(ids[-n:]) if len(ids) >= n else list(ids)


def print_stats(df, name):
    _ = (df, name)


def generate_plots(df):
    plt.figure(figsize=(10, 6))
    if "_EventLabel" in df.columns:
        sns.scatterplot(
            data=df,
            x="WindSpeed_Hub",
            y="ActivePower",
            hue="_EventLabel",
            alpha=0.5,
            s=15,
            palette="viridis",
        )
    else:
        plt.scatter(df["WindSpeed_Hub"], df["ActivePower"], alpha=0.1, s=10, c="blue")

    plt.axhline(y=RATED_POWER_KW, color="r", linestyle="--", label="Rated Power")
    plt.axvline(x=CUT_OUT_SPEED_MPS, color="r", linestyle="--", label="Cut Out")
    plt.axvline(x=LULL_SPEED, color="r", linestyle="--", label="Cut In")
    plt.title("Power Curve Training Data")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "power_curve_training_data.png"))
    plt.close()


if __name__ == "__main__":
    full_df = load_and_clean_data()
    process_data(full_df)
