import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_denormalise(input_path, scaling_path):
    df = pd.read_csv(input_path)
    scales = pd.read_csv(scaling_path, index_col=0)

    df_real = df.copy()
    valid_cols = [c for c in df.columns if c in scales.index]

    for col in valid_cols:
        col_min = scales.loc[col, "min"]
        col_max = scales.loc[col, "max"]
        df_real[col] = df[col] * (col_max - col_min) + col_min

    return df_real, scales


def relabel_events(df):
    """
    Relabels events based on denormalised physics logic from preprocessing.
    Precedence (lowest to highest): Normal -> Lull -> Ramp -> CutOut -> BigCutOut
    """
    from consts import BIG_CUT_OUT_WIND, CUT_OUT_WIND, LULL_SPEED, RAMP_THRESHOLD_KW

    df_relabelled = df.copy()

    # 1. Default all to Normal
    df_relabelled["_EventLabel"] = "Normal"

    # 2. Lull: wind speed < LULL_SPEED for 60+ consecutive data points (10 hours)
    is_low_wind = df_relabelled["WindSpeed_Hub"] < LULL_SPEED
    group_ids = (is_low_wind != is_low_wind.shift()).cumsum()
    group_sizes = is_low_wind.groupby(group_ids).transform("size")
    df_relabelled.loc[is_low_wind & (group_sizes >= 60), "_EventLabel"] = "Lull"

    # 3. Ramp: Power output changes by RAMP_THRESHOLD_KW+ in one time step (Absolute difference)
    power_diff = df_relabelled["ActivePower"].diff().fillna(0).abs()
    df_relabelled.loc[power_diff > RAMP_THRESHOLD_KW, "_EventLabel"] = "Ramp"

    # 4. CutOut: wind speed > CUT_OUT_WIND
    df_relabelled.loc[df_relabelled["WindSpeed_Hub"] > CUT_OUT_WIND, "_EventLabel"] = (
        "CutOut"
    )

    # 5. BigCutOut: wind speed > BIG_CUT_OUT_WIND
    df_relabelled.loc[
        df_relabelled["WindSpeed_Hub"] > BIG_CUT_OUT_WIND, "_EventLabel"
    ] = "BigCutOut"

    return df_relabelled


def identify_event_groups(df, target_col="_EventLabel"):
    df = df.copy()
    df["group_key"] = (df[target_col] != df[target_col].shift()).cumsum()
    df["EventID"] = df[target_col].astype(str) + "_" + df["group_key"].astype(str)
    return df.drop(columns=["group_key"])


def normalise_and_save(df_real, scales, output_path, apply_relabeling=False):
    # Apply relabeling and re-grouping if requested
    if apply_relabeling:
        df_real = relabel_events(df_real)
        df_real = identify_event_groups(df_real)

    df_norm = df_real.copy()
    valid_cols = [c for c in df_real.columns if c in scales.index]

    for col in valid_cols:
        col_min = scales.loc[col, "min"]
        col_max = scales.loc[col, "max"]
        df_norm[col] = (df_real[col] - col_min) / (col_max - col_min)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_norm.to_csv(output_path, index=False)


def plot_validation(df_real, df_syn, method_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Physics Check (Scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_real["WindSpeed_Hub"],
        df_real["ActivePower"],
        c="blue",
        alpha=0.2,
        s=10,
        label="Original",
    )
    plt.scatter(
        df_syn["WindSpeed_Hub"],
        df_syn["ActivePower"],
        c="red",
        alpha=0.6,
        s=10,
        marker="x",
        label="Synthetic",
    )
    plt.title(f"Physics Validation: {method_name}")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "validation_scatter.png"))
    plt.close()

    # 2. Time Series Check (First 3 events)
    syn_ids = df_syn["EventID"].unique()[:3]
    for i, eid in enumerate(syn_ids):
        subset = df_syn[df_syn["EventID"] == eid]
        plt.figure()
        plt.plot(subset["ActivePower"].values, "r--")
        plt.title(f"Synthetic Event Trace: {eid}")
        plt.savefig(os.path.join(save_dir, f"trace_{i}.png"))
        plt.close()
