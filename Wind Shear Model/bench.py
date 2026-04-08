import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from numpy.lib.stride_tricks import sliding_window_view

sys.modules["RFModel"] = sys.modules["__main__"]

RF_MODEL_FILE = "train_original_model.joblib"
PINN_MODEL_FILE = "train_original_train_with_preds_PINN_LSTM.keras"
TEST_CSV_FILE = "test.csv"
SCALER_FILE = "scaling_params.csv"

RF_MODEL_PATH = f"/content/{RF_MODEL_FILE}"
PINN_MODEL_PATH = f"/content/{PINN_MODEL_FILE}"
TEST_CSV_PATH = f"/content/{TEST_CSV_FILE}"
SCALER_PATH = f"/content/{SCALER_FILE}"

HEIGHT_REF = 10.0
HEIGHT_DIR_MAX = 100.0
HEIGHT_HUB = 110.0
WINDOW_SIZE = 6
GROUP_COL = "_EventID"

FEATURES = [
    "Pred_110mWD_Sin",
    "Pred_110mWD_Cos",
    "ERA5_WH",
    "AirDensity",
    "Temperature",
    "Pressure",
    "WindSpeed_Hub",
]


def unscale_column(series, col_name, params):
    if col_name not in params.index:
        if "100mWS" in col_name and "ERA5_10mWS" in params.index:
            ref_col = "ERA5_10mWS"
            return (
                series * (params.loc[ref_col, "max"] - params.loc[ref_col, "min"])
                + params.loc[ref_col, "min"]
            )
        return series
    return (
        series * (params.loc[col_name, "max"] - params.loc[col_name, "min"])
        + params.loc[col_name, "min"]
    )


def get_degrees_from_components(sin_col, cos_col):
    angles = np.degrees(np.arctan2(sin_col, cos_col))
    return (angles + 360) % 360


def calculate_smallest_angle_diff(angle_target, angle_ref):
    diff = angle_target - angle_ref
    return (diff + 180) % 360 - 180


class WindShearRF:
    def __init__(self):
        self.features = [
            "ERA5_10mWS",
            "Pressure",
            "Temperature",
            "ERA5_10mWD_Sin",
            "ERA5_10mWD_Cos",
            "ERA5_WH",
        ]
        self.model_alpha = None
        self.model_veer = None

    def predict(self, df, scaling_params, target_height_m):
        X = df[[c for c in self.features if c in df.columns]]

        pred_alpha = self.model_alpha.predict(X)
        pred_veer_rate = self.model_veer.predict(X)

        v_10_phys = unscale_column(df["ERA5_10mWS"], "ERA5_10mWS", scaling_params)
        sin_10_phys = unscale_column(
            df["ERA5_10mWD_Sin"], "ERA5_10mWD_Sin", scaling_params
        )
        cos_10_phys = unscale_column(
            df["ERA5_10mWD_Cos"], "ERA5_10mWD_Cos", scaling_params
        )
        dir_10_deg = get_degrees_from_components(sin_10_phys, cos_10_phys)

        v_target = v_10_phys * np.power((target_height_m / HEIGHT_REF), pred_alpha)

        height_delta = target_height_m - HEIGHT_REF
        dir_target_deg = dir_10_deg + (pred_veer_rate * height_delta)

        dir_target_rad = np.radians(dir_target_deg)
        sin_target = np.sin(dir_target_rad)
        cos_target = np.cos(dir_target_rad)

        return v_target, sin_target, cos_target


def create_inference_sequences(df, window_size):
    X_list = []
    for event_id, group_df in df.groupby(GROUP_COL, sort=True):
        available_features = [col for col in FEATURES if col in group_df.columns]
        features_vals = group_df[available_features].values.astype("float32")

        if features_vals.shape[1] < len(FEATURES):
            pad = np.zeros(
                (features_vals.shape[0], len(FEATURES) - features_vals.shape[1])
            )
            features_vals = np.hstack([features_vals, pad])

        if len(group_df) < window_size:
            pad_length = window_size - len(group_df)
            features_vals = np.vstack(
                [np.repeat(features_vals[[0]], pad_length, axis=0), features_vals]
            )

        X_windows = sliding_window_view(
            features_vals, window_shape=(window_size, features_vals.shape[1])
        ).squeeze(axis=1)
        X_list.append(X_windows)

    if not X_list:
        return np.array([])
    return np.vstack(X_list)


def run_colab_benchmark():
    test_df = pd.read_csv(TEST_CSV_PATH)

    test_df = test_df.head(144)

    scalers = pd.read_csv(SCALER_PATH, index_col=0)

    rf_pipeline = joblib.load(RF_MODEL_PATH)
    pinn_model = tf.keras.models.load_model(PINN_MODEL_PATH, compile=False)

    # --- A. BENCHMARK RANDOM FOREST ---
    _, _, _ = rf_pipeline.predict(test_df.head(10), scalers, target_height_m=HEIGHT_HUB)

    start_time = time.perf_counter()
    v_pred_110, sin_pred, cos_pred = rf_pipeline.predict(
        test_df, scalers, target_height_m=HEIGHT_HUB
    )
    rf_time_ms = (time.perf_counter() - start_time) * 1000

    # --- PREPARE DATA FOR PINN ---
    test_df_pinn = test_df.copy()
    ws_min = v_pred_110.min()
    ws_max = v_pred_110.max()
    test_df_pinn["WindSpeed_Hub"] = (v_pred_110 - ws_min) / (ws_max - ws_min)
    test_df_pinn["Pred_110mWD_Sin"] = (sin_pred + 1.0) / 2.0
    test_df_pinn["Pred_110mWD_Cos"] = (cos_pred + 1.0) / 2.0

    X_test_pinn = create_inference_sequences(test_df_pinn, WINDOW_SIZE)

    # --- B. BENCHMARK PINN ---
    _ = pinn_model.predict(X_test_pinn[:10], verbose=0)

    start_time = time.perf_counter()
    _ = pinn_model.predict(X_test_pinn, verbose=0)
    pinn_time_ms = (time.perf_counter() - start_time) * 1000
    total_time = rf_time_ms + pinn_time_ms
    print(f"TOTAL 24-HOUR PIPELINE LATENCY: {total_time:.2f} ms")
    print(f"(Processed {len(test_df)} rows / {len(X_test_pinn)} sequences)")


# Run the script
run_colab_benchmark()
