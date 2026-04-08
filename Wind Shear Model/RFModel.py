import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

matplotlib.use("Agg")
HEIGHT_REF = 10.0
HEIGHT_DIR_MAX = 100.0
HEIGHT_HUB = 110.0

# Input Directories
TRAIN_DIR = "../Data/RF Model Data Files/Train"
TEST_DIR = "../Data/RF Model Data Files/Test"

TRAIN_OUT_DIR = "../Data/Energy Model Data Files/Train"
TEST_OUT_DIR = "../Data/Energy Model Data Files/Test"

MODEL_SAVE_DIR = "./Models"
PLOT_SAVE_DIR = "./Plots"

DATA_MAPPINGS = {
    "original_and_smote_data.csv": (
        "original_and_smote_test_data.csv",
        "original_and_smote_scaling_params.csv",
    ),
    "train_original.csv": ("test.csv", "scaling_params.csv"),
    "train_balanced.csv": ("test.csv", "scaling_params.csv"),
    "smote_data.csv": ("smote_test_data.csv", "smote_scaling_params.csv"),
    "gan_data.csv": ("gan_test_data.csv", "gan_scaling_params.csv"),
    "original_and_gan_data.csv": (
        "original_and_gan_test_data.csv",
        "original_and_gan_scaling_params.csv",
    ),
}

np.random.seed(42)


def load_scalers(scaler_path):
    return pd.read_csv(scaler_path, index_col=0)


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


def predict_power_law(v_ref, h_ref, h_target, alpha=(1 / 7)):
    return v_ref * np.power((h_target / h_ref), alpha)


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

    def tune_and_train(self, train_df, scaling_params):
        v_hub = unscale_column(
            train_df["WindSpeed_Hub"], "WindSpeed_Hub", scaling_params
        )
        v_10 = unscale_column(train_df["ERA5_10mWS"], "ERA5_10mWS", scaling_params)

        sin_10 = unscale_column(
            train_df["ERA5_10mWD_Sin"], "ERA5_10mWD_Sin", scaling_params
        )
        cos_10 = unscale_column(
            train_df["ERA5_10mWD_Cos"], "ERA5_10mWD_Cos", scaling_params
        )
        sin_100 = unscale_column(
            train_df["ERA5_100mWD_Sin"], "ERA5_100mWD_Sin", scaling_params
        )
        cos_100 = unscale_column(
            train_df["ERA5_100mWD_Cos"], "ERA5_100mWD_Cos", scaling_params
        )

        epsilon = 1e-6
        v_hub_safe = np.maximum(v_hub, epsilon)
        v_10_safe = np.maximum(v_10, epsilon)

        log_h_ratio = np.log(HEIGHT_HUB / HEIGHT_REF)
        alpha = np.log(v_hub_safe / v_10_safe) / log_h_ratio
        alpha = alpha.clip(-0.5, 1.5)

        dir_10 = get_degrees_from_components(sin_10, cos_10)
        dir_100 = get_degrees_from_components(sin_100, cos_100)

        angle_diff = calculate_smallest_angle_diff(dir_100, dir_10)
        veer_rate = angle_diff / (HEIGHT_DIR_MAX - HEIGHT_REF)

        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.5, neginf=-0.5)
        veer_rate = np.nan_to_num(veer_rate, nan=0.0)

        X = train_df[[c for c in self.features if c in train_df.columns]]

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [2, 4],
        }

        rf_alpha = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        rf_alpha.fit(X, alpha)
        self.model_alpha = rf_alpha.best_estimator_

        rf_veer = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        rf_veer.fit(X, veer_rate)
        self.model_veer = rf_veer.best_estimator_

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


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUT_DIR, exist_ok=True)

    for train_file, (test_file, scaler_file) in DATA_MAPPINGS.items():
        train_path = os.path.join(TRAIN_DIR, train_file)
        test_path = os.path.join(TEST_DIR, test_file)
        scaler_path = os.path.join(TRAIN_DIR, scaler_file)

        scalers = load_scalers(scaler_path)
        train_df = pd.read_csv(train_path)

        rf = WindShearRF()
        rf.tune_and_train(train_df, scalers)

        model_filename = f"{train_file.replace('.csv', '')}_model.joblib"
        save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
        joblib.dump(rf, save_path)

        TARGET_HEIGHT_AUGMENT = 110.0

        v_train_110, sin_train, cos_train = rf.predict(
            train_df, scalers, target_height_m=TARGET_HEIGHT_AUGMENT
        )
        test_df = pd.read_csv(test_path)
        v_test_110, sin_test, cos_test = rf.predict(
            test_df, scalers, target_height_m=TARGET_HEIGHT_AUGMENT
        )

        def normalize_sin_cos(arr):
            return (arr + 1.0) / 2.0

        ws_min = v_train_110.min()
        ws_max = v_train_110.max()

        train_df["Pred_110mWS"] = (v_train_110 - ws_min) / (ws_max - ws_min)
        test_df["Pred_110mWS"] = (v_test_110 - ws_min) / (ws_max - ws_min)

        train_df["Pred_110mWD_Sin"] = normalize_sin_cos(sin_train)
        train_df["Pred_110mWD_Cos"] = normalize_sin_cos(cos_train)
        test_df["Pred_110mWD_Sin"] = normalize_sin_cos(sin_test)
        test_df["Pred_110mWD_Cos"] = normalize_sin_cos(cos_test)

        new_scalers = scalers.copy()
        new_rows = pd.DataFrame(
            {"min": [-1.0, -1.0, ws_min], "max": [1.0, 1.0, ws_max]},
            index=["Pred_110mWD_Sin", "Pred_110mWD_Cos", "Pred_110mWS"],
        )
        new_scalers = pd.concat([new_scalers, new_rows])

        model_id = train_file.replace(".csv", "")
        new_train_filename = f"{model_id}_train_with_preds.csv"
        new_test_filename = f"{model_id}_test_with_preds.csv"
        new_scaler_filename = f"{model_id}_scaling_params_with_preds.csv"

        train_df.to_csv(os.path.join(TRAIN_OUT_DIR, new_train_filename), index=False)
        test_df.to_csv(os.path.join(TEST_OUT_DIR, new_test_filename), index=False)
        new_scalers.to_csv(os.path.join(TRAIN_OUT_DIR, new_scaler_filename))

        v_pred_100_rf, sin_pred_100, cos_pred_100 = rf.predict(
            test_df, scalers, target_height_m=100.0
        )
        v_pred_110_rf, _, _ = rf.predict(test_df, scalers, target_height_m=HEIGHT_HUB)

        v_true_10_all = unscale_column(test_df["ERA5_10mWS"], "ERA5_10mWS", scalers)
        v_pred_110_pl = predict_power_law(
            v_true_10_all, HEIGHT_REF, HEIGHT_HUB, alpha=(1 / 7)
        )

        sin_true_100_all = unscale_column(
            test_df["ERA5_100mWD_Sin"], "ERA5_100mWD_Sin", scalers
        )
        cos_true_100_all = unscale_column(
            test_df["ERA5_100mWD_Cos"], "ERA5_100mWD_Cos", scalers
        )
        v_true_110_all = unscale_column(
            test_df["WindSpeed_Hub"], "WindSpeed_Hub", scalers
        )
        v_true_100_all = unscale_column(test_df["ERA5_100mWS"], "ERA5_100mWS", scalers)

        model_name_clean = train_file.replace(".csv", "").replace("_", " ").title()

        def plot_scatter(y_true, y_pred, title, filename_suffix):
            plt.figure(figsize=(10, 8))
            if "_EventLabel" in test_df.columns:
                plot_labels = test_df["_EventLabel"].fillna("Unknown")
                unique_labels = sorted(plot_labels.unique())
                for label in unique_labels:
                    mask = plot_labels == label
                    plt.scatter(
                        y_true[mask], y_pred[mask], alpha=0.6, s=15, label=label
                    )
                plt.legend()
            else:
                plt.scatter(y_true, y_pred, alpha=0.5, s=10, c="blue")

            all_vals = np.concatenate([y_true, y_pred])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            plt.plot(
                [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit"
            )
            plt.title(title)
            plt.xlabel("Actual Wind Speed (m/s)")
            plt.ylabel("Predicted Wind Speed (m/s)")
            plt.grid(True, alpha=0.3)
            plot_filename = f"{model_id}_{filename_suffix}.png"
            plt.savefig(os.path.join(PLOT_SAVE_DIR, plot_filename))
            plt.close()

        plot_scatter(
            v_true_110_all,
            v_pred_110_rf,
            f"Hub Height Wind Speed: Actual vs Predicted (RF)\nModel: {model_name_clean}",
            "HubSpeed_Scatter_RF",
        )

        plot_scatter(
            v_true_100_all,
            v_pred_100_rf,
            f"100m Wind Speed: Actual vs Predicted (RF)\nModel: {model_name_clean}",
            "100mSpeed_Scatter_RF",
        )

        plot_scatter(
            v_true_110_all,
            v_pred_110_pl,
            f"Hub Height Wind Speed: Actual vs Predicted (Power Law)\nModel: {model_name_clean}",
            "HubSpeed_Scatter_PL",
        )

        results = []
        event_labels = ["All"]
        if "_EventLabel" in test_df.columns:
            event_labels += sorted(test_df["_EventLabel"].dropna().unique().tolist())

        for label in event_labels:
            if label == "All":
                mask = np.ones(len(test_df), dtype=bool)
            else:
                mask = test_df["_EventLabel"] == label

            if not mask.any():
                continue

            mae_sin = mean_absolute_error(sin_true_100_all[mask], sin_pred_100[mask])
            mae_cos = mean_absolute_error(cos_true_100_all[mask], cos_pred_100[mask])
            mae_dir = (mae_sin + mae_cos) / 2

            rmse_dir = np.sqrt(
                (
                    mean_squared_error(sin_true_100_all[mask], sin_pred_100[mask])
                    + mean_squared_error(cos_true_100_all[mask], cos_pred_100[mask])
                )
                / 2
            )

            r2_dir = (
                r2_score(sin_true_100_all[mask], sin_pred_100[mask])
                + r2_score(cos_true_100_all[mask], cos_pred_100[mask])
            ) / 2

            results.append([label, "RF", "100m", "Dir", mae_dir, rmse_dir, r2_dir])

            mae_100_rf = mean_absolute_error(v_true_100_all[mask], v_pred_100_rf[mask])
            rmse_100_rf = np.sqrt(
                mean_squared_error(v_true_100_all[mask], v_pred_100_rf[mask])
            )
            r2_100_rf = r2_score(v_true_100_all[mask], v_pred_100_rf[mask])
            results.append(
                [label, "RF", "100m", "Spd", mae_100_rf, rmse_100_rf, r2_100_rf]
            )

            mae_110_rf = mean_absolute_error(v_true_110_all[mask], v_pred_110_rf[mask])
            rmse_110_rf = np.sqrt(
                mean_squared_error(v_true_110_all[mask], v_pred_110_rf[mask])
            )
            r2_110_rf = r2_score(v_true_110_all[mask], v_pred_110_rf[mask])
            results.append(
                [label, "RF", "Hub", "Spd", mae_110_rf, rmse_110_rf, r2_110_rf]
            )

            mae_110_pl = mean_absolute_error(v_true_110_all[mask], v_pred_110_pl[mask])
            rmse_110_pl = np.sqrt(
                mean_squared_error(v_true_110_all[mask], v_pred_110_pl[mask])
            )
            r2_110_pl = r2_score(v_true_110_all[mask], v_pred_110_pl[mask])
            results.append(
                [label, "Power Law", "Hub", "Spd", mae_110_pl, rmse_110_pl, r2_110_pl]
            )

        res_df = pd.DataFrame(
            results,
            columns=["Regime", "Model", "Level", "Variable", "MAE", "RMSE", "R2"],
        )
        print(f"\nRESULTS FOR: {test_file}")
        print(res_df.to_string(index=False, float_format=lambda x: "{:,.4f}".format(x)))


if __name__ == "__main__":
    main()
