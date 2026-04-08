import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


def set_determinism(seed=42):
    """
    Locks down Python, NumPy, TensorFlow, and hardware-level operations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.config.experimental.enable_op_determinism()


# Initialize immediately
set_determinism(42)

RATED_POWER_KW = 7000.0
LAMBDA_PHYSICS = 0.2

MAX_EPOCHS = 300
WINDOW_SIZE = 6

MODEL_DIR = "./Models"
PLOT_DIR = "./Plots"

DATA_MAPPINGS = {
    "original_and_smote_data_train_with_preds.csv": (
        "original_and_smote_data_test_with_preds.csv",
        "original_and_smote_data_scaling_params_with_preds.csv",
    ),
}

FEATURES = [
    "Pred_110mWD_Sin",
    "Pred_110mWD_Cos",
    "ERA5_WH",
    "AirDensity",
    "Temperature",
    "Pressure",
    "WindSpeed_Hub",
]
TARGET = "ActivePower"

GROUP_COL = "_EventID"
REGIME_COL = "_EventLabel"


def load_scalers(scaler_path):
    return pd.read_csv(scaler_path, index_col=0)


def denorm_column(series, col_name, params):
    if col_name not in params.index:
        return series
    _min = params.loc[col_name, "min"]
    _max = params.loc[col_name, "max"]
    return series * (_max - _min) + _min


def create_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def create_sequences(df, window_size):
    X_list, y_list, group_list, regime_list = [], [], [], []

    for event_id, group_df in df.groupby(GROUP_COL, sort=True):

        if len(group_df) < window_size:
            pad_length = window_size - len(group_df)
            first_row = group_df.iloc[[0]]
            pad_df = pd.concat([first_row] * pad_length, ignore_index=True)
            group_df = pd.concat([pad_df, group_df], ignore_index=True)

        features_vals = group_df[FEATURES].values.astype("float32")
        target_vals = group_df[TARGET].values.astype("float32")

        if REGIME_COL in group_df.columns:
            regime_vals = group_df[REGIME_COL].fillna("Unknown").values
        else:
            regime_vals = ["Unknown"] * len(group_df)

        for i in range(len(group_df) - window_size + 1):
            X_list.append(features_vals[i : i + window_size])
            y_list.append(target_vals[i + window_size - 1])
            group_list.append(event_id)
            regime_list.append(regime_vals[i + window_size - 1])

    return (
        np.array(X_list),
        np.array(y_list),
        np.array(group_list),
        np.array(regime_list),
    )


def get_predictions_df(model, df, scaling_params):
    X_test, y_test, groups_test, regimes_test = create_sequences(df, WINDOW_SIZE)

    if len(X_test) == 0:
        return pd.DataFrame(columns=["Observed", "Predicted", "Regime", "EventID"])

    preds_norm = model.predict(X_test, verbose=0)

    y_pred = denorm_column(preds_norm.flatten(), TARGET, scaling_params)
    y_pred = np.clip(y_pred, 0.0, RATED_POWER_KW)
    y_true = denorm_column(y_test.flatten(), TARGET, scaling_params)

    return pd.DataFrame(
        {
            "Observed": y_true,
            "Predicted": y_pred,
            "Regime": regimes_test,
            "EventID": groups_test,
        }
    )


def plot_timeseries_examples(analysis_df, model_name, num_examples=3):
    if analysis_df.empty or "EventID" not in analysis_df.columns:
        return

    unique_events = analysis_df["EventID"].unique()
    if len(unique_events) == 0:
        return

    plot_rng = np.random.RandomState(42)
    chosen_events = plot_rng.choice(
        unique_events, min(num_examples, len(unique_events)), replace=False
    )

    plt.figure(figsize=(15, 4 * len(chosen_events)))

    for i, event_id in enumerate(chosen_events):
        event_data = analysis_df[analysis_df["EventID"] == event_id].reset_index(
            drop=True
        )

        plt.subplot(len(chosen_events), 1, i + 1)
        plt.plot(
            event_data.index,
            event_data["Observed"],
            label="Observed (Actual)",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            event_data.index,
            event_data["Predicted"],
            label="Predicted",
            color="orange",
            linestyle="--",
            linewidth=2,
        )

        mae = mean_absolute_error(event_data["Observed"], event_data["Predicted"])
        plt.title(
            f"{model_name} | Timeline for EventID: {event_id} | MAE: {mae:.2f} kW"
        )

        plt.xlabel("Timesteps (Sequential sequence within event)")
        plt.ylabel("Active Power (kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{PLOT_DIR}/TimeSeries_{model_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_scatter(analysis_df, model_name):
    if analysis_df.empty:
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(
        analysis_df["Observed"], analysis_df["Predicted"], alpha=0.3, s=10, c="blue"
    )

    min_val = min(analysis_df["Observed"].min(), analysis_df["Predicted"].min())
    max_val = max(analysis_df["Observed"].max(), analysis_df["Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")

    mae = mean_absolute_error(analysis_df["Observed"], analysis_df["Predicted"])
    rmse = np.sqrt(
        mean_squared_error(analysis_df["Observed"], analysis_df["Predicted"])
    )
    r2 = r2_score(analysis_df["Observed"], analysis_df["Predicted"])

    plt.title(f"{model_name}\nMAE: {mae:.2f} kW | RMSE: {rmse:.2f} kW | R2: {r2:.4f}")
    plt.xlabel("Observed Active Power (kW)")
    plt.ylabel("Predicted Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{PLOT_DIR}/Scatter_{model_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_loss_curve(history, model_name):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = np.min(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, "b-", label="Training Loss")
    plt.plot(epochs, val_loss, "r-", label="Validation Loss")

    plt.plot(
        best_epoch,
        best_val_loss,
        "go",
        markersize=10,
        label=f"Best Model (Epoch {best_epoch})",
    )
    plt.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5)

    plt.title(
        f"Training History: {model_name}\nRestored Best Model from Epoch {best_epoch}"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE + Physics Penalty)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{PLOT_DIR}/LossCurve_{model_name}.png"
    plt.savefig(filename)
    plt.close()


def calculate_metrics(analysis_df, model_name, train_dataset, model_type):
    results = []
    if analysis_df.empty:
        return results

    def get_row(data, regime_label):
        if len(data) == 0:
            return None
        mae = mean_absolute_error(data["Observed"], data["Predicted"])
        rmse = np.sqrt(mean_squared_error(data["Observed"], data["Predicted"]))
        r2 = r2_score(data["Observed"], data["Predicted"])

        return {
            "Train Dataset": train_dataset,
            "Model Type": model_type,
            "Regime": regime_label,
            "MAE (kW)": mae,
            "RMSE (kW)": rmse,
            "R2": r2,
            "Count": len(data),
        }

    results.append(get_row(analysis_df, "Overall"))

    for regime, group in analysis_df.groupby("Regime"):
        row = get_row(group, regime)
        if row:
            results.append(row)

    return results


class PINNLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_phy=0.1):
        super().__init__()
        self.lambda_phy = lambda_phy

    def call(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        if self.lambda_phy == 0:
            return mse_loss

        neg_power = tf.reduce_mean(tf.nn.relu(-y_pred))
        over_power = tf.reduce_mean(tf.nn.relu(y_pred - 1.0))

        return mse_loss + self.lambda_phy * (neg_power + over_power)


class WindPowerEstimator:
    def __init__(self, use_physics=True):
        self.use_physics = use_physics
        self.best_model = None

    def build_model(self, window_size, input_dim, lr, lambda_phy, u1=128, u2=64):
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window_size, input_dim)),
                tf.keras.layers.LSTM(u1, return_sequences=True),
                tf.keras.layers.LSTM(u2, return_sequences=False),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=PINNLoss(lambda_phy=lambda_phy),
            metrics=["mae", "mse"],
        )
        return model

    def tune_and_train(self, train_df):
        X, y, groups, _ = create_sequences(train_df, WINDOW_SIZE)
        y = y.reshape(-1, 1)

        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        batch_sizes = [32, 64, 128]
        learning_rates = [1e-3, 1e-4]
        lambda_phys = [0.1, 0.2, 0.5, 1.0] if self.use_physics else [0.0]

        best_loss = float("inf")
        best_params = {}

        for bs in batch_sizes:
            for lr in learning_rates:
                for l_phy in lambda_phys:
                    temp_model = self.build_model(WINDOW_SIZE, X.shape[2], lr, l_phy)

                    hist = temp_model.fit(
                        X_tr,
                        y_tr,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=bs,
                        verbose=0,
                        shuffle=True,
                    )

                    val_loss = hist.history["val_loss"][-1]
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_params = {"batch_size": bs, "lr": lr, "lambda_phy": l_phy}

        final_model = self.build_model(
            WINDOW_SIZE, X.shape[2], best_params["lr"], best_params["lambda_phy"]
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
        )

        history = final_model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=best_params["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
            shuffle=True,
        )

        self.best_model = final_model
        return history


def main():
    create_directories()
    final_report = []

    for train_file, (test_file, scaler_file) in DATA_MAPPINGS.items():
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        scalers = load_scalers(scaler_file)

        dataset_name = train_file.replace(".csv", "")

        experiments = [("Standard_LSTM", False), ("PINN_LSTM", True)]

        for model_type, use_physics in experiments:
            full_model_name = f"{dataset_name}_{model_type}"

            estimator = WindPowerEstimator(use_physics=use_physics)
            history = estimator.tune_and_train(train_df)

            model_path = f"{MODEL_DIR}/{full_model_name}.keras"
            estimator.best_model.save(model_path)

            plot_loss_curve(history, full_model_name)

            analysis_df = get_predictions_df(estimator.best_model, test_df, scalers)

            if not analysis_df.empty:
                plot_scatter(analysis_df, full_model_name)
                plot_timeseries_examples(analysis_df, full_model_name, num_examples=3)

                metrics = calculate_metrics(
                    analysis_df, full_model_name, dataset_name, model_type
                )
                final_report.extend(metrics)
            else:
                pass

    if final_report:
        all_res = pd.DataFrame(final_report)
        cols = [
            "Train Dataset",
            "Model Type",
            "Regime",
            "MAE (kW)",
            "RMSE (kW)",
            "R2",
            "Count",
        ]
        all_res = all_res[cols]
        all_res.to_csv("final_performance_metrics.csv", index=False)

        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        overall_rows = all_res[all_res["Regime"] == "Overall"]
        if not overall_rows.empty:
            print(
                overall_rows[
                    ["Train Dataset", "Model Type", "MAE (kW)", "RMSE (kW)", "R2"]
                ].to_string(index=False)
            )
        else:
            print(all_res.head().to_string(index=False))

        print("\nResults saved to 'final_performance_metrics.csv'")
        print(f"Models saved to '{MODEL_DIR}'")
        print(f"Plots saved to '{PLOT_DIR}'")
    else:
        print("No models trained successfully.")


if __name__ == "__main__":
    main()
