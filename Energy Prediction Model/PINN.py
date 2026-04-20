import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def set_determinism(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


set_determinism(42)

RATED_POWER_KW = 7000.0
MAX_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WINDOW_SIZE = 12
MODEL_DIR = "./Models"
PLOT_DIR = "./Plots"

DATA_MAPPINGS = {
    "gan_data_train_with_preds.csv": (
        "gan_data_scaling_params_with_preds.csv",
        "gan_data_test_with_preds.csv",
    ),
    "train_balanced_train_with_preds.csv": (
        "train_balanced_scaling_params_with_preds.csv",
        "train_balanced_test_with_preds.csv",
    ),
    "train_original_train_with_preds.csv": (
        "train_original_scaling_params_with_preds.csv",
        "train_original_test_with_preds.csv",
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
WIND_FEATURE_IDX = len(FEATURES) - 1


def load_scalers(path):
    return pd.read_csv(path, index_col=0)


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
            pad_df = pd.concat([group_df.iloc[[0]]] * pad_length, ignore_index=True)
            group_df = pd.concat([pad_df, group_df], ignore_index=True)

        features_vals = group_df[FEATURES].values.astype("float32")
        target_vals = group_df[TARGET].values.astype("float32")
        regime_vals = (
            group_df[REGIME_COL].fillna("Unknown").values
            if REGIME_COL in group_df.columns
            else ["Unknown"] * len(group_df)
        )

        for i in range(len(group_df) - window_size + 1):
            X_list.append(features_vals[i: i + window_size])
            y_list.append(target_vals[i + window_size - 1])
            group_list.append(event_id)
            regime_list.append(regime_vals[i + window_size - 1])

    return (
        np.array(X_list),
        np.array(y_list),
        np.array(group_list),
        np.array(regime_list),
    )


def get_predictions_df(model, df, scaling_params, window_size):
    X_test, y_test, groups_test, regimes_test = create_sequences(df, window_size)

    if len(X_test) == 0:
        return pd.DataFrame(columns=["Observed", "Predicted", "Regime", "EventID"])

    preds_norm = model.predict(X_test, verbose=0)
    y_pred = denorm_column(preds_norm.flatten(), TARGET, scaling_params)
    y_pred = np.clip(y_pred, 0.0, RATED_POWER_KW)
    y_true = denorm_column(y_test.flatten(), TARGET, scaling_params)

    return pd.DataFrame({
        "Observed": y_true,
        "Predicted": y_pred,
        "Regime": regimes_test,
        "EventID": groups_test,
    })


def plot_timeseries_examples(analysis_df, model_name, num_examples=3):
    if analysis_df.empty or "EventID" not in analysis_df.columns:
        return

    unique_events = analysis_df["EventID"].unique()
    if len(unique_events) == 0:
        return

    rng = np.random.RandomState(42)
    chosen_events = rng.choice(unique_events, min(num_examples, len(unique_events)), replace=False)

    plt.figure(figsize=(15, 4 * len(chosen_events)))
    for i, event_id in enumerate(chosen_events):
        event_data = analysis_df[analysis_df["EventID"] == event_id].reset_index(drop=True)
        mae = mean_absolute_error(event_data["Observed"], event_data["Predicted"])

        plt.subplot(len(chosen_events), 1, i + 1)
        plt.plot(event_data.index, event_data["Observed"], label="Observed", color="blue", linewidth=2)
        plt.plot(event_data.index, event_data["Predicted"], label="Predicted", color="orange", linestyle="--", linewidth=2)
        plt.title(f"{model_name} | EventID: {event_id} | MAE: {mae:.2f} kW")
        plt.xlabel("Timestep")
        plt.ylabel("Active Power (kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/TimeSeries_{model_name}.png")
    plt.close()


def plot_scatter(analysis_df, model_name):
    if analysis_df.empty:
        return

    mae = mean_absolute_error(analysis_df["Observed"], analysis_df["Predicted"])
    rmse = np.sqrt(mean_squared_error(analysis_df["Observed"], analysis_df["Predicted"]))
    r2 = r2_score(analysis_df["Observed"], analysis_df["Predicted"])

    min_val = min(analysis_df["Observed"].min(), analysis_df["Predicted"].min())
    max_val = max(analysis_df["Observed"].max(), analysis_df["Predicted"].max())

    plt.figure(figsize=(8, 8))
    plt.scatter(analysis_df["Observed"], analysis_df["Predicted"], alpha=0.3, s=10, c="blue")
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")
    plt.title(f"{model_name}\nMAE: {mae:.2f} kW | RMSE: {rmse:.2f} kW | R²: {r2:.4f}")
    plt.xlabel("Observed Active Power (kW)")
    plt.ylabel("Predicted Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/Scatter_{model_name}.png")
    plt.close()


def plot_loss_curve(history, model_name):
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, "b-", label="Training Loss")

    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        best_epoch = np.argmin(val_loss) + 1
        plt.plot(epochs, val_loss, "r-", label="Validation Loss")
        plt.plot(best_epoch, np.min(val_loss), "go", markersize=10, label=f"Best Val Loss (Epoch {best_epoch})")
        plt.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5)
    else:
        plt.plot(len(epochs), loss[-1], "go", markersize=10, label=f"Final Loss (Epoch {len(epochs)})")

    plt.title(f"Training History: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/LossCurve_{model_name}.png")
    plt.close()


def calculate_metrics(analysis_df, model_name, train_dataset, model_type,
                      batch_size, lr, window_size, phy_loss):
    results = []
    if analysis_df.empty:
        return results

    def get_row(data, regime_label):
        if len(data) == 0:
            return None
        mae = mean_absolute_error(data["Observed"], data["Predicted"])
        rmse = np.sqrt(mean_squared_error(data["Observed"], data["Predicted"]))
        r2 = r2_score(data["Observed"], data["Predicted"]) if len(data) > 1 else float("nan")
        return {
            "Train Dataset":    train_dataset,
            "Model Type":       model_type,
            "Batch Size":       batch_size,
            "Learning Rate":    lr,
            "Window Size":      window_size,
            "Val Physics Loss": phy_loss,
            "Regime":           regime_label,
            "MAE (kW)":         mae,
            "RMSE (kW)":        rmse,
            "R2":               r2,
            "Count":            len(data),
        }

    results.append(get_row(analysis_df, "Overall"))
    for regime, group in analysis_df.groupby("Regime"):
        row = get_row(group, regime)
        if row:
            results.append(row)

    return results


class AdaptivePINN(tf.keras.Model):
    def __init__(self, base_model, use_physics=True, alpha=0.1,
                 wind_idx=6, v_min=0.0, v_max=30.0, p_min=-150.0, p_max=7000.0, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.use_physics = use_physics
        self.alpha = alpha
        self.wind_idx = wind_idx
        self.v_min = tf.constant(v_min, dtype=tf.float32)
        self.v_max = tf.constant(v_max, dtype=tf.float32)
        self.p_min = tf.constant(p_min, dtype=tf.float32)
        self.p_max = tf.constant(p_max, dtype=tf.float32)
        self.lambda_phy = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.lambda_tracker = tf.keras.metrics.Mean(name="lambda")
        self.phy_loss_tracker = tf.keras.metrics.Mean(name="phy_loss")
        self.val_phy_loss_tracker = tf.keras.metrics.Mean(name="val_phy_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.val_loss_tracker,
            self.lambda_tracker, self.phy_loss_tracker, self.val_phy_loss_tracker,
        ]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def calculate_physics_loss(self, x, y_pred):
        v_norm = x[:, -1, self.wind_idx]
        v = v_norm * (self.v_max - self.v_min) + self.v_min
        P = tf.squeeze(y_pred * (self.p_max - self.p_min) + self.p_min)

        v_seq = x[:, :, self.wind_idx] * (self.v_max - self.v_min) + self.v_min
        steady_state_mask = tf.cast(tf.abs(v_seq[:, -1] - v_seq[:, -2]) < 1.5, tf.float32)

        L_cutin = tf.reduce_mean(tf.cast(v < 3.0, tf.float32) * tf.nn.relu(P))
        L_cutout = tf.reduce_mean(tf.cast(v > 25.0, tf.float32) * tf.nn.relu(P))
        L_powerlimits = tf.reduce_mean(tf.nn.relu(-150.0 - P) + tf.nn.relu(P - 7000.0))

        mask_betz = tf.cast((v >= 3.0) & (v <= 11.5), tf.float32)
        P_theo = 7000.0 * tf.pow(v / 11.5, 3.0)
        L_betz = tf.reduce_mean(mask_betz * steady_state_mask * tf.nn.relu(P - P_theo))

        return (L_cutin + L_cutout + L_powerlimits + L_betz) / (self.p_max - self.p_min)

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.base_model(x, training=True)
            mse_loss = tf.reduce_mean(tf.square(y - y_pred))

            if self.use_physics:
                phy_loss = self.calculate_physics_loss(x, y_pred)
                total_loss = mse_loss + self.lambda_phy * phy_loss
            else:
                phy_loss = tf.constant(0.0, dtype=tf.float32)
                total_loss = mse_loss

        if self.use_physics and phy_loss > 0.0:
            last_layer_weights = self.base_model.trainable_variables[-2]
            grad_mse = tape.gradient(mse_loss, last_layer_weights)
            grad_phy = tape.gradient(phy_loss, last_layer_weights)

            lambda_hat = tf.math.divide_no_nan(
                tf.reduce_max(tf.abs(grad_mse)),
                tf.reduce_mean(tf.abs(grad_phy)) + 1e-8,
            )
            lambda_hat = tf.clip_by_value(lambda_hat, 0.0, 50.0)
            self.lambda_phy.assign((1.0 - self.alpha) * self.lambda_phy + self.alpha * lambda_hat)

        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        del tape

        self.loss_tracker.update_state(total_loss)
        self.phy_loss_tracker.update_state(phy_loss)

        res = {"loss": self.loss_tracker.result(), "phy_loss": self.phy_loss_tracker.result()}
        if self.use_physics:
            self.lambda_tracker.update_state(self.lambda_phy)
            res["lambda"] = self.lambda_tracker.result()

        return res

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self.base_model(x, training=False)
        mse_loss = tf.reduce_mean(tf.square(y - y_pred))

        if self.use_physics:
            phy_loss = self.calculate_physics_loss(x, y_pred)
            total_loss = mse_loss + self.lambda_phy * phy_loss
        else:
            phy_loss = tf.constant(0.0, dtype=tf.float32)
            total_loss = mse_loss

        self.val_loss_tracker.update_state(total_loss)
        self.val_phy_loss_tracker.update_state(phy_loss)

        return {
            "loss": self.val_loss_tracker.result(),
            "val_phy_loss": self.val_phy_loss_tracker.result(),
        }


class WindPowerEstimator:
    def __init__(self, use_physics=True):
        self.use_physics = use_physics
        self.best_model = None

    def build_model(self, window_size, input_dim, lr, u1=128, u2=64):
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)

        base_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window_size, input_dim)),
            tf.keras.layers.LSTM(u1, return_sequences=True),
            tf.keras.layers.LSTM(u2, return_sequences=False),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ])

        model = AdaptivePINN(base_model=base_model, use_physics=self.use_physics, wind_idx=WIND_FEATURE_IDX)
        model.build(input_shape=(None, window_size, input_dim))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def train(self, train_df):
        X, y, _, _ = create_sequences(train_df, WINDOW_SIZE)
        y = y.reshape(-1, 1)

        self.best_model = self.build_model(WINDOW_SIZE, X.shape[2], LEARNING_RATE)
        history = self.best_model.fit(X, y, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=True)
        return history


def main():
    create_directories()
    final_report = []

    for train_file, (scaler_file, test_file) in DATA_MAPPINGS.items():
        if not os.path.exists(train_file):
            print(f"Skipping: {train_file} not found")
            continue

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        scalers = load_scalers(scaler_file)
        dataset_name = train_file.replace(".csv", "")

        for model_type, use_physics in [("Standard_LSTM", False), ("PINN_LSTM", True)]:
            full_model_name = f"{dataset_name}_{model_type}"
            print(f"\nTraining: {full_model_name}")

            estimator = WindPowerEstimator(use_physics=use_physics)
            history = estimator.train(train_df)

            model_path = f"{MODEL_DIR}/{full_model_name}.keras"
            estimator.best_model.base_model.save(model_path)
            print(f"  Saved -> {model_path}")

            plot_loss_curve(history, full_model_name)

            analysis_df = get_predictions_df(estimator.best_model, test_df, scalers, WINDOW_SIZE)
            plot_scatter(analysis_df, full_model_name)
            plot_timeseries_examples(analysis_df, full_model_name, num_examples=3)

            metrics = calculate_metrics(
                analysis_df, full_model_name, dataset_name,
                model_type, BATCH_SIZE, LEARNING_RATE, WINDOW_SIZE, "N/A",
            )
            final_report.extend(metrics)

    cols = [
        "Train Dataset", "Model Type", "Batch Size", "Learning Rate",
        "Window Size", "Val Physics Loss", "Regime", "MAE (kW)", "RMSE (kW)", "R2", "Count",
    ]
    all_res = pd.DataFrame(final_report)[cols]
    all_res.to_csv("final_performance_metrics.csv", index=False)
    print("\n" + "=" * 60)
    print(all_res.to_string(index=False))
    print("\nDone. Results -> final_performance_metrics.csv")


if __name__ == "__main__":
    main()
