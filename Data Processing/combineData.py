import pandas as pd
import numpy as np

# --- Configuration ---
# Doing 900 for SMOTE, 6000 for GAN to make file sizes small enough
MAX_TOTAL_EVENTS = 900
# MAX_TOTAL_EVENTS = 6000


# --- Input Paths ---
orig_path = "../Data/Processed/train_original.csv"
aug_path = "../Data/SMOTE/Combined/combined_smote.csv"
scaling_path = "../Data/Processed/scaling_params.csv"
test_path = "../Data/Processed/test.csv"

# --- Output Paths: Combined Data ---
output_data_path = "../Data/RF Model Data Files/Train/original_and_smote_data.csv"
output_scaling_path = (
    "../Data/RF Model Data Files/Train/original_and_smote_scaling_params.csv"
)
output_test_path = "../Data/RF Model Data Files/Test/original_and_smote_test_data.csv"

# --- Output Paths: SMOTE-Only Data ---
output_aug_renormalised_path = "../Data/RF Model Data Files/Train/smote_data.csv"
output_aug_scaling_path = "../Data/RF Model Data Files/Train/smote_scaling_params.csv"
output_aug_test_path = "../Data/RF Model Data Files/Test/smote_test_data.csv"

features = [
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


def unnormalise_df(df, params, features):
    df_new = df.copy()
    for col in features:
        if col in params.index:
            min_val = params.loc[col, "min"]
            max_val = params.loc[col, "max"]
            df_new[col] = df[col] * (max_val - min_val) + min_val
    return df_new


def normalise_df(df, params, features):
    df_new = df.copy()
    for col in features:
        if col in params.index:
            min_val = params.loc[col, "min"]
            max_val = params.loc[col, "max"]
            diff = max_val - min_val
            if diff == 0:
                df_new[col] = 0
            else:
                # Clips to strictly enforce the [0, 1] range
                df_new[col] = ((df[col] - min_val) / diff).clip(0, 1)
    return df_new


def get_event_list(df, label_col, id_col):
    events_by_label = {}
    df_sorted = df.sort_values(by=[label_col, id_col], kind="stable")

    for label, group in df_sorted.groupby(label_col, sort=True):
        events_by_label[label] = [g for _, g in group.groupby(id_col, sort=True)]
    return events_by_label


def main():
    rng = np.random.RandomState(42)

    # 1. Load Data
    train_original = pd.read_csv(orig_path).drop(columns=["StartTime"], errors="ignore")
    train_augmented = pd.read_csv(aug_path).drop(columns=["StartTime"], errors="ignore")
    test_data = pd.read_csv(test_path).drop(columns=["StartTime"], errors="ignore")
    scaling_params = pd.read_csv(scaling_path, index_col=0)

    # 2. Unscale all data
    train_orig_unscaled = unnormalise_df(train_original, scaling_params, features)
    train_aug_unscaled = unnormalise_df(train_augmented, scaling_params, features)
    test_unscaled = unnormalise_df(test_data, scaling_params, features)

    # 3. Calculate Capped Target Size
    orig_events = get_event_list(train_orig_unscaled, "_EventLabel", "_EventID")
    aug_events = get_event_list(train_aug_unscaled, "_EventLabel", "_EventID")

    labels = ["Normal", "Lull", "Ramp", "CutOut"]
    max_per_class = MAX_TOTAL_EVENTS // len(labels)  # 1500 per class

    orig_counts = {k: len(v) for k, v in orig_events.items()}
    # Cap target size so we don't blow up the rows
    target_size = min(max(orig_counts.values()), max_per_class)

    final_events_list = []

    for label in labels:
        orig_list = orig_events.get(label, [])
        aug_list = aug_events.get(label, [])

        combined_pool = orig_list + aug_list
        n_available = len(combined_pool)

        current_selection = []

        if n_available >= target_size:
            current_selection.extend(orig_list)
            needed = target_size - len(current_selection)

            if needed > 0:
                selected_indices = rng.choice(len(aug_list), needed, replace=False)
                for idx in selected_indices:
                    current_selection.append(aug_list[idx])
            elif needed < 0:
                current_selection = current_selection[:target_size]

        else:
            current_selection.extend(combined_pool)
            needed = target_size - n_available

            if needed > 0:
                indices = rng.choice(n_available, needed, replace=True)
                for idx in indices:
                    current_selection.append(combined_pool[idx])

        final_events_list.extend(current_selection)

    rng.shuffle(final_events_list)

    processed_frames = []
    cols_to_keep = features + ["_EventLabel"]

    for i, event_df in enumerate(final_events_list):
        df_subset = event_df[cols_to_keep].copy()
        df_subset["_EventID"] = i
        processed_frames.append(df_subset)

    train_balanced_unscaled = pd.concat(processed_frames, ignore_index=True)

    aug_capped_list = []
    for label in labels:
        aug_list = aug_events.get(label, [])
        if len(aug_list) > target_size:
            # Randomly sample down to the target size limit
            indices = rng.choice(len(aug_list), target_size, replace=False)
            aug_capped_list.extend([aug_list[i] for i in indices])
        else:
            aug_capped_list.extend(aug_list)

    rng.shuffle(aug_capped_list)

    aug_processed_frames = []
    for i, event_df in enumerate(aug_capped_list):
        df_subset = event_df[cols_to_keep].copy()
        df_subset["_EventID"] = i
        aug_processed_frames.append(df_subset)

    train_aug_capped = pd.concat(aug_processed_frames, ignore_index=True)

    new_stats = train_balanced_unscaled[features].describe().T[["min", "max"]]
    new_stats.to_csv(output_scaling_path)

    train_balanced_norm = normalise_df(train_balanced_unscaled, new_stats, features)
    train_balanced_norm.to_csv(output_data_path, index=False)

    test_renormalised_combined = normalise_df(test_unscaled, new_stats, features)
    test_renormalised_combined.to_csv(output_test_path, index=False)

    aug_stats = train_aug_capped[features].describe().T[["min", "max"]]
    aug_stats.to_csv(output_aug_scaling_path)

    train_aug_renormalised = normalise_df(train_aug_capped, aug_stats, features)
    train_aug_renormalised.to_csv(output_aug_renormalised_path, index=False)

    test_renormalised_aug = normalise_df(test_unscaled, aug_stats, features)
    test_renormalised_aug.to_csv(output_aug_test_path, index=False)


if __name__ == "__main__":
    main()
