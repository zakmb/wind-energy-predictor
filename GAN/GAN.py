import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def set_determinism(seed=42):
    """
    Locks down all sources of randomness across Python, NumPy, PyTorch, and cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


set_determinism(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_FILE = "../Data/Processed/train_balanced.csv"
SCALING_FILE = "../Data/Processed/scaling_params.csv"

OUTPUT_DIR = "../Data/GAN"
PLOT_DIR = "../Data/GAN/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SEQ_LEN = 24
BATCH_SIZE = 64
NOISE_DIM = 100
EPOCHS = 100
LR = 0.0002
WARMUP_EPOCHS = 10
DIVERSITY_WEIGHT = 0.005

COND_WIND_COL = ["WindSpeed_Hub"]
TARGET_OUTPUT_COLS = [
    "ActivePower",
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
]

RATED_POWER_KW = 7000.0
CUT_IN_SPEED_MPS = 3.0
RATED_SPEED_MPS = 11.5
CUT_OUT_SPEED_MPS = 25.0


def load_scaling_limits(filepath):
    df = pd.read_csv(filepath, index_col=0)

    w_min = df.loc["WindSpeed_Hub", "min"]
    w_max = df.loc["WindSpeed_Hub", "max"]
    p_min = df.loc["ActivePower", "min"]
    p_max = df.loc["ActivePower", "max"]

    all_limits = {}
    for col in df.index:
        all_limits[col] = {"min": df.loc[col, "min"], "max": df.loc[col, "max"]}

    return {
        "cut_out": (CUT_OUT_SPEED_MPS - w_min) / (w_max - w_min),
        "cut_in": (CUT_IN_SPEED_MPS - w_min) / (w_max - w_min),
        "rated_v": (RATED_SPEED_MPS - w_min) / (w_max - w_min),
        "rated_p": (RATED_POWER_KW - p_min) / (p_max - p_min),
        "w_min": w_min,
        "w_max": w_max,
        "p_min": p_min,
        "p_max": p_max,
        "all_limits": all_limits,
    }


class WindDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        df = pd.read_csv(csv_file)
        self.seq_len = seq_len

        self.cond_wind = df[COND_WIND_COL].values.astype(np.float32)
        self.targets = df[TARGET_OUTPUT_COLS].values.astype(np.float32)

        dummies = pd.get_dummies(df["_EventLabel"])

        dummies = dummies.reindex(sorted(dummies.columns), axis=1)

        self.event_classes = dummies.columns.tolist()
        self.cond_event = dummies.values.astype(np.float32)

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        w = torch.tensor(self.cond_wind[idx : idx + self.seq_len])
        e = torch.tensor(self.cond_event[idx : idx + self.seq_len])
        t = torch.tensor(self.targets[idx : idx + self.seq_len])
        return torch.cat([w, e], dim=-1), t


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(
            noise_dim + cond_dim, hidden_dim, num_layers=2, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, conditions):
        combined = torch.cat([z, conditions], dim=2)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out)


class Discriminator(nn.Module):
    def __init__(self, target_dim, cond_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            target_dim + cond_dim, hidden_dim, num_layers=2, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x, conditions):
        combined = torch.cat([x, conditions], dim=2)
        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out)


class PhysicsLoss:
    def __init__(self, limits):
        self.limits = limits
        self.idx_power = 0
        self.idx_era5 = 4

    def compute(self, gen_batch, target_wind_speed):
        gen_power = gen_batch[:, :, self.idx_power]
        gen_era5 = gen_batch[:, :, self.idx_era5]
        target_w_flat = target_wind_speed.squeeze()

        loss = 0.0
        loss += torch.mean((target_w_flat - gen_era5) ** 2) * 1.0

        mask_cut_in = (target_w_flat < self.limits["cut_in"]).float()
        loss += torch.mean(mask_cut_in * (gen_power**2)) * 5.0

        safe_wind = torch.clamp(target_w_flat, min=0.01)
        safe_rated = max(self.limits["rated_v"], 0.01)

        ratio = safe_wind / safe_rated
        theo_limit = torch.clamp(ratio**3, max=1.0) * self.limits["rated_p"]

        loss += torch.mean(torch.relu(gen_power - theo_limit)) * 5.0

        mask_cut_out = (target_w_flat > self.limits["cut_out"]).float()
        loss += torch.mean(mask_cut_out * (gen_power**2)) * 5.0

        return loss


def diversity_loss(batch):
    flat = batch.view(batch.size(0), -1)
    return -torch.mean(torch.cdist(flat, flat, p=2))


def train_model():
    limits = load_scaling_limits(SCALING_FILE)

    dataset = WindDataset(TRAIN_FILE, SEQ_LEN)

    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g
    )

    cond_dim = 1 + len(dataset.event_classes)

    G = Generator(NOISE_DIM, cond_dim, len(TARGET_OUTPUT_COLS)).to(DEVICE)
    D = Discriminator(len(TARGET_OUTPUT_COLS), cond_dim).to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    physics = PhysicsLoss(limits)

    for epoch in range(EPOCHS):
        phys_alpha = (
            0.0
            if epoch < WARMUP_EPOCHS
            else min(1.0, (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS))
        )
        total_d, total_g = 0, 0

        p_loss_val = 0.0
        div_loss_val = 0.0

        for i, (real_conds, real_targ) in enumerate(dataloader):
            real_conds, real_targ = real_conds.to(DEVICE), real_targ.to(DEVICE)
            bs = real_conds.size(0)
            real_wind = real_conds[:, :, 0:1]

            opt_D.zero_grad()
            real_labels = torch.full((bs, SEQ_LEN, 1), 0.9).to(DEVICE)
            d_loss_real = criterion(D(real_targ, real_conds), real_labels)

            z = torch.randn(bs, SEQ_LEN, NOISE_DIM).to(DEVICE)
            fake_targ = G(z, real_conds)

            d_loss_fake = criterion(
                D(fake_targ.detach(), real_conds),
                torch.zeros(bs, SEQ_LEN, 1).to(DEVICE),
            )

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            adv_loss = criterion(
                D(fake_targ, real_conds), torch.ones(bs, SEQ_LEN, 1).to(DEVICE)
            )

            real_mean = torch.mean(real_targ, dim=0)
            fake_mean = torch.mean(fake_targ, dim=0)
            stat_loss = torch.mean((real_mean - fake_mean) ** 2)

            p_loss = physics.compute(fake_targ, real_wind)
            div_loss = diversity_loss(fake_targ)

            g_loss = (
                adv_loss
                + (2.0 * stat_loss)
                + (phys_alpha * p_loss)
                + (DIVERSITY_WEIGHT * div_loss)
            )

            g_loss.backward()
            opt_G.step()

            total_d += d_loss.item()
            total_g += g_loss.item()
        
            p_loss_val = p_loss.item()
            div_loss_val = div_loss.item()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            pass

    return G, limits, dataset.event_classes


def validate_and_save(G, limits, event_classes):
    G.eval()

    norm_cut_in = limits["cut_in"]
    norm_cut_out = limits["cut_out"]
    norm_rated_v = limits["rated_v"]
    norm_rated_p = limits["rated_p"]

    def is_physically_valid(row):
        v_norm, p_norm = row["WindSpeed_Hub"], row["ActivePower"]
        if v_norm < norm_cut_in or v_norm > norm_cut_out:
            return p_norm <= 0.05
        if p_norm > (norm_rated_p + 0.05):
            return False
        if norm_rated_v > v_norm > norm_cut_in:
            max_theoretical = ((v_norm / norm_rated_v) ** 3) * norm_rated_p
            if p_norm > (max_theoretical * 1.2 + 0.05):
                return False
        return True

    all_generated_df = []
    global_event_id = 0

    event_wind_ranges_phys = {
        "Lull": (0.1, 4.0),
        "Ramp": (4.0, 20.0),
        "Normal": (4.0, 25.0),
        "CutOut": (24.0, 30.0),
    }

    w_min, w_max = limits["w_min"], limits["w_max"]
    all_scaling = limits["all_limits"]

    PTS_PER_EVENT = 8000
    batch_size = PTS_PER_EVENT // SEQ_LEN

    val_rng = np.random.RandomState(42)

    for class_index, event_name in enumerate(event_classes):
        wind_min, wind_max = event_wind_ranges_phys.get(event_name, (0.1, 25.0))

        base_wind = np.linspace(wind_min, wind_max, PTS_PER_EVENT, dtype=np.float32)
        turbulence = val_rng.normal(0, 0.5, PTS_PER_EVENT).astype(np.float32)
        target_wind_mps = np.clip(base_wind + turbulence, 0.01, 30.0)

        target_wind_norm = (target_wind_mps - w_min) / (w_max - w_min)
        target_wind_norm_batch = target_wind_norm[: batch_size * SEQ_LEN].reshape(
            batch_size, SEQ_LEN, 1
        )

        cond_one_hot = np.zeros(
            (batch_size, SEQ_LEN, len(event_classes)), dtype=np.float32
        )
        cond_one_hot[:, :, class_index] = 1.0

        combined_cond = np.concatenate([target_wind_norm_batch, cond_one_hot], axis=2)
        cond_tensor = torch.tensor(combined_cond, dtype=torch.float32).to(DEVICE)

        z = torch.randn(batch_size, SEQ_LEN, NOISE_DIM).to(DEVICE)

        with torch.no_grad():
            fake_out = G(z, cond_tensor).cpu().numpy()

        for seq_i in range(batch_size):
            global_event_id += 1
            seq_out = fake_out[seq_i]
            seq_wind_norm = target_wind_norm[seq_i * SEQ_LEN : (seq_i + 1) * SEQ_LEN]

            df_event = pd.DataFrame(seq_out, columns=TARGET_OUTPUT_COLS)
            df_event["WindSpeed_Hub"] = seq_wind_norm

            valid_mask = df_event.apply(is_physically_valid, axis=1)
            df_filtered = df_event[valid_mask].copy()

            for col in df_filtered.columns:
                if col in all_scaling:
                    c_min = all_scaling[col]["min"]
                    c_max = all_scaling[col]["max"]
                    df_filtered[col] = df_filtered[col] * (c_max - c_min) + c_min

            df_filtered["Generated_Event"] = event_name
            df_filtered["Generated_EventID"] = global_event_id

            all_generated_df.append(df_filtered)

    final_combined_df = pd.concat(all_generated_df, ignore_index=True)

    cols = ["Generated_EventID", "Generated_Event", "ActivePower", "WindSpeed_Hub"] + [
        c for c in TARGET_OUTPUT_COLS if c != "ActivePower"
    ]
    final_combined_df = final_combined_df[cols]

    output_path = os.path.join(OUTPUT_DIR, "synthetic_events_unscaled.csv")
    final_combined_df.to_csv(output_path, index=False)

    plt.figure(figsize=(12, 7))
    colors = {"CutOut": "red", "Lull": "blue", "Ramp": "orange", "Normal": "green"}

    for event_name in event_classes:
        subset = final_combined_df[final_combined_df["Generated_Event"] == event_name]
        c = colors.get(event_name, "purple")
        plt.scatter(
            subset["WindSpeed_Hub"],
            subset["ActivePower"],
            alpha=0.4,
            s=5,
            label=event_name,
            c=c,
        )

    plt.axvline(
        x=RATED_SPEED_MPS, color="black", linestyle="--", alpha=0.5, label="Rated Speed"
    )
    plt.axvline(
        x=CUT_OUT_SPEED_MPS, color="red", linestyle="--", alpha=0.5, label="Cut Out"
    )

    plt.title(f"GAN Output (Physical Units)")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Active Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "gan_curve_physical.png"))
    plt.close()


if __name__ == "__main__":
    model, limits, classes = train_model()
    validate_and_save(model, limits, classes)
