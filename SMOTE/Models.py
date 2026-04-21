import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from scipy.ndimage import label


class BaseTimeSeriesSMOTE:
    """
    Base class for time-series SMOTE generation.
    """

    def __init__(
        self,
        n_synthetic: int,
        k_neighbours: int,
        signal_features: List[str],
        random_state: int = 42,
    ):
        self.n = n_synthetic
        self.k = k_neighbours
        self.feats = signal_features
        self.rng = np.random.RandomState(random_state)

    def extract_metadata(self, df_class: pd.DataFrame) -> Tuple[list, np.ndarray]:
        """
        Converts time-series events into vectors for KNN.
        """
        events = []
        meta = []

        df_sorted = df_class.sort_index(kind="stable")

        for eid, subset in df_sorted.groupby("EventID", sort=True):
            if len(subset) < 2:
                continue

            vec = np.concatenate(
                (
                    [len(subset), subset["WindSpeed_Hub"].max()],
                    subset[self.feats].mean().values,
                )
            )
            events.append(subset)
            meta.append(vec)

        return events, np.array(meta)

    def resample_event(self, source: pd.DataFrame, target_len: int) -> np.ndarray:
        """
        Linearly interpolates a source event to match a target temporal length.
        """
        if len(source) == target_len:
            return source[self.feats].values

        x_old = np.linspace(0, 1, len(source))
        x_new = np.linspace(0, 1, target_len)
        vals = source[self.feats].values
        new_vals = np.zeros((target_len, vals.shape[1]))

        for i in range(vals.shape[1]):
            f = interp1d(x_old, vals[:, i], kind="linear", fill_value="extrapolate")
            new_vals[:, i] = f(x_new)

        return new_vals


class IntervalSMOTE(BaseTimeSeriesSMOTE):
    """
    Generates synthetic time-series data using standard interval interpolation.
    """

    def generate(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        syn_data = []
        classes = sorted([c for c in df[target_col].unique() if c != "Normal"])

        for cls in classes:
            df_cls = df[df[target_col] == cls]
            events, meta = self.extract_metadata(df_cls)

            if len(events) < self.k:
                continue

            nbrs = NearestNeighbors(n_neighbors=self.k).fit(meta)

            for i in range(self.n):
                valid_sample = False
                max_retries = 20
                attempts = 0

                while not valid_sample and attempts < max_retries:
                    attempts += 1

                    idx_a = self.rng.randint(len(meta))
                    _, indices = nbrs.kneighbors([meta[idx_a]])
                    idx_b = indices[0][self.rng.randint(1, len(indices[0]))]

                    ev_a, ev_b = events[idx_a], events[idx_b]

                    sig_a = ev_a[self.feats].values
                    sig_b = self.resample_event(ev_b, len(ev_a))

                    alpha = self.rng.rand()
                    new_sig = sig_a * (1 - alpha) + sig_b * alpha

                    df_new = pd.DataFrame(new_sig, columns=self.feats)

                    if (
                        "WindSpeed_Hub" in df_new.columns
                        and (df_new["WindSpeed_Hub"] < 0.0).any()
                    ):
                        continue

                    df_new["EventID"] = f"Syn_{cls}_{i}"

                    for c in ev_a.columns:
                        if c not in self.feats and c != "EventID":
                            df_new[c] = ev_a[c].iloc[0]

                    syn_data.append(df_new)
                    valid_sample = True

        return (
            pd.concat(syn_data).reset_index(drop=True) if syn_data else pd.DataFrame()
        )


class ExtrapolatingSMOTE(BaseTimeSeriesSMOTE):
    """
    Generates extreme synthetic events by extrapolating beyond target boundaries.
    """

    def generate(
        self,
        df: pd.DataFrame,
        target_col: str,
        threshold: float = 11.0,
        cap: float = 35.0,
    ) -> pd.DataFrame:
        syn_data = []
        classes = sorted([c for c in df[target_col].unique() if c != "Normal"])

        for cls in classes:
            df_cls = df[df[target_col] == cls]
            events, meta = self.extract_metadata(df_cls)

            high_wind_ids = [i for i, m in enumerate(meta) if m[1] > threshold]
            if len(high_wind_ids) < 2:
                continue

            relevant_meta = meta[high_wind_ids]
            nbrs = NearestNeighbors(n_neighbors=min(self.k, len(relevant_meta))).fit(
                relevant_meta
            )

            for i in range(self.n):
                idx_local = self.rng.randint(len(relevant_meta))
                idx_global_base = high_wind_ids[idx_local]

                _, indices = nbrs.kneighbors([relevant_meta[idx_local]])
                idx_neighbour_local = indices[0][self.rng.choice(len(indices[0]))]
                idx_global_target = high_wind_ids[idx_neighbour_local]

                base = events[idx_global_base]
                target = events[idx_global_target]

                if base["WindSpeed_Hub"].max() > target["WindSpeed_Hub"].max():
                    base, target = target, base

                sig_target = self.resample_event(target, len(base))

                intensity = self.rng.uniform(0.5, 1.5)
                delta = sig_target - self.resample_event(base, len(base))

                new_sig = sig_target + (delta * intensity)

                df_new = pd.DataFrame(new_sig, columns=self.feats)
                df_new["WindSpeed_Hub"] = df_new["WindSpeed_Hub"].clip(upper=cap)
                df_new["ActivePower"] = df_new["ActivePower"].clip(lower=0.0)

                df_new["EventID"] = f"Extrap_{cls}_{i}"
                for c in base.columns:
                    if c not in self.feats and c != "EventID":
                        df_new[c] = base[c].iloc[0]

                syn_data.append(df_new)

        return (
            pd.concat(syn_data).reset_index(drop=True) if syn_data else pd.DataFrame()
        )


class PhysicsAwareSMOTE(ExtrapolatingSMOTE):
    """
    Extrapolating SMOTE constrained by physical wind turbine operating parameters.
    Treats the theoretical power curve as a strict upper bound.
    """

    def __init__(
        self,
        n_synthetic: int,
        k_neighbours: int,
        signal_features: List[str],
        rated_power: float = 7000.0,
        cut_in: float = 3.0,
        cut_out: float = 25.0,
        rated_speed: float = 11.5,
        random_state: int = 42,
    ):
        super().__init__(
            n_synthetic, k_neighbours, signal_features, random_state=random_state
        )
        self.rated_power = rated_power
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.rated_speed = rated_speed

    def generate(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df_syn = super().generate(df, target_col)

        if df_syn.empty:
            return df_syn

        valid_fragments = []
        for eid, event in df_syn.groupby("EventID", sort=True):
            fragments = self.extract_viable_segments(event)
            valid_fragments.extend(fragments)

        return (
            pd.concat(valid_fragments).reset_index(drop=True)
            if valid_fragments
            else pd.DataFrame(columns=df_syn.columns)
        )

    def extract_viable_segments(self, event: pd.DataFrame) -> List[pd.DataFrame]:
        v = event["WindSpeed_Hub"].values
        p = event["ActivePower"].values

        mask = p <= self.rated_power

        mask_stationary = (v < self.cut_in) | (v > self.cut_out)
        mask &= ~(mask_stationary & (p > 10.0))

        mask_ramp = (v >= self.cut_in) & (v < self.rated_speed)
        k = self.rated_power / (self.rated_speed**3)
        theoretical_max = k * (v**3) + 500
        mask &= ~(mask_ramp & (p > theoretical_max))

        labeled_array, num_features = label(mask)

        fragments = []
        for i in range(1, num_features + 1):
            frag = event.iloc[labeled_array == i].copy()

            if len(frag) >= 3:
                frag["EventID"] = f"{frag['EventID'].iloc[0]}_part{i}"
                fragments.append(frag)

        return fragments


class CombinedSMOTE:
    """
    Hybrid approach blending Interval and Physics-Aware SMOTE based on wind regime.
    """

    def __init__(
        self,
        n_synthetic: int,
        k_neighbours: int,
        signal_features: List[str],
        switch_threshold: float = 22.0,
        random_state: int = 42,
    ):
        self.interval_model = IntervalSMOTE(
            1000, k_neighbours, signal_features, random_state=random_state
        )
        self.physics_model = PhysicsAwareSMOTE(
            n_synthetic, k_neighbours, signal_features, random_state=random_state + 1
        )
        self.threshold = switch_threshold

    def generate(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        syn_results = []

        syn_interval_all = self.interval_model.generate(df, target_col)
        if not syn_interval_all.empty:
            max_winds = syn_interval_all.groupby("EventID")["WindSpeed_Hub"].max()
            min_winds = syn_interval_all.groupby("EventID")["WindSpeed_Hub"].min()

            valid_ids = max_winds[
                (max_winds <= self.threshold) & (min_winds >= 0.0)
            ].index
            syn_low = syn_interval_all[
                syn_interval_all["EventID"].isin(valid_ids)
            ].copy()

            if not syn_low.empty:
                syn_low["EventID"] = syn_low["EventID"].apply(lambda x: f"LowWind_{x}")
                syn_results.append(syn_low)

        syn_physics_all = self.physics_model.generate(df, target_col)
        if not syn_physics_all.empty:
            max_winds_physics = syn_physics_all.groupby("EventID")[
                "WindSpeed_Hub"
            ].max()

            valid_physics_ids = max_winds_physics[
                max_winds_physics > self.threshold
            ].index
            syn_high = syn_physics_all[
                syn_physics_all["EventID"].isin(valid_physics_ids)
            ].copy()

            if not syn_high.empty:
                syn_high["EventID"] = syn_high["EventID"].apply(
                    lambda x: f"HighWind_{x}"
                )
                syn_results.append(syn_high)

        return (
            pd.concat(syn_results).reset_index(drop=True)
            if syn_results
            else pd.DataFrame()
        )
