"""
Baselines for the review (M2): raw NWP forecast, persistence and persistence ensemble (PeEn).

Samples are identical to the A2E/AnEn evaluation (anen_metrics.py, a2e_metrics.py):
validation targets y_t at the last `test_size` time steps before the final `foresight` step.
Issue time is t-2 (most recent known observation y_{t-2}).

- NWP:          point forecast x_t
- Persistence:  point forecast y_{t-2}
- PeEn(k):      equally weighted ensemble of the k most recent observations y_{t-2}, ..., y_{t-k-1}
                k is selected on the training targets (2024-01-01 until the start of the validation period)

RMSE follows the paper equation: RMSE of the ensemble mean over all samples of a location.
Bias, CRPS and SCRPS use the same functions as Table 2 (a2e.metrics.computation).
SCRPS is not defined for point forecasts (E|X - X'| = 0) and is reported as NaN.

Run from the repository root: python -m experiments.evaluation.baselines.baselines_metrics
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import EVALUATION_DIR

OUT_DIR = EVALUATION_DIR / "baselines"
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48]
FORESIGHT, TIME_TO_TARGET = 1, 2
TRAIN_START = 8760 * 2  # first training target: 2024-01-01 00:00


def scores(ens: np.ndarray, y: np.ndarray, point: bool) -> dict:
    x = tf.constant(ens, tf.float32)
    w = tf.ones_like(x) / x.shape[1]
    yt = tf.constant(y, tf.float32)
    bias_s = a2e.metrics.computation.compute_bias(x, w, yt).numpy().ravel()
    crps_s = -a2e.metrics.computation.compute_crps(x, w, yt).numpy()
    out = dict(Bias=bias_s.mean(), RMSE=np.sqrt(np.mean(bias_s ** 2)), CRPS=crps_s.mean(), SCRPS=np.nan)
    if not point:
        out["SCRPS"] = (-a2e.metrics.computation.compute_scrps(x, w, yt)).numpy().mean()
    return out


def lagged(obs: np.ndarray, tau: np.ndarray, k: int) -> np.ndarray:
    """Members y_{tau-2-j}, j = 0..k-1, shape (n_samples, k)."""
    return np.stack([obs[tau - TIME_TO_TARGET - j] for j in range(k)], axis=1)


loader = WindDataLoader()
observations, forecasts = loader.get_all_data(obs_components=["total"], fcst_components=["total"])
obs_all, fcst_all = observations[..., 0].astype(np.float32), forecasts[..., 0].astype(np.float32)
T, n_loc = obs_all.shape

# Validation target indices in the original time axis (identical to anen_metrics.py / a2e_metrics.py)
test_size = int((T - 8760 * 2) * 0.3) - 2
tau_val = np.arange(T - test_size - FORESIGHT, T - FORESIGHT)
tau_train = np.arange(TRAIN_START, tau_val[0])

# Alignment check against the shifted arrays used in the evaluation scripts
idx = FORESIGHT + TIME_TO_TARGET
shifted_obs = obs_all[idx:]
assert np.array_equal(shifted_obs[-test_size - FORESIGHT:-FORESIGHT], obs_all[tau_val])

# 1) Select k for PeEn on the training targets (mean CRPS over all locations)
k_rows = []
for k in K_GRID:
    crps = [scores(lagged(obs_all[:, l], tau_train, k), obs_all[tau_train, l], point=(k == 1))["CRPS"] for l in range(n_loc)]
    k_rows.append(dict(k=k, CRPS_train=np.mean(crps)))
    print(f"PeEn k={k:3d}  CRPS(train) = {np.mean(crps):.4f}")
k_sel = pd.DataFrame(k_rows).set_index("k")
k_best = int(k_sel["CRPS_train"].idxmin())
print(f"Selected k = {k_best}")

# 2) Evaluate baselines on the validation targets
rows = []
for l in range(n_loc):
    y = obs_all[tau_val, l]
    for name, ens, point in [("NWP", fcst_all[tau_val, l][:, None], True),
                             ("Persistence", obs_all[tau_val - TIME_TO_TARGET, l][:, None], True),
                             (f"PeEn", lagged(obs_all[:, l], tau_val, k_best), False)]:
        rows.append(dict(location=f"Location_{l}", model=name, **scores(ens, y, point)))
    # k-sensitivity on validation (reported for transparency, not used for selection)
    for k in K_GRID:
        r = scores(lagged(obs_all[:, l], tau_val, k), y, point=(k == 1))
        k_rows.append(dict(k=k, location=l, CRPS_val=r["CRPS"]))

df = pd.DataFrame(rows).pivot(index="location", columns="model")
df = df.reindex(sorted(df.index, key=lambda s: int(s.split("_")[1])))
OUT_DIR.mkdir(exist_ok=True)
df.to_csv(OUT_DIR / "baselines_metrics.csv")
k_val = pd.DataFrame([r for r in k_rows if "CRPS_val" in r]).groupby("k")["CRPS_val"].mean()
k_sel.join(k_val).to_csv(OUT_DIR / "peen_k_selection.csv")
print(f"\nPeEn k = {k_best}; validation samples per location: {len(tau_val)}")
print(df.mean().unstack("model").round(4).to_string())
