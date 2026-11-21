"""
AnEn with optimised predictor weights and number of analogs k (appendix).

Predictors as in anen_metrics.py: wind speed forecast x and lagged observation (window t±1).
Only the ratio of the predictor weights changes the analog ranking, so the grid uses
alpha = W_obs / (W_fc + W_obs); alpha = 0.5 equals the AnEn of Table 2 (W = 1 for both predictors).

- Tuning: training targets 2024-01-01 ... start of the validation period, archive 2022-01-01 ... 2023-12-31.
- Evaluation: validation samples and static archive identical to anen_metrics.py (Table 2).
- Dissimilarity replicates a2e.model.AnEn: sum_f W_f/sigma_f * sqrt(sum_window diff^2), sigma from the archive rows.
- Scores per (location, alpha, k, split): Bias, RMSE of the ensemble mean, CRPS, SCRPS (as in Table 2).

Run from the repository root (about 10 min with 4 processes):
    python -m experiments.evaluation.AnEn.anen_optimization
Output: experiments/evaluation/AnEn/anen_optimization_grid.csv
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, pandas as pd
from multiprocessing import Pool

from ...project_paths import WIND_DIR, EVALUATION_DIR
OUT = EVALUATION_DIR / "AnEn" / "anen_optimization_grid.csv"
ALPHAS = np.round(np.linspace(0, 1, 11), 2)
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
KMAX = max(KS)

obs_all = pd.read_csv(WIND_DIR / "observations_wind_speed.csv", index_col=0).to_numpy(np.float64)
fc_all = pd.read_csv(WIND_DIR / "forecasts_wind_speed.csv", index_col=0).to_numpy(np.float64)
T = obs_all.shape[0]
test_size = int((T - 8760 * 2) * 0.3) - 2
IDX = 3  # foresight + time_to_target


def windows(a):  # (n, F) -> (n-2, F, 3), window i = rows i..i+2, centre row i+1
    return np.lib.stride_tricks.sliding_window_view(a, 3, axis=0)


def dist(tw, hw, f):
    """sqrt(sum_window diff^2) for feature f; tw (nt, F, 3), hw (nh, F, 3) -> (nt, nh)."""
    a, b = tw[:, f, :], hw[:, f, :]
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def members_grid(tw, hw, yh_c, sig):
    """Sorted top-KMAX members for each alpha: (n_alpha, nt, KMAX)."""
    out = np.empty((len(ALPHAS), tw.shape[0], KMAX), np.float32)
    for s in range(0, tw.shape[0], 500):
        d_fc = dist(tw[s:s + 500], hw, 0) / sig[0]
        d_ob = dist(tw[s:s + 500], hw, 1) / sig[1]
        for ai, a in enumerate(ALPHAS):
            D = (1 - a) * d_fc + a * d_ob
            idx = np.argpartition(D, KMAX, axis=1)[:, :KMAX]
            order = np.argsort(np.take_along_axis(D, idx, 1), axis=1)
            out[ai, s:s + 500] = yh_c[np.take_along_axis(idx, order, 1)]
    return out


def scores(m, y):
    """m (n, k) equally weighted members, y (n,) -> Bias, RMSE (ensemble mean), CRPS, SCRPS (as in Table 2)."""
    k = m.shape[1]
    ms = np.sort(m, 1).astype(np.float64)
    i = np.arange(1, k + 1)
    t1 = (2.0 / k ** 2) * (ms * (2 * i - k - 1)).sum(1)          # E|X-X'|
    t2 = np.abs(ms - y[:, None]).mean(1)                           # E|X-y|
    e = ms.mean(1) - y
    t1g = t1 + 1e-8
    return dict(Bias=e.mean(), RMSE=np.sqrt((e ** 2).mean()), CRPS=(t2 - 0.5 * t1).mean(),
                SCRPS=(t2 / t1g + 0.5 * np.log(t1g)).mean())


def run(loc):
    o, x = obs_all[:, loc], fc_all[:, loc]
    feat = np.stack([x[IDX:], o[:-IDX]], 1)       # row i: [x_{i+3}, y_i]
    tgt = o[IDX:]                                  # row i: y_{i+3}
    res = []
    # --- validation (identical to anen_metrics.py, a2e.io.anen_inputs): archive rows [:-(test_size+1)], i.e. up to
    # the window whose value lies at the issue time of the first query; query windows rows [-(test_size+2):]
    xh, yh = feat[:-test_size - 1], tgt[:-test_size - 1]
    sig = xh.std(0)
    tw, hw = windows(feat[-(test_size + 2):]), windows(xh)
    y_val = tgt[-test_size - 1:-1]
    mv = members_grid(tw, hw, yh[1:-1], sig)
    # --- tuning: targets = 2024 training targets, archive = windows with centre target before 2024-01-01 minus 2 h
    c_tr = np.arange(8760 * 2 - IDX, T - IDX - test_size - 1)      # centre rows of training targets (y = tgt[c])
    n_arch = 8760 * 2 - IDX - 2                                     # archive rows 0..n_arch-1
    xh2 = feat[:n_arch]
    sig2 = xh2.std(0)
    tw2 = windows(feat[c_tr[0] - 1:c_tr[-1] + 2])
    mt = members_grid(tw2, windows(xh2), tgt[:n_arch][1:-1], sig2)
    y_tr = tgt[c_tr]
    for ai, a in enumerate(ALPHAS):
        for k in KS:
            res.append(dict(location=loc, alpha=a, k=k, split="train", **scores(mt[ai, :, :k], y_tr)))
            res.append(dict(location=loc, alpha=a, k=k, split="val", **scores(mv[ai, :, :k], y_val)))
    return res


if __name__ == "__main__":
    with Pool(4) as p:
        rows = [r for rs in p.imap_unordered(run, range(obs_all.shape[1])) for r in rs]
    pd.DataFrame(rows).sort_values(["location", "split", "alpha", "k"]).to_csv(OUT, index=False)
    print("DONE")
