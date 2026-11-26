"""
Significance of the score differences between the models.

Input: the per-sample files of the evaluation scripts (evaluation/<model type>/per_sample/<name>.npz), 3720 hourly
validation samples per location.

For each model pair (A, B) and score (CRPS, SCRPS), with D = score_B - score_A per sample (D < 0: B is better):
    - mean_A, mean_B, mean_D over all locations and samples,
    - paired Wilcoxon signed-rank test over the location means of D, and the number of locations with D < 0,
    - moving block bootstrap over time: blocks of BLOCK_DAYS consecutive days, drawn jointly for all locations
      (keeps the spatial correlation and the autocorrelation within a block); 95 % percentile interval of mean_D and
      the two-sided bootstrap p-value 2 * min(P(D* <= 0), P(D* >= 0)), with P estimated as (count + 1) / (N + 1)
      (smallest value about 2 / N_BOOTSTRAP).
For each model also the 95 % bootstrap interval of its mean score with the same draws.

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.Significance.significance

Results:
    evaluation/Significance/significance_pairs.csv   (A, B, Score, block_days, n_locations, mean_A, mean_B, mean_D,
                                                     ci_low, ci_high, p_bootstrap, p_wilcoxon, n_B_better)
    evaluation/Significance/significance_models.csv  (Model, Score, block_days, n_locations, mean, ci_low, ci_high)
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from ..per_sample import MODELS, load
from ...project_paths import EVALUATION_DIR

SCORES = ["CRPS", "SCRPS"]
BLOCK_DAYS = [1, 5]       # block lengths of the bootstrap in days
N_BOOTSTRAP = 10000
SEED = 42
OUTPUT_DIR = EVALUATION_DIR / "Significance"

# (A, B): Table 2, then the ablations
PAIRS = [
    ("AnEn", "A2E"), ("AnEn", "CA2E"), ("AnEn", "SA2E"),
    ("A2E", "CA2E"), ("A2E", "SA2E"), ("CA2E", "SA2E"),
    ("AnEn", "A2E T3"), ("A2E T3", "A2E"),
    ("AnEn no_obs", "A2E no_obs"), ("A2E no_obs", "A2E"),
    ("A2E euclidean", "A2E"), ("CA2E euclidean", "CA2E"),
]


def day_index(time):
    """Day number (0 ...) of each sample, counted from the first validation day."""
    days = time.astype("datetime64[D]")
    return (days - days.min()).astype(int)


def block_draws(n_days, block_days, rng):
    """N_BOOTSTRAP x n_days matrix of resampled day numbers (moving blocks of block_days consecutive days)."""
    n_blocks = int(np.ceil(n_days / block_days))
    starts = rng.integers(0, n_days - block_days + 1, size=(N_BOOTSTRAP, n_blocks))
    days = (starts[:, :, None] + np.arange(block_days)[None, None, :]).reshape(N_BOOTSTRAP, -1)
    return days[:, :n_days]


def bootstrap_means(values, day, draws):
    """Bootstrap distribution of the mean of `values` (locations x samples) with days resampled by `draws`."""
    n_days = draws.shape[1]
    day_sum = np.bincount(day, weights=values.sum(axis=0), minlength=n_days)
    day_count = np.bincount(day, minlength=n_days) * values.shape[0]
    return day_sum[draws].sum(axis=1) / day_count[draws].sum(axis=1)


data = {model: load(model) for model in MODELS}
for model, d in data.items():
    print(f"{model}: {len(d)} locations")

# Common time axis
reference = next(d for d in data.values() if d)
time = next(iter(reference.values()))["time"]
day = day_index(time)
n_days = int(day.max()) + 1
for model, d in data.items():
    for location, f in d.items():
        assert np.array_equal(f["time"], time), f"{model}, location {location}: different time axis"

draws = {b: block_draws(n_days, b, np.random.default_rng(SEED)) for b in BLOCK_DAYS}

pair_rows = []
for a, b in PAIRS:
    locations = sorted(set(data[a]) & set(data[b]))
    if not locations:
        print(f"{a} vs. {b}: no common locations, skipped")
        continue
    for score in SCORES:
        key = score.lower()
        score_a = np.stack([data[a][loc][key] for loc in locations])
        score_b = np.stack([data[b][loc][key] for loc in locations])
        diff = score_b - score_a
        diff_loc = diff.mean(axis=1)
        p_wilcoxon = float(wilcoxon(diff_loc).pvalue) if len(locations) > 1 else np.nan
        for block_days in BLOCK_DAYS:
            boot = bootstrap_means(diff, day, draws[block_days])
            pair_rows.append({
                "A": a, "B": b, "Score": score, "block_days": block_days, "n_locations": len(locations),
                "mean_A": score_a.mean(), "mean_B": score_b.mean(), "mean_D": diff.mean(),
                "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                "p_bootstrap": min(1.0, 2 * (min(np.sum(boot <= 0), np.sum(boot >= 0)) + 1) / (N_BOOTSTRAP + 1)),
                "p_wilcoxon": p_wilcoxon, "n_B_better": int(np.sum(diff_loc < 0)),
            })

model_rows = []
for model, d in data.items():
    if not d:
        continue
    for score in SCORES:
        values = np.stack([d[loc][score.lower()] for loc in sorted(d)])
        for block_days in BLOCK_DAYS:
            boot = bootstrap_means(values, day, draws[block_days])
            model_rows.append({"Model": model, "Score": score, "block_days": block_days, "n_locations": len(d),
                               "mean": values.mean(), "ci_low": np.percentile(boot, 2.5),
                               "ci_high": np.percentile(boot, 97.5)})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pairs = pd.DataFrame(pair_rows)
models = pd.DataFrame(model_rows)
pairs.to_csv(OUTPUT_DIR / "significance_pairs.csv", index=False)
models.to_csv(OUTPUT_DIR / "significance_models.csv", index=False)

pd.set_option("display.width", 200)
p_columns = ["p_bootstrap", "p_wilcoxon"]
print("\n" + pairs.round({c: 4 for c in pairs.columns if c not in p_columns})
      .to_string(index=False, formatters={c: "{:.1e}".format for c in p_columns}))
print("\n" + models.round(4).to_string(index=False))
print(f"\nResults in {OUTPUT_DIR}")
