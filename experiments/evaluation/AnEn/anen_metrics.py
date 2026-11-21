"""
Evaluation of the Analog Ensemble (Table 2): predictors x_{t-1}, x_t, x_{t+1} and y_{t-4}, y_{t-3}, y_{t-2}.

Validation period (3720 samples per location) with a static archive of all earlier windows whose value is known at
the issue time of the first sample (a2e.io.evaluation_split).
Ensemble: k = 100 analogs of the window t ± 1, predictor weights 1, uniform member weights.
Scores: bias, RMSE of the ensemble mean, CRPS, SCRPS (m/s); per sample also the randomized PIT.

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.AnEn.anen_metrics            # all locations
    .venv/bin/python -m experiments.evaluation.AnEn.anen_metrics 0 41       # locations 0 ... 40 (end exclusive)

Results:
    evaluation/metrics_evaluation.csv (Model AnEn)
    evaluation/AnEn/per_sample/AnEn_Location{i}.npz
"""

from ..common import Data, anen_ensemble, location_range, scores, summary, save_per_sample, write_metrics, print_summary
from ...project_paths import METRICS_EVALUATION_CSV

# Load Data
data = Data()

results = {}
for location in location_range(data.n_locations):
    # Split (same queries and archive end as A2E) and AnEn ensemble
    ensemble = anen_ensemble(data, location, observation_inputs=True)

    # Metrics, per-sample output
    sample_scores = scores(ensemble, pit_seed=location)
    save_per_sample("AnEn", f"AnEn_Location{location}", data, ensemble, sample_scores)
    results[location] = summary(sample_scores)
    print(f"Location {location} - " + ", ".join(f"{m}: {v:.4f}" for m, v in results[location].items()))

# Location means into the CSV
write_metrics(METRICS_EVALUATION_CSV, results, model="AnEn")
print_summary(results, "AnEn")
