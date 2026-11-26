"""
Evaluation of A2E without observation inputs (ablation).

Validation period (3720 samples per location) with a static archive of all earlier windows whose value is known at
the issue time of the first sample (a2e.io.evaluation_split).
Ensemble: softmax-weighted top-k of the attention weights, k = 100.
Scores: bias, RMSE of the ensemble mean, CRPS, SCRPS (m/s); per sample also the randomized PIT.

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.AblationStudy.a2e_no_obs_metrics            # all locations
    .venv/bin/python -m experiments.evaluation.AblationStudy.a2e_no_obs_metrics 0 41       # locations 0 ... 40 (end exclusive)

Results:
    evaluation/AblationStudy/metrics_evaluation.csv (Model A2E, Variant cosine_similarity_no_obs)
    evaluation/A2E/per_sample/A2E_cosine_similarity_no_obs_Location{i}.npz
"""

from ..common import (Data, ModelSpec, ModelEnsembles, location_range, scores, summary, save_per_sample,
                      write_metrics, print_summary)
from ...project_paths import ABLATION_METRICS_EVALUATION_CSV

SPEC = ModelSpec("A2E", "cosine_similarity", variant="no_obs")

# Load Data
data = Data()
ensembles = ModelEnsembles(SPEC, data)

results = {}
for location in location_range(data.n_locations):
    if not ensembles.available(location):
        print(f"Location {location}: no trained model at {SPEC.paths(location)['model']}, skipped")
        results[location] = None
        continue

    # Config and model, split (queries, archive), embeddings and top-k ensemble
    ensemble = ensembles(location)

    # Metrics, per-sample output
    sample_scores = scores(ensemble, pit_seed=location)
    save_per_sample(SPEC.model_type, SPEC.name(location), data, ensemble, sample_scores)
    results[location] = summary(sample_scores)
    print(f"Location {location} - " + ", ".join(f"{m}: {v:.4f}" for m, v in results[location].items()))

# Location means into the CSV
write_metrics(ABLATION_METRICS_EVALUATION_CSV, results, model="A2E", variant=SPEC.label)
print_summary(results, f"{SPEC.model_type} {SPEC.label}")
