"""
Per-sample files of the evaluation scripts (evaluation/<model type>/per_sample/<name>.npz, written by
experiments.evaluation.common.save_per_sample): time (target time, UTC), y, ens_mean, bias, crps, scrps, pit.
"""
import numpy as np
from ..project_paths import EVALUATION_DIR

N_LOCATIONS = 82

# Model label: per-sample file relative to evaluation/, {} is the location
MODELS = {
    "AnEn": "AnEn/per_sample/AnEn_Location{}.npz",
    "A2E": "A2E/per_sample/A2E_cosine_similarity_Location{}.npz",
    "CA2E": "CA2E/per_sample/CA2E_cosine_similarity_Location{}.npz",
    "SA2E": "SA2E/per_sample/SA2E_cosine_similarity_Location{}.npz",
    "A2E T3": "A2E/per_sample/A2E_cosine_similarity_T3_Location{}.npz",
    "A2E no_obs": "A2E/per_sample/A2E_cosine_similarity_no_obs_Location{}.npz",
    "AnEn no_obs": "AnEn/per_sample/AnEn_no_obs_Location{}.npz",
    "A2E euclidean": "A2E/per_sample/A2E_euclidean_distance_Location{}.npz",
    "CA2E euclidean": "CA2E/per_sample/CA2E_euclidean_distance_Location{}.npz",
}


def load(model):
    """{location: {key: array}} of the locations with a per-sample file of `model` (label of MODELS)."""
    result = {}
    for location in range(N_LOCATIONS):
        path = EVALUATION_DIR / MODELS[model].format(location)
        if path.exists():
            with np.load(path) as f:
                result[location] = {key: f[key] for key in f.files}
    return result
