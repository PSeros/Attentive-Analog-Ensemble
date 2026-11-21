"""
Manual smoke test: evaluates tests/test.keras (written by tests/test_training.py) for location 0 with the
evaluation helpers (a2e.io.build_features, a2e.io.evaluation_split, softmax top-k with k = 100).

Run from the repository root:
    .venv/bin/python tests/test_evaluation.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import a2e

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.data.data_loader import WindDataLoader

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)

# Prepare paths
model_path = PROJECT_ROOT / "tests" / "test.keras"
config_path = PROJECT_ROOT / "tests" / "test_config.json"
config = a2e.io.ModelConfig.load_from_json(config_path)
api = a2e.io.Api(config=config)

# Only evaluate location 0
location = 0
print(f"Evaluating Location {location}")
features, targets = a2e.io.build_features(forecasts[:, location], observations[:, location], config)

# Validation samples of Table 2 and the static archive before them
test_size = a2e.io.evaluation_test_size(forecasts.shape[0], lookback=8760 * 2)
split = a2e.io.evaluation_split(features, targets, config, test_size)

# Embeddings, top-k ensemble
embeddings = api.embed(features, model_path=str(model_path), verbose=True)
ensemble, weights = api.retrieve(embeddings[split.query], embeddings[split.archive], split.y_h, k=100)

# Metrics (RMSE of the ensemble mean)
bias = a2e.metrics.computation.compute_bias(ensemble, weights, split.y_t).numpy()
crps = -a2e.metrics.computation.compute_crps(ensemble, weights, split.y_t)
scrps = -a2e.metrics.computation.compute_scrps(ensemble, weights, split.y_t)

print("\nA2E Evaluation Metrics")
print(f"Location: {location}")
print(f"Bias:  {np.mean(bias):.4f}")
print(f"RMSE:  {np.sqrt(np.mean(bias ** 2)):.4f}")
print(f"CRPS:  {tf.reduce_mean(crps).numpy():.4f}")
print(f"SCRPS: {tf.reduce_mean(scrps).numpy():.4f}")
