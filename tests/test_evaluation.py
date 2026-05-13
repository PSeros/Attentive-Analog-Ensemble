import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

import tensorflow as tf
import a2e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data.data_loader import WindDataLoader


# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)

observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)


# Prepare paths
model_name = "test"
directory = "tests"
model_path = f"{directory}/{model_name}.keras"
config_path = f"{directory}/{model_name}_config.json"

# Same config as in training
config = a2e.io.ModelConfig.load_from_json(config_path)

# Initialize Api
api = a2e.io.Api(config=config)

# Only evaluate location 0
location = 0
print(f"Evaluating Location {location}")

# Set up Data [time, location, variable]
locational_observations = observations[:, location, :]
locational_forecasts = forecasts[:, location, :]

# Same causal feature setup as in training
idx = config.foresight + config.time_to_target
locational_forecasts = tf.concat(
    [locational_forecasts[idx:], locational_observations[:-idx]],
    axis=-1
)
locational_observations = locational_observations[idx:]


# Set indices based on model configurations
test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2

t_start = -test_size - config.foresight if config.foresight else -test_size
t_end = -config.foresight if config.foresight else None

h_start = config.seq_len - 1
h_end = -test_size


# Evaluation data
y_t = locational_observations[t_start:t_end, 0]
y_h = locational_observations[h_start:h_end, 0]


# Compute embeddings and load model into api
embeddings = api.embed(
    locational_forecasts,
    model_path=model_path,
    verbose=True
)

# Split embeddings
z_t = embeddings[-test_size:]
z_h = embeddings[:-test_size]


# Retrieve top-k A2E ensemble and corresponding weights
a2e_ensemble, a2e_weights = api.retrieve(
    z_t,
    z_h,
    y_h,
    k=100
)


# Compute metrics
bias = a2e.metrics.computation.compute_bias(a2e_ensemble, a2e_weights, y_t)
rmse = a2e.metrics.computation.compute_rmse(a2e_ensemble, a2e_weights, y_t)
crps = -a2e.metrics.computation.compute_crps(a2e_ensemble, a2e_weights, y_t)
scrps = -a2e.metrics.computation.compute_scrps(a2e_ensemble, a2e_weights, y_t)


# Compute mean
bias = tf.reduce_mean(bias).numpy()
rmse = tf.reduce_mean(rmse).numpy()
crps = tf.reduce_mean(crps).numpy()
scrps = tf.reduce_mean(scrps).numpy()


# Print only metrics
print("\nA2E Evaluation Metrics")
print(f"Location: {location}")
print(f"Bias:  {bias:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"CRPS:  {crps:.4f}")
print(f"SCRPS: {scrps:.4f}")