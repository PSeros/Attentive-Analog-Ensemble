import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import tensorflow as tf
from Data.data_loader import WindDataLoader
import A2E

# Load the existing CSV
csv_path = "Evaluation/metrics_evaluation.csv"
df = pd.read_csv(csv_path, index_col=0, header=[0, 1])

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)

# Evaluate all locations (change range to 82 for all locations)
for location in range(82):
    print(f"Evaluating Location {location}")
    # Add observation as causal feature
    foresight = 1
    time_to_target = 2
    idx = foresight + time_to_target

    # Set up Data [time, location, variable]
    locational_observations=observations[:,location,:]
    locational_forecasts=forecasts[:,location,:]

    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
    t_start = -test_size - foresight if foresight else -test_size
    t_end = -foresight if foresight else None

    # Set up evaluation data (feature 0 -> total wind-speed)
    y_t = locational_observations[t_start:t_end, 0]  # Observation of time t

    # Create AnEn Model
    model_type = "AnEn"
    AnEn = A2E.model.AnEn(
        n_analogs=100,
        temporal_window=foresight
    )
    anen_inputs = [
        locational_forecasts[-(test_size+2):], # +2 to produce test_size many ensembles
        locational_forecasts[:-test_size],
        locational_observations[:-test_size]
    ]
    AnEn.build((anen_inputs[0].shape, anen_inputs[1].shape, anen_inputs[2].shape))

    # Retrieve top-k AnEn ensemble
    ensemble = AnEn(anen_inputs)
    weights = tf.ones(ensemble.shape)/ensemble.shape[-1]

    # Compute metrics
    bias = A2E.metrics.computation.compute_bias(ensemble, weights, y_t)
    rmse = A2E.metrics.computation.compute_rmse(ensemble, weights, y_t)
    crps = -A2E.metrics.computation.compute_crps(ensemble, weights, y_t)
    scrps = -A2E.metrics.computation.compute_scrps(ensemble, weights, y_t)

    # Compute mean
    bias = tf.reduce_mean(bias).numpy()
    rmse = tf.reduce_mean(rmse).numpy()
    crps = tf.reduce_mean(crps).numpy()
    scrps = tf.reduce_mean(scrps).numpy()

    # Update DataFrame
    location_idx = f"Location_{location}"
    df.loc[location_idx, ("Bias", model_type)] = bias
    df.loc[location_idx, ("RMSE", model_type)] = rmse
    df.loc[location_idx, ("CRPS", model_type)] = crps
    df.loc[location_idx, ("SCRPS", model_type)] = scrps

    print(f"Location {location} - Bias: {bias:.4f}, RMSE: {rmse:.4f}, CRPS: {crps:.4f}, SCRPS: {scrps:.4f}")

# Save updated DataFrame
df.to_csv(csv_path)
print(f"\nMetrics evaluation complete. Results saved to {csv_path}")

# Display summary statistics
print("\nA2E Metrics Summary:")
a2e_metrics = df.loc[:, (["Bias", 'RMSE', 'CRPS', 'SCRPS'], model_type)]
print(a2e_metrics.describe())