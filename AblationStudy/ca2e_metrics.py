import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import tensorflow as tf
from Data.data_loader import WindDataLoader
import A2E

# Load the existing CSV
csv_path = "AblationStudy/metrics_evaluation.csv"
df = pd.read_csv(csv_path, index_col=0, header=[0, 1, 2])

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)

for similarity_metric in ["cosine_similarity", "euclidean_distance"]:
    # Prepare paths
    model_type = "CA2E"
    model_name = f"{model_type}_{similarity_metric}"
    directory = f"Trained_Models/{model_type}"
    model_path = f"{directory}/{model_name}.keras"
    config_path = f"{directory}/{model_name}_config.json"

    # Load Model Configs
    config = A2E.io.ModelConfig.load_from_json(config_path)

    # Initialize Api
    api = A2E.io.Api(config)
    for location in range(82):
        print(f"Evaluating Location {location}")
        # Set up Data [time, location, variable]
        locational_observations = observations[:, location, :]
        locational_forecasts = forecasts[:, location, :]

        # Add observation as causal feature
        idx = config.foresight + config.time_to_target
        locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
        locational_observations = locational_observations[idx:, :1]

        # Set indices based on model configurations (-2 -> to get the number of ensembles!)
        test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
        t_start = -test_size - config.foresight if config.foresight else -test_size
        t_end = -config.foresight if config.foresight else None
        h_start = config.seq_len - 1
        h_end = -test_size

        # Set up evaluation data (feature 0 -> total wind-speed)
        y_t = locational_observations[t_start:t_end, 0]    # Observation of time t
        y_h = locational_observations[h_start:h_end, 0]    # Observation archive aligned to hist embeddings

        # Compute embeddings (at the same time load model into api)
        embeddings = api.embed(locational_forecasts, model_path=model_path, verbose=True)

        # Split embeddings
        z_t = embeddings[-test_size:]
        z_h = embeddings[:-test_size]

        # Retrieve top-k A2E ensemble and corresponding weights
        a2e_ensemble, a2e_weights = api.retrieve(z_t, z_h, y_h, k=100)

        # Compute metrics
        bias = A2E.metrics.computation.compute_bias(a2e_ensemble, a2e_weights, y_t)
        rmse = A2E.metrics.computation.compute_rmse(a2e_ensemble, a2e_weights, y_t)
        crps = -A2E.metrics.computation.compute_crps(a2e_ensemble, a2e_weights, y_t)
        scrps = -A2E.metrics.computation.compute_scrps(a2e_ensemble, a2e_weights, y_t)

        # Compute mean
        bias = tf.reduce_mean(bias).numpy()
        rmse = tf.reduce_mean(rmse).numpy()
        crps = tf.reduce_mean(crps).numpy()
        scrps = tf.reduce_mean(scrps).numpy()

        # Update DataFrame
        location_idx = f"Location_{location}"
        df.loc[location_idx, ("Bias", model_type, similarity_metric)] = bias
        df.loc[location_idx, ("RMSE", model_type, similarity_metric)] = rmse
        df.loc[location_idx, ("CRPS", model_type, similarity_metric)] = crps
        df.loc[location_idx, ("SCRPS", model_type, similarity_metric)] = scrps

        print(f"Location {location} - Bias: {bias:.4f}, RMSE: {rmse:.4f}, CRPS: {crps:.4f}, SCRPS: {scrps:.4f}")


# Save updated DataFrame
df.to_csv(csv_path)
print(f"\nMetrics evaluation complete. Results saved to {csv_path}")

# Display summary statistics
print("\nA2E Metrics Summary:")
a2e_metrics = df.loc[:, (
                            ['SCRPS'],
                            ["CA2E"],
                            ["cosine_similarity", "euclidean_distance"],
                        )]
print(a2e_metrics.describe())