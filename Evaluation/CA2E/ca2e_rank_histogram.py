import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import tensorflow as tf
from Data.data_loader import WindDataLoader
import A2E

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)

# Initialize lists to store results for all locations
combined_y_t = []
combined_ensemble = []
combined_weights = []

for location in range(forecasts.shape[1]):
    print(f"Evaluating Location {location}")
    # Prepare paths
    model_type = "CA2E"
    similarity_metric = "euclidean_distance"
    model_name = f"{model_type}_{similarity_metric}"
    directory = f"Trained_Models/{model_type}"
    model_path = f"{directory}/{model_name}.keras"
    config_path = f"{directory}/{model_name}_config.json"

    # Load Model Configs
    config = A2E.io.ModelConfig.load_from_json(config_path)

    # Initialize Api
    api = A2E.io.Api(config)

    # Set up Data [time, location, variable]
    locational_observations=observations[:,location,:]
    locational_forecasts=forecasts[:,location,:]

    # Add observation as causal feature
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) -2
    t_start = -test_size -config.foresight if config.foresight else -test_size
    t_end = -config.foresight if config.foresight else None
    h_start = config.seq_len -1
    h_end = -test_size

    # Set up evaluation data (feature 0 -> total wind-speed)
    x_t = locational_forecasts   [t_start:t_end, 0]    # NWP for time t
    y_t = locational_observations[t_start:t_end, 0]    # Observation of time t
    y_h = locational_observations[h_start:h_end, 0]    # Observation archive aligned to hist embeddings

    # Compute embeddings (at the same time load model into api)
    embeddings = api.embed(
        forecasts=locational_forecasts,
        model_path=model_path,
        verbose=True
    )

    # Split embeddings
    z_t = embeddings[-test_size:]
    z_h = embeddings[:-test_size]

    # Retrieve top-k CA2E ensemble and corresponding weights
    ensemble, weights = api.retrieve(z_t, z_h, y_h, k=100)

    # Append the data for this location to the combined lists
    combined_y_t.extend(y_t.numpy())
    combined_ensemble.extend(ensemble.numpy())
    combined_weights.extend(weights.numpy())

# Now create a single rank histogram plot with all combined data
print("\rInitializing Combined Rank Histogram...", end="", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))

A2E.io.plotting.rank_histogram(
    y_true=tf.constant(combined_y_t),
    ensemble_observations=tf.constant(combined_ensemble),
    member_weights=tf.constant(combined_weights),
    handle_ties="uniform",
    ax=ax,
    title=model_type
)

plt.tight_layout()
plot_type = "rank-histogram"
save_path = f"Evaluation/figures/{model_type}_{plot_type}.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()
print("\rFinished.", end="", flush=True)
