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

# Params
model_type = "AnEn"
foresight = 1
time_to_target = 1
seq_len = 3

# Initialize lists to store results for all locations
combined_y_t = []
combined_ensemble = []

for location in range(forecasts.shape[1]):
    print(f"Evaluating Location {location}")
    # Set up Data [time, location, variable]
    locational_observations=observations[:,location,:]
    locational_forecasts=forecasts[:,location,:]

    idx = foresight + time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) -2
    t_start = -test_size -foresight if foresight else -test_size
    t_end = -foresight if foresight else None
    h_start = seq_len -1
    h_end = -test_size

    # Set up evaluation data (feature 0 -> total wind-speed)
    x_t = locational_forecasts   [t_start:t_end, 0]    # NWP for time t
    y_t = locational_observations[t_start:t_end, 0]    # Observation of time t
    y_h = locational_observations[h_start:h_end, 0]    # Observation archive aligned to hist embeddings

    # AnEn
    AnEn = A2E.model.AnEn(
        n_analogs=100,
        temporal_window=foresight
    )

    inputs = [
        locational_forecasts[-(test_size+2):], # +2 to produce test_size many ensembles
        locational_forecasts[:-test_size],
        locational_observations[:-test_size]
    ]
    AnEn.build((inputs[0].shape, inputs[1].shape, inputs[2].shape))

    # Retrieve top-k AnEn ensemble
    ensemble = AnEn(inputs)

    # Append the data for this location to the combined lists
    combined_y_t.extend(y_t.numpy())
    combined_ensemble.extend(ensemble.numpy())

# Now create a single rank histogram plot with all combined data
print("\rInitializing Combined Rank Histogram...", end="", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))

A2E.io.plotting.rank_histogram(
    y_true=tf.constant(combined_y_t),
    ensemble_observations=tf.constant(combined_ensemble),
    handle_ties="uniform",
    ax=ax,
    title=model_type
)

plt.tight_layout()
plot_type = "combined_rank-histogram"
save_path = f"Trained_Models/{model_type}/combined_{plot_type}.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()
print("\rFinished.", end="", flush=True)
