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

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

model_configs = [
    {"name": "AnEn", "similarity": None, "location_specific": False},
    {"name": "A2E", "similarity": "cosine_similarity", "location_specific": True},
    {"name": "CA2E", "similarity": "cosine_similarity", "location_specific": False},
    {"name": "SA2E", "similarity": "cosine_similarity", "location_specific": True}
]

for model_idx, model_config in enumerate(model_configs):
    print(f"\nProcessing {model_config['name']}...")

    # Initialize lists to store results for all locations
    combined_y_t = []
    combined_ensemble = []
    combined_weights = []

    # Determine number of locations to process
    num_locations = forecasts.shape[1] if model_config['name'] != 'SA2E' else 82

    for location in range(num_locations):
        print(f"  Evaluating Location {location}")

        if model_config['name'] == "A2E":
            # A2E Model Processing
            model_type = "A2E"
            similarity_metric = "cosine_similarity"
            model_name = f"{model_type}_{similarity_metric}_Location{location}"
            directory = f"Trained_Models/{model_type}/Location{location}"
            model_path = f"{directory}/{model_name}.keras"
            config_path = f"{directory}/{model_name}_config.json"

            config = A2E.io.ModelConfig.load_from_json(config_path)
            api = A2E.io.Api(config)

            locational_observations = observations[:, location, :]
            locational_forecasts = forecasts[:, location, :]

            idx = config.foresight + config.time_to_target
            locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
            locational_observations = locational_observations[idx:, :1]

            test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
            t_start = -test_size - config.foresight if config.foresight else -test_size
            t_end = -config.foresight if config.foresight else None
            h_start = config.seq_len - 1
            h_end = -test_size

            y_t = locational_observations[t_start:t_end, 0]
            y_h = locational_observations[h_start:h_end, 0]

            embeddings = api.embed(forecasts=locational_forecasts, model_path=model_path, verbose=False)
            z_t = embeddings[-test_size:]
            z_h = embeddings[:-test_size]

            ensemble, weights = api.retrieve(z_t, z_h, y_h, k=100)

            combined_y_t.extend(y_t.numpy())
            combined_ensemble.extend(ensemble.numpy())
            combined_weights.extend(weights.numpy())

        elif model_config['name'] == "CA2E":
            # CA2E Model Processing
            model_type = "CA2E"
            similarity_metric = "cosine_similarity"
            model_name = f"{model_type}_{similarity_metric}"
            directory = f"Trained_Models/{model_type}"
            model_path = f"{directory}/{model_name}.keras"
            config_path = f"{directory}/{model_name}_config.json"

            config = A2E.io.ModelConfig.load_from_json(config_path)
            api = A2E.io.Api(config)

            locational_observations = observations[:, location, :]
            locational_forecasts = forecasts[:, location, :]

            idx = config.foresight + config.time_to_target
            locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
            locational_observations = locational_observations[idx:, :1]

            test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
            t_start = -test_size - config.foresight if config.foresight else -test_size
            t_end = -config.foresight if config.foresight else None
            h_start = config.seq_len - 1
            h_end = -test_size

            y_t = locational_observations[t_start:t_end, 0]
            y_h = locational_observations[h_start:h_end, 0]

            embeddings = api.embed(forecasts=locational_forecasts, model_path=model_path, verbose=False)
            z_t = embeddings[-test_size:]
            z_h = embeddings[:-test_size]

            ensemble, weights = api.retrieve(z_t, z_h, y_h, k=100)

            combined_y_t.extend(y_t.numpy())
            combined_ensemble.extend(ensemble.numpy())
            combined_weights.extend(weights.numpy())

        elif model_config['name'] == "AnEn":
            # AnEn Model Processing
            foresight = 1
            time_to_target = 1
            seq_len = 3

            locational_observations = observations[:, location, :]
            locational_forecasts = forecasts[:, location, :]

            idx = foresight + time_to_target
            locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
            locational_observations = locational_observations[idx:, :1]

            test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
            t_start = -test_size - foresight if foresight else -test_size
            t_end = -foresight if foresight else None
            h_start = seq_len - 1
            h_end = -test_size

            y_t = locational_observations[t_start:t_end, 0]

            AnEn = A2E.model.AnEn(n_analogs=100, temporal_window=foresight)

            inputs = [
                locational_forecasts[-(test_size + 2):],
                locational_forecasts[:-test_size],
                locational_observations[:-test_size]
            ]
            AnEn.build((inputs[0].shape, inputs[1].shape, inputs[2].shape))
            ensemble = AnEn(inputs)

            combined_y_t.extend(y_t.numpy())
            combined_ensemble.extend(ensemble.numpy())
            # AnEn doesn't use weights, so we'll create uniform weights
            uniform_weights = tf.ones((ensemble.shape[0], ensemble.shape[1])) / ensemble.shape[1]
            combined_weights.extend(uniform_weights.numpy())

        elif model_config['name'] == "SA2E":
            # SA2E Model Processing
            model_type = "SA2E"
            similarity_metric = "cosine_similarity"
            model_name = f"{model_type}_{similarity_metric}_Location{location}"
            directory = f"Trained_Models/{model_type}/Location{location}"
            model_path = f"{directory}/{model_name}.keras"
            config_path = f"{directory}/{model_name}_config.json"

            config = A2E.io.ModelConfig.load_from_json(config_path)
            api = A2E.io.Api(config)

            nearest_locations = loader.find_nearest_locations(config.d_loc - 1)
            neighbours = nearest_locations[location, :]

            locational_observations = tf.gather(observations, neighbours, axis=1)
            locational_forecasts = tf.gather(forecasts, neighbours, axis=1)

            idx = config.foresight + config.time_to_target
            locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
            locational_observations = locational_observations[idx:, 0, :1]

            test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
            t_start = -test_size - config.foresight if config.foresight else -test_size
            t_end = -config.foresight if config.foresight else None
            h_start = config.seq_len - 1
            h_end = -test_size

            y_t = locational_observations[t_start:t_end, 0]
            y_h = locational_observations[h_start:h_end, 0]

            embeddings = api.embed(locational_forecasts, model_path=model_path, verbose=False)
            z_t = embeddings[-test_size:]
            z_h = embeddings[:-test_size]

            ensemble, weights = api.retrieve(z_t, z_h, y_h, k=100)

            combined_y_t.extend(y_t.numpy())
            combined_ensemble.extend(ensemble.numpy())
            combined_weights.extend(weights.numpy())

    # Create rank histogram for current model
    print(f"  Creating rank histogram for {model_config['name']}...")

    # Handle weights (AnEn uses uniform weights if not provided)
    if model_config['name'] == "AnEn":
        A2E.io.plotting.rank_histogram(
            y_true=tf.constant(combined_y_t),
            ensemble_observations=tf.constant(combined_ensemble),
            handle_ties="uniform",
            ax=axes[model_idx],
            title=model_config['name']
        )
    else:
        A2E.io.plotting.rank_histogram(
            y_true=tf.constant(combined_y_t),
            ensemble_observations=tf.constant(combined_ensemble),
            member_weights=tf.constant(combined_weights),
            handle_ties="uniform",
            ax=axes[model_idx],
            title=model_config['name']
        )

# Adjust layout and save
plt.tight_layout()
save_path = "Evaluation/figures/combined_rank_histograms.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\nFinished creating combined rank histograms!")