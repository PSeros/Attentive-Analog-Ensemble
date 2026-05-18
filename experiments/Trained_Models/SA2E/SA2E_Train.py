import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import model_paths

# Setup ModelConfig
config = a2e.io.ModelConfig(
    model_type="SA2E",
    similarity_metric="cosine_similarity",
    d_model=32,
    n_blocks=5,
    seq_len=5*(2**4), # 80 time steps, 16 per block
    lookback=8760*2,  # 2 years
    foresight=1,      # t+1
    time_to_target=2, # -> lead time
    d_vars=2,         # Total wind speed forecast and causal observation
    d_loc=6,          # Number of nearest neighbors
    dropout=0.3,
)

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, tf.float32)
forecasts = tf.cast(forecasts, tf.float32)

# Find the nearest locations
nearest_locations = loader.find_nearest_locations(config.d_loc-1)

for location, neighbours in enumerate(nearest_locations):
    print(f"\n Training Location {location}/{forecasts.shape[1]-1}")
    # Set up Data [time, location, variable]
    locational_observations = tf.gather(observations, neighbours, axis=1)
    locational_forecasts = tf.gather(forecasts, neighbours, axis=1)

    # Use Causal Observations as Variable
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, 0, :1] # Desired Location and first Variable

    # Name the model
    model_name = f"{config.model_type}_{config.similarity_metric}_Location{location}"
    paths = model_paths(config.model_type, model_name, location)
    directory = paths["directory"]
    save_path = str(paths["model"])
    directory.mkdir(parents=True, exist_ok=True)

    # a2e.io.Api is a High-Level api for Training, Embedding and Retrieving
    api = a2e.io.Api(config=config)
    api.train(
        forecasts=locational_forecasts,
        observations=locational_observations,
        save_path=save_path,
        epochs=100,
        test_size=0.3,
        batch_size=32,
        accum_steps=2,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
        loss=a2e.loss.SCRPS(),
        metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric()],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss", mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, mode="min",
                                             restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if epoch < 5 else lr * tf.math.exp(-0.01)),
        ],
    )
exit()
