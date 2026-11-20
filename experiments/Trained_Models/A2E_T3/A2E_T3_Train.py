"""
Ablation (review, major comment 2): A2E with a short temporal window, T = 3.

Identical to A2E/A2E_Train.py (data, splits, loss, optimizer, callbacks, epochs) except for the encoder size:
seq_len = 4 and n_blocks = 2 give one causal convolution (kernel 2, dilation 1) per block,
i.e. a receptive field of 1 + 2 * 1 = 3 time steps: x_{t-1}, x_t, x_{t+1} and y_{t-4}, y_{t-3}, y_{t-2}.
This is the same window as the AnEn baseline (t ± 1).
(The paper model uses seq_len = 80, n_blocks = 5 with 4 layers each, receptive field 1 + 5 * 15 = 76.)

Models are saved to Trained_Models/A2E_T3/Location<i>/A2E_T3_cosine_similarity_Location<i>.keras.
Locations with a saved config are skipped, so the script can be restarted.

Run from the repository root:
    python -m experiments.Trained_Models.A2E_T3.A2E_T3_Train              # all 82 locations
    python -m experiments.Trained_Models.A2E_T3.A2E_T3_Train 0 41         # locations 0..40 (end exclusive)
"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import model_paths

# Setup ModelConfig
config = a2e.io.ModelConfig(
    model_type="A2E",
    similarity_metric="cosine_similarity",
    d_model=32,
    n_blocks=2,
    seq_len=4,        # 2 blocks with 2 time steps each -> receptive field 3
    lookback=8760*2,  # 2 years
    foresight=1,      # t+1
    time_to_target=2, # -> lead time
    d_vars=2,         # Total wind speed forecast and causal observation
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

first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
last = int(sys.argv[2]) if len(sys.argv) > 2 else forecasts.shape[1]

for location in range(first, last):
    print(f"\n Training Location {location}/{forecasts.shape[1]-1}")
    # Name the model
    model_name = f"A2E_T3_{config.similarity_metric}_Location{location}"
    paths = model_paths("A2E_T3", model_name, location)
    if paths["config"].exists():
        print(f"Already trained, skipping: {paths['model']}")
        continue
    directory = paths["directory"]
    save_path = str(paths["model"])
    directory.mkdir(parents=True, exist_ok=True)

    # Set up Data [time, location, variable]
    locational_observations = observations[:,location,:]
    locational_forecasts = forecasts[:,location,:]

    # Use Causal Observations as Variable
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:] # First Location and first Variable

    # a2e.utils.Api is a High-Level wrapper for Training, Embedding and Retrieving
    api = a2e.io.Api(config=config)
    api.train(
        forecasts=locational_forecasts,
        observations=locational_observations,
        save_path=save_path,
        epochs=100,
        test_size=0.3,
        batch_size=64,
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
