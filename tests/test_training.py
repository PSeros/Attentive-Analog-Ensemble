import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import a2e

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.data.data_loader import WindDataLoader

# Setup ModelConfig
config = a2e.io.ModelConfig(
    model_type="A2E",
    similarity_metric="cosine_similarity",
    d_model=32,
    n_blocks=5,
    seq_len=5*(2**4),
    lookback=160,
    foresight=1,
    time_to_target=2,
    d_vars=2,
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

# Set up Data [time, location, variable]
locational_observations = observations[:,0,:]
locational_forecasts = forecasts[:,0,:]

# Use Causal Observations as Variable
idx = config.foresight + config.time_to_target
locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
locational_observations = locational_observations[idx:] # First Location and first Variable

# Name the model
model_name = f"test"
directory = f"tests"
save_path = f"{directory}/{model_name}.keras"
os.makedirs(directory, exist_ok=True)

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

exit()