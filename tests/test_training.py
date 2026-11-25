"""
Manual smoke test: trains A2E for location 0 with a short archive (lookback 160) and writes tests/test.keras
(with _config.json and _history.json). tests/test_evaluation.py evaluates this model.

Run from the repository root (delete tests/test* model files of an earlier run first):
    .venv/bin/python tests/test_training.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import a2e

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.data.data_loader import WindDataLoader

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Setup ModelConfig (as A2E/A2E_Train.py with a short archive)
config = a2e.io.ModelConfig(
    model_type="A2E",
    similarity_metric="cosine_similarity",
    d_model=32,
    n_blocks=1,
    seq_len=1*(2**4),
    lookback=160,
    foresight=1,
    time_to_target=2,
    d_vars=2,
    dropout=0.3,
    seed=SEED,
)

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)

# Features [x_{r+3}, y_r] and targets y_{r+3} of location 0
features, targets = a2e.io.build_features(forecasts[:, 0], observations[:, 0], config)

# Name the model
directory = PROJECT_ROOT / "tests"
save_path = str(directory / "test.keras")

api = a2e.io.Api(config=config)
api.train(
    forecasts=features,
    observations=targets,
    save_path=save_path,
    epochs=10,
    test_size=0.3,
    batch_size=64,
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
    loss=a2e.loss.SCRPS(),
    metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric(), a2e.metrics.keras.ESSMetric()],
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, mode="min",
                                            restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if epoch < 5 else float(lr * tf.math.exp(-0.01).numpy())),
    ],
)

exit()
