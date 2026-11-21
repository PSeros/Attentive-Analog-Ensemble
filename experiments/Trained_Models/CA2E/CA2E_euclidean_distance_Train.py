"""
CA2E with Euclidean distance as similarity (appendix, ablation of the similarity metric).

Difference from CA2E_Train.py: similarity_metric="euclidean_distance" (negative Euclidean distance, no
learned temperature g).

Training as in the paper (Table 1): SCRPS loss, AdamW 1e-4 with exponential decay after 5 epochs, at most 100 epochs,
early stopping (patience 15) and checkpoint on val_loss of the validation period (last 30 %).
Api.train saves the weights of the best epoch and writes the history JSON (ESS, g, gradient share outside top-100).

Run from the repository root:
    .venv/bin/python -m experiments.Trained_Models.CA2E.CA2E_euclidean_distance_Train

Output: Trained_Models/CA2E/CA2E_euclidean_distance.keras
        with _config.json, _history.json and the training plot.
The run is skipped if the history JSON exists (finished run). If only the model file exists (interrupted run),
Api.train asks whether to resume or to start over.
"""
import os
import sys
from dataclasses import replace
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import model_paths

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Setup ModelConfig
config = a2e.io.ModelConfig(
    model_type="CA2E",
    similarity_metric="euclidean_distance",
    d_model=32,
    n_blocks=5,
    seq_len=76,       # receptive field 1 + 5·(2^4 − 1) = 76: x_{t-74} ... x_{t+1}
    lookback=8760*2,  # 2 years
    foresight=1,      # t+1
    time_to_target=2, # -> lead time
    d_vars=2,         # Total wind speed forecast and causal observation
    max_locations=82, # All locations in a single CA2E Model
    dropout=0.3,
    seed=SEED,
)

# Load Data [time, location, variable]
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)

model_name = f"{config.model_type}_{config.similarity_metric}"
paths = model_paths(config.model_type, model_name)
if paths["history"].exists():
    print(f"{model_name}: already trained, skipped ({paths['history']})")
    sys.exit()
print(f"\n Training {model_name}")
paths["directory"].mkdir(parents=True, exist_ok=True)

# Features [x_{r+3}, y_r] and targets y_{r+3} of all locations, see a2e.io.build_features
features, targets = a2e.io.build_features(forecasts, observations, config)

api = a2e.io.Api(config=replace(config))
api.train(
    forecasts=features,
    observations=targets,
    save_path=str(paths["model"]),
    epochs=100,
    test_size=0.3,
    batch_size=64,
    accum_steps=1,
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
    loss=a2e.loss.SCRPS(),
    metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric(), a2e.metrics.keras.ESSMetric()],
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, mode="min",
                                         restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if epoch < 5 else lr * tf.math.exp(-0.01)),
    ],
)
