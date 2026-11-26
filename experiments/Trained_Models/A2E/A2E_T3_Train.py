"""
A2E with a temporal window of T = 3 (ablation).

Difference from A2E_Train.py: seq_len=3, n_blocks=2. One causal convolution (kernel 2, dilation 1) per block gives
a receptive field of 1 + 2·(2^1 − 1) = 3: x_{t-1}, x_t, x_{t+1} and y_{t-4}, y_{t-3}, y_{t-2}, the window of AnEn
(t ± 1).

Training as in the paper (Table 1): SCRPS loss, AdamW 1e-4 with exponential decay after 5 epochs, at most 100 epochs,
early stopping (patience 15) and checkpoint on val_loss of the validation period (last 30 %).
Api.train saves the weights of the best epoch and writes the history JSON (ESS, g, gradient share outside top-100).

Run from the repository root:
    .venv/bin/python -m experiments.Trained_Models.A2E.A2E_T3_Train            # all 82 locations
    .venv/bin/python -m experiments.Trained_Models.A2E.A2E_T3_Train 0 41       # locations 0 ... 40 (end exclusive)

Output: Trained_Models/A2E/Location{i}/A2E_cosine_similarity_T3_Location{i}.keras
        with _config.json, _history.json and the training plot.
A location is skipped if its history JSON exists (finished run). If only the model file exists (interrupted run),
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
    model_type="A2E",
    similarity_metric="cosine_similarity",
    d_model=32,
    n_blocks=2,
    seq_len=3,        # receptive field 1 + 2·(2^1 − 1) = 3: x_{t-1} ... x_{t+1}
    lookback=8760*2,  # 2 years
    foresight=1,      # t+1
    time_to_target=2, # -> lead time
    d_vars=2,         # Total wind speed forecast and causal observation
    dropout=0.3,
    seed=SEED,
)

# Load Data [time, location, variable]
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)

first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
last = int(sys.argv[2]) if len(sys.argv) > 2 else forecasts.shape[1]

for location in range(first, last):
    model_name = f"{config.model_type}_{config.similarity_metric}_T3_Location{location}"
    paths = model_paths(config.model_type, model_name, location)
    if paths["history"].exists():
        print(f"Location {location}: already trained, skipped ({paths['history']})")
        continue
    print(f"\n Training Location {location}/{forecasts.shape[1]-1}: {model_name}")
    paths["directory"].mkdir(parents=True, exist_ok=True)

    # Features [x_{r+3}, y_r] and targets y_{r+3}, see a2e.io.build_features
    features, targets = a2e.io.build_features(forecasts[:, location], observations[:, location], config)

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
