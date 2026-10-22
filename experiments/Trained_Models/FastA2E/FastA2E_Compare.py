"""
Trains A2E and FastA2E on one location with the A2E settings of the paper, evaluates both like
experiments/evaluation/A2E/a2e_metrics.py and writes a comparison table, training curves and timings.

Run from the repository root:
    uv run python -m experiments.Trained_Models.FastA2E.FastA2E_Compare --location 0
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import METRICS_EVALUATION_CSV, TRAINED_MODELS_DIR

parser = argparse.ArgumentParser(description="Compare A2E and FastA2E on one location.")
parser.add_argument("--location", type=int, default=0)
parser.add_argument("--variants", nargs="+", default=["A2E", "FastA2E"], choices=["A2E", "FastA2E"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--key-stride", type=int, default=1, help="FastA2E only: every n-th historical step as key")
parser.add_argument("--out", type=str, default=None, help="Output directory (default: new folder in Trained_Models/FastA2E)")
args = parser.parse_args()

# Devices
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"TensorFlow {tf.__version__}, GPUs: {[gpu.name for gpu in gpus] or 'none, running on CPU'}")

# Output directory (always new, so Api.train never asks whether to resume)
out = args.out or str(TRAINED_MODELS_DIR / "FastA2E" / f"Compare_Location{args.location}_{time.strftime('%Y%m%d-%H%M%S')}")
os.makedirs(out, exist_ok=False)
print(f"Writing results to {out}")

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, tf.float32)
forecasts = tf.cast(forecasts, tf.float32)

# Set up Data [time, location, variable] -> Use Causal Observations as Variable
foresight, time_to_target = 1, 2
idx = foresight + time_to_target
locational_observations = observations[:, args.location, :]
locational_forecasts = forecasts[:, args.location, :]
locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
locational_observations = locational_observations[idx:]

class EpochTimer(tf.keras.callbacks.Callback):
    """Records the wall time of every epoch"""
    def __init__(self):
        super().__init__()
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self.start)

def evaluate(api, config):
    """Evaluation as in experiments/evaluation/A2E/a2e_metrics.py (top-k 100, archive before the test period)"""
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
    y_t = locational_observations[-test_size - config.foresight:-config.foresight, 0]
    y_h = locational_observations[config.seq_len - 1:-test_size, 0]
    embeddings = api.embed(locational_forecasts, verbose=False)
    ensemble, weights = api.retrieve(embeddings[-test_size:], embeddings[:-test_size], y_h, k=100)
    computation = a2e.metrics.computation
    return {
        "Bias": float(tf.reduce_mean(computation.compute_bias(ensemble, weights, y_t))),
        "RMSE": float(tf.reduce_mean(computation.compute_rmse(ensemble, weights, y_t))),
        "CRPS": float(tf.reduce_mean(-computation.compute_crps(ensemble, weights, y_t))),
        "SCRPS": float(tf.reduce_mean(-computation.compute_scrps(ensemble, weights, y_t))),
    }

results, histories = {}, {}
for model_type in args.variants:
    print(f"\n Training {model_type} on Location {args.location}")
    config = a2e.io.ModelConfig(
        model_type=model_type,
        similarity_metric="cosine_similarity",
        d_model=32,
        n_blocks=5,
        seq_len=5*(2**4),  # 80 time steps, 16 per block
        lookback=8760*2,   # 2 years
        foresight=foresight,
        time_to_target=time_to_target,
        d_vars=2,
        dropout=0.3,
        key_stride=args.key_stride if model_type == "FastA2E" else None,
    )
    save_path = f"{out}/{model_type}/{model_type}_cosine_similarity_Location{args.location}.keras"
    timer = EpochTimer()

    start = time.perf_counter()
    api = a2e.io.Api(config=config)
    api.train(
        forecasts=locational_forecasts,
        observations=locational_observations,
        save_path=save_path,
        epochs=args.epochs,
        test_size=0.3,
        batch_size=args.batch_size,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
        loss=a2e.loss.SCRPS(),
        metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric()],
        callbacks=[
            timer,
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss", mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, mode="min",
                                             restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if epoch < 5 else float(lr * tf.math.exp(-0.01).numpy())),
        ],
    )
    train_minutes = (time.perf_counter() - start) / 60

    history = {k: [float(v) for v in values] for k, values in api.training_history.history.items()}
    history["epoch_seconds"] = timer.seconds
    histories[model_type] = history
    best_epoch = int(pd.Series(history["val_loss"]).idxmin()) + 1
    results[model_type] = {
        **evaluate(api, config),
        "epochs trained": len(history["loss"]),
        "best epoch": best_epoch,
        "best val_loss": min(history["val_loss"]),
        "seconds per epoch": sum(timer.seconds) / len(timer.seconds),
        "training minutes": train_minutes,
    }
    print(pd.Series(results[model_type]).round(4).to_string())

    # Save after every variant, so a crash keeps what is done
    json.dump(histories, open(f"{out}/histories.json", "w"), indent=2)

# Comparison table incl. the paper results of this location
paper = pd.read_csv(METRICS_EVALUATION_CSV, index_col=0, header=[0, 1]).loc[f"Location_{args.location}"]
table = pd.DataFrame({
    "AnEn (paper)": {m: paper[(m, "AnEn")] for m in ["Bias", "RMSE", "CRPS", "SCRPS"]},
    "A2E (paper)": {m: paper[(m, "A2E")] for m in ["Bias", "RMSE", "CRPS", "SCRPS"]},
    **{f"{name} (this run)": values for name, values in results.items()},
})
table.to_csv(f"{out}/comparison.csv")
print(f"\nLocation {args.location}\n{table.round(4).to_string()}")

# Training curves
figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, history in histories.items():
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label=f"{name} loss")
    axes[0].plot(epochs, history["val_loss"], "--", label=f"{name} val_loss")
    axes[1].plot(epochs, history["val_crps"], label=f"{name} val_crps")
axes[0].set_title("SCRPS loss (training with dropout, validation without)")
axes[1].set_title("Validation CRPS")
for axis in axes:
    axis.set_xlabel("Epoch")
    axis.grid(alpha=0.3)
    axis.legend()
figure.tight_layout()
figure.savefig(f"{out}/training_curves.png", dpi=150)
print(f"\nSaved comparison.csv, histories.json and training_curves.png to {out}")
