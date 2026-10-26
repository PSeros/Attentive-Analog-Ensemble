"""
FastA2E evaluation: model performance and training time compared to A2E.

1) Trains FastA2E with the A2E settings of the paper on every location (skips locations already trained)
   and records the training time.
2) Trains A2E for a few epochs on the timing locations, on the same hardware, for its time per epoch.
3) Evaluates every FastA2E model like experiments/evaluation/A2E/a2e_metrics.py and writes the FastA2E
   columns into metrics_evaluation.csv.
4) Compares FastA2E with A2E and AnEn (paper results in metrics_evaluation.csv) and the training times.

Run from the repository root:
    uv run python -m experiments.evaluation.FastA2E.fasta2e_evaluation
Quick test on one location (results in a separate folder, metrics_evaluation.csv stays untouched):
    uv run python -m experiments.evaluation.FastA2E.fasta2e_evaluation --locations 0 --epochs 2 --out /tmp/fasta2e_test
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import tempfile
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import *

parser = argparse.ArgumentParser(description="Evaluate FastA2E: model performance and training time compared to A2E.")
parser.add_argument("--locations", type=int, nargs="+", default=None, help="Locations to train and evaluate (default: all)")
parser.add_argument("--epochs", type=int, default=100, help="Maximum FastA2E training epochs (early stopping as in the paper)")
parser.add_argument("--timing-locations", type=int, nargs="+", default=[0], help="Locations on which A2E is timed")
parser.add_argument("--a2e-timing-epochs", type=int, default=3, help="A2E epochs per timing location")
parser.add_argument("--out", type=str, default=None,
                    help="Write models and results to this folder instead of Trained_Models/FastA2E and evaluation/")
args = parser.parse_args()

METRICS = ["Bias", "RMSE", "CRPS", "SCRPS"]
COLORS = {"A2E": "#2a78d6", "FastA2E": "#eb6834"}

# Devices
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"TensorFlow {tf.__version__}, GPUs: {[gpu.name for gpu in gpus] or 'none, running on CPU'}")

# Output paths
results_dir = Path(args.out) if args.out else EVALUATION_DIR / "FastA2E"
results_dir.mkdir(parents=True, exist_ok=True)
metrics_csv_out = results_dir / "metrics_evaluation.csv" if args.out else METRICS_EVALUATION_CSV
training_csv = results_dir / "fasta2e_training_times.csv"
a2e_timing_csv = results_dir / "a2e_timing.csv"

def fast_paths(location):
    model_name = f"FastA2E_cosine_similarity_Location{location}"
    if not args.out:
        return model_paths("FastA2E", model_name, location)
    directory = results_dir / "FastA2E" / f"Location{location}"
    return {"directory": directory, "model": directory / f"{model_name}.keras",
            "config": directory / f"{model_name}_config.json"}

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)
locations = args.locations if args.locations is not None else list(range(forecasts.shape[1]))

def make_config(model_type):
    """A2E settings of the paper (experiments/Trained_Models/A2E/A2E_Train.py)"""
    return a2e.io.ModelConfig(
        model_type=model_type,
        similarity_metric="cosine_similarity",
        d_model=32,
        n_blocks=5,
        seq_len=5*(2**4), # 80 time steps, 16 per block
        lookback=8760*2,  # 2 years
        foresight=1,      # t+1
        time_to_target=2, # -> lead time
        d_vars=2,         # Total wind speed forecast and causal observation
        dropout=0.3,
        key_stride=1 if model_type == "FastA2E" else None,
    )

def location_data(location, config):
    """Forecasts with the causal observation as second variable, and the aligned observations"""
    locational_observations = observations[:, location, :]
    locational_forecasts = forecasts[:, location, :]
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]
    return locational_forecasts, locational_observations

class EpochTimer(tf.keras.callbacks.Callback):
    """Records the wall time of every epoch (training and validation)"""
    def __init__(self):
        super().__init__()
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self.start)

def train(model_type, location, save_path, epochs):
    """Trains with the settings of the paper and returns the training record"""
    config = make_config(model_type)
    locational_forecasts, locational_observations = location_data(location, config)
    timer = EpochTimer()
    start = time.perf_counter()
    api = a2e.io.Api(config=config)
    api.train(
        forecasts=locational_forecasts,
        observations=locational_observations,
        save_path=str(save_path),
        epochs=epochs,
        test_size=0.3,
        batch_size=64,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
        loss=a2e.loss.SCRPS(),
        metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric()],
        callbacks=[
            timer,
            tf.keras.callbacks.ModelCheckpoint(str(save_path), save_best_only=True, monitor="val_loss", mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, mode="min",
                                             restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if epoch < 5 else float(lr * tf.math.exp(-0.01).numpy())),
        ],
    )
    val_loss = api.training_history.history["val_loss"]
    return {
        "location": location,
        "epochs trained": len(val_loss),
        "best epoch": int(pd.Series(val_loss).idxmin()) + 1,
        "seconds per epoch": float(pd.Series(timer.seconds[1:] or timer.seconds).median()),  # epoch 1 includes tracing
        "training minutes": (time.perf_counter() - start) / 60,
    }

def load_records(path):
    return pd.read_csv(path, index_col="location") if path.exists() else pd.DataFrame()

def save_record(path, record):
    records = load_records(path)
    records.loc[record["location"], [k for k in record if k != "location"]] = [v for k, v in record.items() if k != "location"]
    records.index.name = "location"
    records.sort_index().to_csv(path)

########################################################################################################################
# 1) Train FastA2E on every location
########################################################################################################################
for location in locations:
    paths = fast_paths(location)
    if location in load_records(training_csv).index:
        print(f"Location {location}: FastA2E already trained, skipping")
        continue
    if paths["model"].exists():
        print(f"Location {location}: FastA2E model exists without a training record (trained elsewhere or "
              f"interrupted). It is evaluated as is; delete {paths['directory']} to retrain and time it.")
        continue
    print(f"\n Training FastA2E on Location {location}/{forecasts.shape[1]-1}")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    record = train("FastA2E", location, paths["model"], args.epochs)
    save_record(training_csv, record)
    print(pd.Series(record).round(2).to_string())

########################################################################################################################
# 2) Time A2E on the timing locations (same hardware and settings, a few epochs)
########################################################################################################################
for location in args.timing_locations:
    if location in load_records(a2e_timing_csv).index:
        print(f"Location {location}: A2E already timed, skipping")
        continue
    print(f"\n Timing A2E on Location {location} for {args.a2e_timing_epochs} epochs")
    with tempfile.TemporaryDirectory() as directory:
        record = train("A2E", location, Path(directory) / "A2E_timing.keras", args.a2e_timing_epochs)
    save_record(a2e_timing_csv, record)
    print(pd.Series(record).round(2).to_string())

########################################################################################################################
# 3) Evaluate FastA2E like experiments/evaluation/A2E/a2e_metrics.py
########################################################################################################################
df = pd.read_csv(METRICS_EVALUATION_CSV, index_col=0, header=[0, 1])
evaluated = []
for location in locations:
    paths = fast_paths(location)
    if not paths["model"].exists():
        print(f"Location {location}: no FastA2E model, skipping evaluation")
        continue
    print(f"Evaluating Location {location}")
    config = a2e.io.ModelConfig.load_from_json(paths["config"])
    api = a2e.io.Api(config)
    locational_forecasts, locational_observations = location_data(location, config)

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
    t_start = -test_size - config.foresight if config.foresight else -test_size
    t_end = -config.foresight if config.foresight else None
    y_t = locational_observations[t_start:t_end, 0]                    # Observation of time t
    y_h = locational_observations[config.seq_len - 1:-test_size, 0]    # Observation archive aligned to hist embeddings

    # Embed, split and retrieve top-k ensemble
    embeddings = api.embed(locational_forecasts, model_path=paths["model"], verbose=False)
    ensemble, weights = api.retrieve(embeddings[-test_size:], embeddings[:-test_size], y_h, k=100)

    computation = a2e.metrics.computation
    values = {
        "Bias": computation.compute_bias(ensemble, weights, y_t),
        "RMSE": computation.compute_rmse(ensemble, weights, y_t),
        "CRPS": -computation.compute_crps(ensemble, weights, y_t),
        "SCRPS": -computation.compute_scrps(ensemble, weights, y_t),
    }
    for metric, value in values.items():
        df.loc[f"Location_{location}", (metric, "FastA2E")] = float(tf.reduce_mean(value))
    evaluated.append(f"Location_{location}")
    row = df.loc[f"Location_{location}"]
    print(f"Location {location} - " + ", ".join(f"{m}: {row[(m, 'FastA2E')]:.4f}" for m in METRICS))

# Keep the column layout: metrics grouped, FastA2E after the other models
models = list(dict.fromkeys(model for _, model in df.columns))
df = df[[(m, model) for m in METRICS for model in models if (m, model) in df.columns]]
df.to_csv(metrics_csv_out)
print(f"\nMetrics saved to {metrics_csv_out}")

########################################################################################################################
# 4) Comparison: model performance and training time
########################################################################################################################
if not evaluated:
    raise SystemExit("No FastA2E model evaluated, nothing to compare.")
results = df.loc[evaluated]
n = len(evaluated)

# Performance: means and the number of locations where FastA2E is better (|Bias|, lower is better for all metrics)
means, wins = {}, {}
for metric in METRICS:
    score = results[metric].abs() if metric == "Bias" else results[metric]
    name = metric if metric != "Bias" else "|Bias|"
    means[name] = {f"{model} mean": score[model].mean() for model in ["AnEn", "A2E", "FastA2E"]}
    wins[name] = {f"FastA2E better than {model}": f"{int((score['FastA2E'] < score[model]).sum())}/{n}"
                  for model in ["A2E", "AnEn"]}
performance = pd.concat([pd.DataFrame(means).T.round(4), pd.DataFrame(wins).T], axis=1)
performance.to_csv(results_dir / "fasta2e_performance.csv")
print(f"\nModel performance over {n} locations (A2E and AnEn: paper results)\n{performance.to_string()}")

# Training time: A2E time per epoch measured on the timing locations, FastA2E on every location
training = load_records(training_csv).reindex([int(l.split("_")[1]) for l in evaluated]).dropna()
a2e_timing = load_records(a2e_timing_csv)
if len(training) and len(a2e_timing):
    timed = a2e_timing.index.intersection(training.index)
    a2e_epoch = a2e_timing["seconds per epoch"].mean()
    fast_epoch = training["seconds per epoch"].mean()
    speedup = (a2e_timing.loc[timed, "seconds per epoch"] / training.loc[timed, "seconds per epoch"]).mean() \
        if len(timed) else a2e_epoch / fast_epoch
    fast_hours = training["training minutes"].sum() / 60
    a2e_hours = a2e_epoch * training["epochs trained"].sum() / 3600   # same epochs as FastA2E
    timing = pd.Series({
        "A2E seconds per epoch": a2e_epoch,
        "FastA2E seconds per epoch": fast_epoch,
        "speedup per epoch": speedup,
        "FastA2E training hours (measured)": fast_hours,
        "A2E training hours (estimated, same epochs)": a2e_hours,
        "locations": len(training),
        "mean epochs trained": training["epochs trained"].mean(),
    })
    timing.to_csv(results_dir / "fasta2e_training_time.csv", header=["value"])
    print(f"\nTraining time (A2E timed on locations {list(a2e_timing.index)}, "
          f"{args.a2e_timing_epochs} epochs each)\n{timing.round(2).to_string()}")

    # Figure: CRPS per location (A2E paper vs FastA2E) and training time per location
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axis = axes[0]
    low = min(results[("CRPS", "A2E")].min(), results[("CRPS", "FastA2E")].min()) * 0.97
    high = max(results[("CRPS", "A2E")].max(), results[("CRPS", "FastA2E")].max()) * 1.03
    axis.plot([low, high], [low, high], color="#8a8a85", linewidth=1, zorder=1)
    axis.scatter(results[("CRPS", "A2E")], results[("CRPS", "FastA2E")], s=36, color=COLORS["FastA2E"],
                 edgecolor="white", linewidth=1, zorder=2)
    better = int((results[("CRPS", "FastA2E")] < results[("CRPS", "A2E")]).sum())
    axis.text(0.96, 0.04, f"below the diagonal: FastA2E better\n{better} of {n} locations",
              transform=axis.transAxes, ha="right", va="bottom", color="#3d3d3a")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("A2E CRPS (paper)")
    axis.set_ylabel("FastA2E CRPS")
    axis.set_title("CRPS per location")

    axis = axes[1]
    per_location = {
        "A2E (estimated)": a2e_hours * 60 / len(training),
        "FastA2E": fast_hours * 60 / len(training),
    }
    bars = axis.bar(list(per_location), list(per_location.values()), width=0.5,
                    color=[COLORS["A2E"], COLORS["FastA2E"]])
    for bar, value in zip(bars, per_location.values()):
        axis.annotate(f"{value:.1f} min", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom", color="#3d3d3a")
    axis.set_ylim(0, max(per_location.values()) * 1.12)
    axis.set_ylabel("Training minutes per location")
    axis.set_title(f"Training time ({speedup:.1f}x faster per epoch)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.set_axisbelow(True)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(results_dir / "fasta2e_comparison.png", dpi=300)
    plt.close(figure)
    print(f"\nSaved fasta2e_performance.csv, fasta2e_training_time.csv and fasta2e_comparison.png to {results_dir}")
else:
    print("\nNo training records, skipping the training time comparison.")
