"""
Smoke test of the training and evaluation scripts before the retraining (section T of the retraining spec).

Every training script of S1 runs for location 0 (CA2E: one model) with 2 epochs of a few steps and a small batch; all
outputs go to a temporary directory (Trained_Models, evaluation). Checks per training script:
- model, config, history JSON and training plot are written
- the history has ess, val_ess, g (None for Euclidean distance), lr and the gradient/weight share outside top-100
- the saved model has the weights of best_epoch
- no seq_len / receptive field warning
Then the evaluation scripts of S2 run on the smoke models (location 0): the CSV row and the per-sample npz exist.

Each script runs in its own process. If systemd-run is available, the process runs in a user scope with a memory
limit (--memory, default 7G), so that a script that needs too much memory is stopped alone. The peak memory (RSS)
of every script is reported.

Run from the repository root (takes several minutes on a CPU):
    .venv/bin/python tests/smoke_run.py
    .venv/bin/python tests/smoke_run.py --keep --memory 10G     # keep the outputs, other memory limit
    .venv/bin/python tests/smoke_run.py --only training          # training | evaluation | auxiliary
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import json
import resource
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EPOCHS = 2
TRAIN_STEPS = 4
TEST_STEPS = 2
BATCH_SIZE = 8
BATCH_SIZE_SA2E = 2  # spatial encoder over 17520 x 6 archive steps: about 1 GB of activations per sample in training

TRAINING = [
    ("experiments.Trained_Models.A2E.A2E_Train", "A2E", "A2E_cosine_similarity_Location0", 0),
    ("experiments.Trained_Models.A2E.A2E_euclidean_distance_Train", "A2E", "A2E_euclidean_distance_Location0", 0),
    ("experiments.Trained_Models.A2E.A2E_T3_Train", "A2E", "A2E_cosine_similarity_T3_Location0", 0),
    ("experiments.Trained_Models.A2E.A2E_no_obs_Train", "A2E", "A2E_cosine_similarity_no_obs_Location0", 0),
    ("experiments.Trained_Models.SA2E.SA2E_Train", "SA2E", "SA2E_cosine_similarity_Location0", 0),
    ("experiments.Trained_Models.CA2E.CA2E_Train", "CA2E", "CA2E_cosine_similarity", None),
    ("experiments.Trained_Models.CA2E.CA2E_euclidean_distance_Train", "CA2E", "CA2E_euclidean_distance", None),
]

EVALUATION = [
    # module, csv, column, per-sample file
    ("experiments.evaluation.AnEn.anen_metrics", "main", ("CRPS", "AnEn"), "AnEn/per_sample/AnEn_Location0.npz"),
    ("experiments.evaluation.A2E.a2e_metrics", "main", ("CRPS", "A2E"),
     "A2E/per_sample/A2E_cosine_similarity_Location0.npz"),
    ("experiments.evaluation.CA2E.ca2e_metrics", "main", ("CRPS", "CA2E"),
     "CA2E/per_sample/CA2E_cosine_similarity_Location0.npz"),
    ("experiments.evaluation.SA2E.sa2e_metrics", "main", ("CRPS", "SA2E"),
     "SA2E/per_sample/SA2E_cosine_similarity_Location0.npz"),
    ("experiments.evaluation.AblationStudy.a2e_euclidean_distance_metrics", "ablation",
     ("CRPS", "A2E", "euclidean_distance"), "A2E/per_sample/A2E_euclidean_distance_Location0.npz"),
    ("experiments.evaluation.AblationStudy.ca2e_euclidean_distance_metrics", "ablation",
     ("CRPS", "CA2E", "euclidean_distance"), "CA2E/per_sample/CA2E_euclidean_distance_Location0.npz"),
    ("experiments.evaluation.AblationStudy.a2e_t3_metrics", "ablation",
     ("CRPS", "A2E", "cosine_similarity_T3"), "A2E/per_sample/A2E_cosine_similarity_T3_Location0.npz"),
    ("experiments.evaluation.AblationStudy.a2e_no_obs_metrics", "ablation",
     ("CRPS", "A2E", "cosine_similarity_no_obs"), "A2E/per_sample/A2E_cosine_similarity_no_obs_Location0.npz"),
    ("experiments.evaluation.AblationStudy.anen_no_obs_metrics", "ablation",
     ("CRPS", "AnEn", "no_obs"), "AnEn/per_sample/AnEn_no_obs_Location0.npz"),
]

AUXILIARY = [
    "experiments.evaluation.A2E.a2e_rank_histogram",
    "experiments.evaluation.A2E.a2e_training-test-bias",
    "experiments.evaluation.A2E.a2e_comparison",
    "experiments.evaluation.CA2E.ca2e_comparison",
    "experiments.evaluation.SA2E.sa2e_rank_histogram",
    "experiments.evaluation.AnEn.anen_training-test-bias",
    "experiments.evaluation.values_comparison",
]


# ----------------------------------------------------------------------------------------------------------------------
# Child process: patches, then runs one script like `python -m module args`
# ----------------------------------------------------------------------------------------------------------------------
def child(output_dir: Path, module: str, args):
    # Redirect all outputs before any experiment module imports the paths
    import experiments.project_paths as project_paths
    project_paths.TRAINED_MODELS_DIR = output_dir / "Trained_Models"
    project_paths.EVALUATION_DIR = output_dir / "evaluation"
    project_paths.FIGURES_DIR = project_paths.EVALUATION_DIR / "figures"
    project_paths.ABLATION_STUDY_DIR = project_paths.EVALUATION_DIR / "AblationStudy"
    project_paths.METRICS_EVALUATION_CSV = project_paths.EVALUATION_DIR / "metrics_evaluation.csv"
    project_paths.ABLATION_METRICS_EVALUATION_CSV = project_paths.ABLATION_STUDY_DIR / "metrics_evaluation.csv"

    import numpy as np
    import tensorflow as tf
    import a2e
    from a2e.pipeline.base import BaseTrainingPipeline

    # Few steps per epoch, small batches, 2 epochs; trainable weights of every epoch for the best-epoch check
    BaseTrainingPipeline.get_train_steps = lambda self: TRAIN_STEPS
    BaseTrainingPipeline.get_test_steps = lambda self: TEST_STEPS

    class WeightRecorder(tf.keras.callbacks.Callback):
        def __init__(self, path):
            super().__init__()
            self.path = path
            self.weights = {}

        def on_epoch_end(self, epoch, logs=None):
            for i, w in enumerate(self.model.trainable_weights):
                self.weights[f"epoch{epoch + 1}_{i}"] = w.numpy()
            np.savez(self.path, **self.weights)

    train = a2e.io.Api.train

    def smoke_train(self, *a, **kwargs):
        kwargs["epochs"] = EPOCHS
        kwargs["batch_size"] = BATCH_SIZE_SA2E if self.config.model_type == "SA2E" else BATCH_SIZE
        recorder = WeightRecorder(os.path.splitext(kwargs["save_path"])[0] + "_smoke_weights.npz")
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [recorder]
        return train(self, *a, **kwargs)

    a2e.io.Api.train = smoke_train

    sys.argv = [module] + list(args)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
    rf = [str(w.message) for w in caught if "receptive field" in str(w.message)]
    (output_dir / "warnings.json").write_text(json.dumps(rf))
    print(f"PEAK_RSS_MB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}")


# ----------------------------------------------------------------------------------------------------------------------
# Parent process: runs the scripts and checks their outputs
# ----------------------------------------------------------------------------------------------------------------------
def run(output_dir: Path, module: str, args, memory: str, log_dir: Path):
    """Runs one script in a child process; returns (ok, peak memory in MB, runtime in s, log file)."""
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(output_dir), module, *args]
    if shutil.which("systemd-run"):
        command = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={memory}",
                   "-p", "MemorySwapMax=0", *command]
    log = log_dir / f"{module.split('.')[-1]}.log"
    start = time.time()
    with open(log, "w") as f:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
    peak = None
    for line in log.read_text().splitlines():
        if line.startswith("PEAK_RSS_MB"):
            peak = float(line.split()[1])
    return result.returncode == 0, peak, time.time() - start, log


def check_training(output_dir, model_type, name, location):
    import numpy as np
    import tensorflow as tf
    import a2e  # noqa: F401  (registers the custom objects)
    import experiments.project_paths as project_paths
    directory = output_dir / "Trained_Models" / model_type
    directory = directory / f"Location{location}" if location is not None else directory
    model, config, history = (directory / f"{name}{suffix}" for suffix in [".keras", "_config.json", "_history.json"])
    problems = [f"missing {p.name}" for p in (model, config, history) if not p.exists()]
    if not list(directory.glob(f"{name}_epoch_*_training.png")):
        problems.append("missing training plot")
    rf = json.loads((output_dir / "warnings.json").read_text())
    if rf:
        problems.append(f"receptive field warning: {rf[0][:80]}")
    if problems:
        return problems, ""
    record = json.loads(history.read_text())
    h = record["history"]
    for key in ["loss", "val_loss", "ess", "val_ess", "g", "lr",
                "grad_share_outside_top100", "weight_share_outside_top100"]:
        if len(h.get(key, [])) != EPOCHS:
            problems.append(f"history[{key}] has {len(h.get(key, []))} values")
    if ("euclidean" in name) != all(v is None for v in h.get("g", [])):
        problems.append(f"g: {h.get('g')}")
    best = int(np.argmin(h["val_loss"])) + 1
    if record["best_epoch"] != best or record["seed"] != 42 or record["epochs_trained"] != EPOCHS:
        problems.append(f"best_epoch {record['best_epoch']} (expected {best}), seed {record['seed']}")
    recorded = np.load(directory / f"{name}_smoke_weights.npz")
    saved = [w.numpy() for w in tf.keras.models.load_model(model).trainable_weights]
    if not all(np.array_equal(w, recorded[f"epoch{best}_{i}"]) for i, w in enumerate(saved)):
        problems.append("saved weights differ from the best epoch")
    info = (f"best epoch {best}, val_loss {[round(v, 4) for v in h['val_loss']]}, "
            f"ess {[round(v, 1) for v in h['ess']]}, g {h['g']}, "
            f"grad share {[round(v, 3) for v in h['grad_share_outside_top100']]}")
    return problems, info


def check_evaluation(output_dir, csv, column, npz):
    import numpy as np
    import pandas as pd
    path = output_dir / "evaluation" / ("metrics_evaluation.csv" if csv == "main" else
                                        "AblationStudy/metrics_evaluation.csv")
    problems, info = [], ""
    try:
        df = pd.read_csv(path, index_col=0, header=[0, 1] if csv == "main" else [0, 1, 2])
        value = df.loc["Location_0", column]
        info = f"{column} Location_0: {value:.4f}"
        if not np.isfinite(value):
            problems.append(f"{column} at Location_0 is {value}")
    except Exception as e:
        problems.append(f"CSV: {e}")
    npz_path = output_dir / "evaluation" / npz
    if not npz_path.exists():
        problems.append(f"missing {npz}")
    else:
        data = np.load(npz_path)
        if set(data.files) != {"time", "y", "ens_mean", "bias", "crps", "scrps", "pit"} or len(data["y"]) != 3720:
            problems.append(f"npz keys {data.files}, {len(data['y'])} samples")
        elif not (0 <= data["pit"].min() and data["pit"].max() <= 1):
            problems.append("PIT outside [0, 1]")
    return problems, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the output directory")
    parser.add_argument("--memory", default="7G", help="memory limit per script (systemd-run MemoryMax)")
    parser.add_argument("--only", choices=["training", "evaluation", "auxiliary"], action="append")
    parser.add_argument("--output", help="output directory (default: new temporary directory)")
    parser.add_argument("--match", help="run only the scripts whose module name contains this text")
    options = parser.parse_args()
    groups = options.only or ["training", "evaluation", "auxiliary"]

    output_dir = Path(options.output) if options.output else Path(tempfile.mkdtemp(prefix="a2e_smoke_"))
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir} (logs in {log_dir})", flush=True)

    steps = []
    if "training" in groups:
        steps += [("training", m, ["0", "1"] if loc is not None else [], (mt, name, loc)) for m, mt, name, loc in TRAINING]
    if "evaluation" in groups:
        steps += [("evaluation", m, ["0", "1"], (csv, col, npz)) for m, csv, col, npz in EVALUATION]
    if "auxiliary" in groups:
        steps += [("auxiliary", m, ["0", "1"], None) for m in AUXILIARY]

    if options.match:
        steps = [step for step in steps if options.match in step[1]]

    failures = []
    for kind, module, args, expected in steps:
        ok, peak, runtime, log = run(output_dir, module, args, options.memory, log_dir)
        problems, info = [], ""
        if not ok:
            problems.append(f"exit code != 0 (memory limit {options.memory}?), see {log}")
        elif kind == "training":
            problems, info = check_training(output_dir, *expected)
        elif kind == "evaluation":
            problems, info = check_evaluation(output_dir, *expected)
        peak_text = f"{peak:.0f} MB" if peak is not None else "n/a"
        print(f"{'OK    ' if not problems else 'FAILED'} {kind:10s} {module}  [{runtime:.0f} s, peak {peak_text}]"
              + (f"\n         {info}" if info else "")
              + (f"\n         {'; '.join(problems)}" if problems else ""), flush=True)
        if problems:
            failures.append(module)

    print("\n" + ("All scripts OK." if not failures else f"FAILED: {', '.join(failures)}"))
    if options.keep or failures:
        print(f"Outputs kept in {output_dir}")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    else:
        main()
