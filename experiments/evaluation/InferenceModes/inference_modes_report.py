"""
Summary of the inference modes and of the attention during training.

Inference (from inference_modes_metrics.py):
    inference_modes_summary.csv: mean over the locations per model, mode and k (Bias, RMSE, CRPS, SCRPS, ESS)
    attention_summary.csv:       learned temperature g, archive size and weight share outside the top 100 per model
                                 (mean, median, min, max over the locations)
    inference_modes_k.png:       CRPS and SCRPS over k, softmax and uniform top-k, per model
Training (from the history JSONs <model_name>_history.json of the main models):
    training_attention_summary.csv: per model at the saved (best) epoch, mean over the locations: ESS of the full
                                    softmax on training/validation batches, g, gradient and weight share of the
                                    probe batch outside the top 100
    training_attention.png:         the same quantities per epoch (mean over the locations that trained that long)

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.InferenceModes.inference_modes_report
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..common import ModelSpec
from ...project_paths import EVALUATION_DIR

MODELS = ["A2E", "CA2E", "SA2E"]
N_LOCATIONS = 82
OUTPUT_DIR = EVALUATION_DIR / "InferenceModes"
TRAINING_KEYS = ["ess", "val_ess", "g", "grad_share_outside_top100", "weight_share_outside_top100"]


def inference_summary():
    frames, attention = [], []
    for model in MODELS:
        path = OUTPUT_DIR / f"inference_modes_{model}.csv"
        if not path.exists():
            print(f"{model}: {path.name} missing, skipped")
            continue
        df = pd.read_csv(path, dtype={"k": str})
        summary = df.groupby(["Mode", "k"], sort=False)[["Bias", "RMSE", "CRPS", "SCRPS", "ESS"]].mean()
        summary.insert(0, "n_locations", df.groupby(["Mode", "k"], sort=False)["Location"].nunique())
        frames.append(summary.assign(Model=model).set_index("Model", append=True).reorder_levels([2, 0, 1]))

        att = pd.read_csv(OUTPUT_DIR / f"attention_{model}.csv").drop(columns="Location")
        attention.append(att.agg(["mean", "median", "min", "max"]).assign(Model=model)
                         .set_index("Model", append=True).reorder_levels([1, 0]))
    if not frames:
        return
    summary = pd.concat(frames)
    summary.to_csv(OUTPUT_DIR / "inference_modes_summary.csv")
    pd.concat(attention).to_csv(OUTPUT_DIR / "attention_summary.csv")
    print(summary.round(4).to_string())
    print(pd.concat(attention).round(3).to_string())

    # CRPS and SCRPS over k
    models = summary.index.get_level_values("Model").unique()
    fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 6), squeeze=False, sharex=True)
    for col, model in enumerate(models):
        n_all = float(pd.read_csv(OUTPUT_DIR / f"attention_{model}.csv")["n_archive"].mean())
        for row, score in enumerate(["CRPS", "SCRPS"]):
            ax = axes[row, col]
            for mode, marker in (("softmax", "o"), ("uniform", "s")):
                d = summary.loc[(model, mode)]
                k = [n_all if i == "all" else float(i) for i in d.index]
                ax.plot(k, d[score], marker=marker, label=f"{mode} top-k")
            ax.axvline(100, color="grey", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_title(f"{model}" if row == 0 else "")
            ax.set_ylabel(f"{score} (m/s)" if score == "CRPS" else score)
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("k (rightmost point: whole archive)")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inference_modes_k.png", dpi=300)
    plt.close()


def training_summary():
    rows, curves = [], {}
    for model in MODELS:
        spec = ModelSpec(model, "cosine_similarity")
        locations = [0] if model == "CA2E" else range(N_LOCATIONS)
        for location in locations:
            path = spec.paths(location)["history"]
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            history = pd.DataFrame({k: record["history"].get(k) for k in TRAINING_KEYS}, dtype=float)
            history.index = np.arange(record["initial_epoch"] + 1, record["initial_epoch"] + 1 + len(history))
            curves.setdefault(model, []).append(history)
            best = history.loc[record["best_epoch"]] if record["best_epoch"] in history.index else history.iloc[-1]
            rows.append({"Model": model, "Location": location, "best_epoch": record["best_epoch"],
                         "epochs_trained": record["epochs_trained"], "stopped_early": record["stopped_early"],
                         **best.to_dict()})
    if not rows:
        print("No history JSONs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "training_attention_per_location.csv", index=False)
    summary = df.drop(columns="Location").groupby("Model", sort=False).mean()
    summary.insert(0, "n_models", df.groupby("Model", sort=False).size())
    summary.to_csv(OUTPUT_DIR / "training_attention_summary.csv")
    print(summary.round(3).to_string())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for model, histories in curves.items():
        mean = pd.concat(histories, axis=1, keys=range(len(histories))).T.groupby(level=1).mean().T
        axes[0].plot(mean.index, mean["ess"], label=f"{model} training")
        axes[0].plot(mean.index, mean["val_ess"], ls="--", label=f"{model} validation")
        axes[1].plot(mean.index, mean["g"], label=model)
        axes[2].plot(mean.index, mean["grad_share_outside_top100"], label=model)
        axes[3].plot(mean.index, mean["weight_share_outside_top100"], label=model)
    for ax, title in zip(axes, ["ESS of the full softmax", "Temperature g",
                                "Gradient share outside top 100", "Weight share outside top 100"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_attention.png", dpi=300)
    plt.close()


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
inference_summary()
training_summary()
print(f"\nResults in {OUTPUT_DIR}")
