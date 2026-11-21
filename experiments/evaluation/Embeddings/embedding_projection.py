"""
Embedding space and analog case study of one location.

1. 2-D projection of the embeddings of all windows (archive and validation period), L2-normalized as in the cosine
   similarity: PCA of all windows and t-SNE (cosine metric) of a random subset of N_TSNE windows. Coloured by
   season of the target time, observed wind speed at the target time and hour of day (UTC).
2. Case study for one query: its input window, the five analogs with the largest weights of the model and of AnEn
   (forecast and observations on the relative time axis of the window) and both ensembles with the observed value.
   Default query: the validation sample with the highest observed wind speed; another one via its target time.

Run from the repository root (model type A2E, CA2E or SA2E; location; optional target time, UTC):
    .venv/bin/python -m experiments.evaluation.Embeddings.embedding_projection A2E 0
    .venv/bin/python -m experiments.evaluation.Embeddings.embedding_projection A2E 0 "2025-02-14 12:00"

Results in evaluation/Embeddings/:
    <model>_Location<i>_embedding_projection.png
    <model>_Location<i>_case_<YYYYmmdd-HH>.png
    <model>_Location<i>_case_<YYYYmmdd-HH>.csv   (analogs of both methods: rank, target time, weight, value)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ..common import Data, ModelSpec, ModelEnsembles
from .analogs import model_analogs, anen_analogs
from ...project_paths import EVALUATION_DIR

N_TSNE = 5000
N_SHOWN = 5
SEED = 42
OUTPUT_DIR = EVALUATION_DIR / "Embeddings"
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_COLORS = {"DJF": "tab:blue", "MAM": "tab:green", "JJA": "tab:red", "SON": "tab:orange"}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "A2E"
LOCATION = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CASE_TIME = sys.argv[3] if len(sys.argv) > 3 else None
assert MODEL in ("A2E", "CA2E", "SA2E"), f"Model type must be A2E, CA2E or SA2E, not {MODEL}"
SPEC = ModelSpec(MODEL, "cosine_similarity")
PREFIX = f"{MODEL}_Location{LOCATION}"


def projection(embeddings, target_time, data, observations):
    """PCA and t-SNE of the normalized embeddings, three colourings each."""
    z = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    time = data.time_index[target_time]
    colourings = [
        ("Season", np.array([SEASON_COLORS[SEASONS[m]] for m in time.month]), None),
        ("Observed wind speed (m/s)", observations[target_time], "viridis"),
        ("Hour of day (UTC)", time.hour.to_numpy(), "twilight"),
    ]
    rng = np.random.default_rng(SEED)
    subset = np.sort(rng.choice(len(z), size=min(N_TSNE, len(z)), replace=False))
    pca = PCA(n_components=2, random_state=SEED).fit(z)
    coordinates = {
        f"PCA ({100 * pca.explained_variance_ratio_.sum():.0f} % variance)": (pca.transform(z), slice(None)),
        f"t-SNE ({len(subset)} windows)": (TSNE(n_components=2, metric="cosine", init="pca", random_state=SEED)
                                           .fit_transform(z[subset]), subset),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for row, (title, (xy, index)) in enumerate(coordinates.items()):
        for col, (label, colour, cmap) in enumerate(colourings):
            ax = axes[row, col]
            c = colour[index]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=1, alpha=0.5, rasterized=True)
            if cmap is not None:
                fig.colorbar(sc, ax=ax, label=label)
            ax.set_title(f"{title}: {label.split(' (')[0].lower()}")
            ax.set_xticks([])
            ax.set_yticks([])
    for season, colour in SEASON_COLORS.items():
        axes[0, 0].scatter([], [], c=colour, s=12, label=season)
    axes[0, 0].legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / f"{PREFIX}_embedding_projection.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def window(series, indices):
    """series[indices] with NaN for indices before the start of the data."""
    return np.where(indices >= 0, series[np.clip(indices, 0, None)], np.nan)


def case_study(model, anen, config, data, forecasts, observations):
    """Figure and table of the analogs of one query."""
    if CASE_TIME is None:
        q = int(np.argmax(model.y_t))
    else:
        target = pd.Timestamp(CASE_TIME, tz=data.time_index.tz)
        matches = np.flatnonzero(data.time_index[model.query_time] == target)
        assert matches.size, f"{CASE_TIME} is not a target time of the validation period"
        q = int(matches[0])
    t = int(model.query_time[q])
    stamp = data.time_index[t].strftime("%Y%m%d-%H")

    # Relative time axis of the window: forecasts t - seq_len + 2 ... t + foresight, observations up to t - ttt
    lead_x = np.arange(-(config.seq_len - 1 - config.foresight), config.foresight + 1)
    lead_y = np.arange(lead_x[0] - config.foresight - config.time_to_target, -config.time_to_target + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [2, 2, 1]})
    table = []
    for ax, (name, analogs) in zip(axes[:2], [(MODEL, model), ("AnEn", anen)]):
        order = np.argsort(-analogs.weights[q], kind="stable")
        ax.plot(lead_x, forecasts[t + lead_x], color="black", lw=2, label="query forecast")
        ax.plot(lead_y, observations[t + lead_y], color="black", lw=2, ls=":", label="query observations")
        ax.scatter([0], [observations[t]], color="black", marker="*", s=120, zorder=5, label="y_t")
        for j in order[:N_SHOWN]:
            h = int(analogs.analog_time[q, j])
            line, = ax.plot(lead_x, window(forecasts, h + lead_x), lw=1, alpha=0.8,
                            label=f"{data.time_index[h]:%Y-%m-%d %H}h, w={analogs.weights[q, j]:.3f}")
            ax.plot(lead_y, window(observations, h + lead_y), lw=1, ls=":", color=line.get_color(), alpha=0.8)
        for rank, j in enumerate(order):
            h = int(analogs.analog_time[q, j])
            table.append({"Method": name, "Rank": rank + 1, "Target_time": data.time_index[h],
                          "Weight": analogs.weights[q, j], "Value": analogs.values[q, j]})
        if name == "AnEn":
            ax.axvspan(-1, 1, color="grey", alpha=0.15, label="AnEn window t ± 1")
        ax.set_title(f"{name}: query {data.time_index[t]:%Y-%m-%d %H}h UTC and top {N_SHOWN} analogs")
        ax.set_xlabel("Hours relative to the target time t")
        ax.set_ylabel("Wind speed (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")

    edges = np.linspace(min(model.values[q].min(), anen.values[q].min()),
                        max(model.values[q].max(), anen.values[q].max()), 25)
    axes[2].hist(model.values[q], bins=edges, weights=model.weights[q], alpha=0.6, label=MODEL)
    axes[2].hist(anen.values[q], bins=edges, weights=anen.weights[q], alpha=0.6, label="AnEn")
    axes[2].axvline(observations[t], color="black", lw=2, label="y_t")
    axes[2].set_title("Ensembles (member weights)")
    axes[2].set_xlabel("Wind speed (m/s)")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR / f"{PREFIX}_case_{stamp}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    pd.DataFrame(table).to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved {path} and {path.with_suffix('.csv').name}")


data = Data()
ensembles = ModelEnsembles(SPEC, data)
assert ensembles.available(LOCATION), f"No trained model at {SPEC.paths(LOCATION)['model']}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
forecasts = np.asarray(data.forecasts[:, LOCATION, 0])
observations = np.asarray(data.observations[:, LOCATION, 0])

model, embeddings, config = model_analogs(ensembles, data, LOCATION)
anen = anen_analogs(data, LOCATION)

# Target time of every embedding: window j ends in features row j + seq_len - 1
target_time = np.arange(embeddings.shape[0]) + config.seq_len - 1 + config.time_to_target
projection(embeddings, target_time, data, observations)
case_study(model, anen, config, data, forecasts, observations)
