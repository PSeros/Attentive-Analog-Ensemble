"""
Inference modes and ensemble size k of A2E, CA2E and SA2E.

Queries and static archive as in the evaluation of Table 2 (a2e.io.evaluation_split). The attention logits
s = g * cos(z_t, z_h) of each query are sorted once; every mode uses a prefix of this order:
    softmax top-k: the k members with the largest logits, softmax over their logits (Table 2: k = 100),
    uniform top-k: the same k members with equal weights 1/k,
    k = all:       the whole archive (softmax: full softmax as in training; uniform: all archive values).
k in {10, 25, 50, 100, 200, 500, 1000, all}.
Scores: bias, RMSE of the ensemble mean, CRPS, SCRPS (m/s) and the effective ensemble size ESS = 1 / sum_j w_j^2.
Per location also the learned temperature g, the archive size and the weight share of the full softmax outside
the top 100.

Run from the repository root (model type A2E, CA2E or SA2E):
    .venv/bin/python -m experiments.evaluation.InferenceModes.inference_modes_metrics A2E          # all locations
    .venv/bin/python -m experiments.evaluation.InferenceModes.inference_modes_metrics SA2E 0 41    # locations 0 ... 40

Results (rows of finished locations are replaced, other rows are kept):
    evaluation/InferenceModes/inference_modes_<model>.csv  (Location, Mode, k, Bias, RMSE, CRPS, SCRPS, ESS)
    evaluation/InferenceModes/attention_<model>.csv        (Location, g, n_archive, weight_share_outside_top100)
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import a2e
from ..common import Data, ModelSpec, ModelEnsembles, location_range
from ...project_paths import EVALUATION_DIR

K_VALUES = [10, 25, 50, 100, 200, 500, 1000, None]  # None: whole archive
TOP_K = 100                                         # weight share of the full softmax outside the top 100
BATCH_SIZE = 64
OUTPUT_DIR = EVALUATION_DIR / "InferenceModes"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "A2E"
assert MODEL in ("A2E", "CA2E", "SA2E"), f"Model type must be A2E, CA2E or SA2E, not {MODEL}"
SPEC = ModelSpec(MODEL, "cosine_similarity")
MODES_CSV = OUTPUT_DIR / f"inference_modes_{MODEL}.csv"
ATTENTION_CSV = OUTPUT_DIR / f"attention_{MODEL}.csv"


def mode_scores(attention, query, archive, y_h, y_t):
    """Per-sample scores of all modes and k; returns ({(mode, k): {score: array}}, weight share outside top 100)."""
    n_archive = int(archive.shape[0])
    per_sample = {}
    outside_top = []
    for start in range(0, int(query.shape[0]), BATCH_SIZE):
        q = query[start:start + BATCH_SIZE]
        y = y_t[start:start + BATCH_SIZE]
        s = attention.scores(q[:, None, :], archive[None, :, :])        # (batch, n_archive)
        s_sorted, order = tf.math.top_k(s, k=n_archive)                 # descending, as in CrossAttention.call
        v_sorted = tf.gather(y_h, order)
        outside_top.append(1.0 - tf.reduce_sum(tf.nn.softmax(s_sorted)[:, :TOP_K], axis=-1).numpy())

        for k in K_VALUES:
            n = k or n_archive
            s_k, v_k = s_sorted[:, :n], v_sorted[:, :n]
            for mode, w in (("softmax", tf.nn.softmax(s_k)), ("uniform", tf.fill(tf.shape(s_k), 1.0 / n))):
                values = {
                    "bias": a2e.metrics.computation.compute_bias(v_k, w, y).numpy()[:, 0],
                    "crps": -a2e.metrics.computation.compute_crps(v_k, w, y).numpy(),
                    "scrps": -a2e.metrics.computation.compute_scrps(v_k, w, y).numpy(),
                    "ess": a2e.metrics.computation.compute_ess(w).numpy(),
                }
                key = (mode, "all" if k is None else str(k))
                for name, value in values.items():
                    per_sample.setdefault(key, {}).setdefault(name, []).append(value)
    per_sample = {key: {name: np.concatenate(v) for name, v in d.items()} for key, d in per_sample.items()}
    return per_sample, float(np.mean(np.concatenate(outside_top)))


def update_csv(path, rows, locations):
    """Replaces the rows of `locations` in the CSV at `path` (created if missing)."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype={"k": str})
        new = pd.concat([old[~old["Location"].isin(locations)], new], ignore_index=True)
    new = new.sort_values("Location", kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


# Load Data
data = Data()
ensembles = ModelEnsembles(SPEC, data)

for location in location_range(data.n_locations, arg=2):
    if not ensembles.available(location):
        print(f"Location {location}: no trained model at {SPEC.paths(location)['model']}, skipped")
        continue

    # Config and model, split (queries, archive), embeddings
    api = ensembles.api(location)
    features, targets = data.inputs(api.config, location)
    split = a2e.io.evaluation_split(features, targets, api.config, data.test_size)
    embeddings = api.embed(features, verbose=False, location=location if MODEL == "CA2E" else None)
    attention = api.model.cross_attention

    # Scores of all modes
    per_sample, outside_top = mode_scores(attention, embeddings[split.query], embeddings[split.archive],
                                          split.y_h, split.y_t)
    rows = [{"Location": location, "Mode": mode, "k": k,
             "Bias": float(np.mean(s["bias"])), "RMSE": float(np.sqrt(np.mean(s["bias"] ** 2))),
             "CRPS": float(np.mean(s["crps"])), "SCRPS": float(np.mean(s["scrps"])), "ESS": float(np.mean(s["ess"]))}
            for (mode, k), s in per_sample.items()]
    attention_row = {"Location": location, "g": float(attention.g.numpy()),
                     "n_archive": int(split.y_h.shape[0]), f"weight_share_outside_top{TOP_K}": outside_top}

    update_csv(MODES_CSV, rows, [location])
    update_csv(ATTENTION_CSV, [attention_row], [location])

    table = pd.DataFrame(rows).set_index(["Mode", "k"])
    print(f"Location {location} - g: {attention_row['g']:.2f}, archive: {attention_row['n_archive']}, "
          f"weight share outside top {TOP_K}: {outside_top:.3f}")
    print(table.loc[:, ["CRPS", "SCRPS", "ESS"]].round(4).to_string())

print(f"\nResults: {MODES_CSV}, {ATTENTION_CSV}")
