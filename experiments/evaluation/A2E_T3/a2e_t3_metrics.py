"""
Evaluation of the ablation A2E with T = 3 (Trained_Models/A2E_T3), identical to A2E/a2e_metrics.py:
same validation samples, archive and inference (softmax-weighted top-k, k = 100).

RMSE is the RMSE of the ensemble mean (paper equation); RMSE_old is compute_rmse as used for the original Table 2.
Results: evaluation/A2E_T3/metrics_A2E_T3.csv (one row per location).

Run from the repository root:
    python -m experiments.evaluation.A2E_T3.a2e_t3_metrics
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
import a2e
from ...data import WindDataLoader
from ...project_paths import EVALUATION_DIR, model_paths

OUT_CSV = EVALUATION_DIR / "A2E_T3" / "metrics_A2E_T3.csv"

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)

rows = []
for location in range(forecasts.shape[1]):
    model_name = f"A2E_T3_cosine_similarity_Location{location}"
    paths = model_paths("A2E_T3", model_name, location)
    if not paths["config"].exists():
        print(f"Location {location}: no trained model, skipped")
        continue

    config = a2e.io.ModelConfig.load_from_json(paths["config"])
    api = a2e.io.Api(config)

    # Set up Data [time, location, variable]
    locational_observations = observations[:, location, :]
    locational_forecasts = forecasts[:, location, :]

    # Add observation as causal feature
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
    t_start = -test_size - config.foresight if config.foresight else -test_size
    t_end = -config.foresight if config.foresight else None
    h_start = config.seq_len - 1
    h_end = -test_size

    y_t = locational_observations[t_start:t_end, 0]    # Observation of time t
    y_h = locational_observations[h_start:h_end, 0]    # Observation archive aligned to hist embeddings

    # Embeddings and top-k ensemble
    embeddings = api.embed(locational_forecasts, model_path=paths["model"], verbose=False)
    z_t = embeddings[-test_size:]
    z_h = embeddings[:-test_size]
    ensemble, weights = api.retrieve(z_t, z_h, y_h, k=100)

    bias = a2e.metrics.computation.compute_bias(ensemble, weights, y_t).numpy().ravel()
    rows.append(dict(
        location=f"Location_{location}",
        Bias=bias.mean(),
        RMSE=np.sqrt(np.mean(bias ** 2)),
        RMSE_old=tf.reduce_mean(a2e.metrics.computation.compute_rmse(ensemble, weights, y_t)).numpy(),
        CRPS=tf.reduce_mean(-a2e.metrics.computation.compute_crps(ensemble, weights, y_t)).numpy(),
        SCRPS=tf.reduce_mean(-a2e.metrics.computation.compute_scrps(ensemble, weights, y_t)).numpy(),
    ))
    print(f"Location {location} - " + ", ".join(f"{k}: {v:.4f}" for k, v in rows[-1].items() if k != "location"))

df = pd.DataFrame(rows).set_index("location")
OUT_CSV.parent.mkdir(exist_ok=True)
df.to_csv(OUT_CSV)
print(f"\n{len(df)} locations evaluated. Results saved to {OUT_CSV}")
print(df.mean().round(4).to_string())
