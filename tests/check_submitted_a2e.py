"""
Check of the evaluation helpers with the submitted A2E model of location 0 (section T.4 of the retraining spec).

The model is read from the tag paper-submission (seq_len 80, trained with the shifted archive). With the corrected
archive (a2e.io.evaluation_split) the CRPS must reproduce the value of the user's fixed evaluation,
0.2916 (evaluation/A2E_Fixed/metrics_A2E_Fixed.csv, row A2E,fixed); with the old alignment
(y_h = targets[seq_len-1 : -test_size]) the value of Table 2, 0.2976. Deviations up to 1e-3 are possible because
Api.embed now uses float32 instead of float16 (F6).

Run from the repository root:
    .venv/bin/python tests/check_submitted_a2e.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import a2e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.data.data_loader import WindDataLoader

TAG = "paper-submission"
MODEL = "experiments/Trained_Models/A2E/Location0/A2E_cosine_similarity_Location0"
EXPECTED = {"fixed": 0.2916, "shifted": 0.2976}
TOLERANCE = 1e-3


def from_tag(path, target):
    with open(target, "wb") as f:
        subprocess.run(["git", "show", f"{TAG}:{path}"], cwd=PROJECT_ROOT, stdout=f, check=True)


def crps(api, embeddings, query, archive, y_t, y_h):
    ensemble, weights = api.retrieve(embeddings[query], embeddings[archive], y_h, k=100)
    return float(tf.reduce_mean(-a2e.metrics.computation.compute_crps(ensemble, weights, y_t)))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        model_path, config_path = Path(tmp) / "model.keras", Path(tmp) / "model_config.json"
        from_tag(f"{MODEL}.keras", model_path)
        from_tag(f"{MODEL}_config.json", config_path)
        config = a2e.io.ModelConfig.load_from_json(config_path)
        api = a2e.io.Api(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # receptive field warning for seq_len 80
            api.model = tf.keras.models.load_model(model_path)

    loader = WindDataLoader()
    observations, forecasts = loader.get_all_data(obs_components=["total"], fcst_components=["total"])
    features, targets = a2e.io.build_features(forecasts[:, 0], observations[:, 0], config)
    test_size = a2e.io.evaluation_test_size(forecasts.shape[0], lookback=config.lookback)
    split = a2e.io.evaluation_split(features, targets, config, test_size)
    embeddings = api.embed(features, verbose=False)

    results = {"fixed": crps(api, embeddings, split.query, split.archive, split.y_t, split.y_h)}
    # Old alignment of the submitted evaluation scripts
    old_y_h = targets[config.seq_len - 1:-test_size, 0]
    results["shifted"] = crps(api, embeddings, split.query, slice(0, -test_size), split.y_t, old_y_h)

    ok = True
    for key, value in results.items():
        deviation = abs(value - EXPECTED[key])
        ok &= deviation <= TOLERANCE
        print(f"CRPS location 0, archive {key}: {value:.4f} (expected {EXPECTED[key]:.4f}, deviation {deviation:.1e})")
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
