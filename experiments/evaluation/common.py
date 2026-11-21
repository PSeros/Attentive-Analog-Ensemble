"""
Shared steps of the evaluation scripts.

- Data: raw forecasts/observations with their time axis, the evaluation test size (Table 2: 3720 samples).
- Ensembles: A2E/CA2E/SA2E via a2e.io.build_features, a2e.io.evaluation_split, Api.embed and Api.retrieve
  (softmax top-k, k = 100); AnEn (window t ± 1, k = 100, predictor weights 1) with the same queries and the same
  archive end via a2e.io.anen_inputs.
- Scores per sample (bias, CRPS, SCRPS, ensemble mean, randomized PIT) and their means; RMSE is the RMSE of the
  ensemble mean, sqrt(mean((sum_i w_i x_i - y)^2)).
- Output: per-sample npz files (evaluation/<model type>/per_sample/<name>.npz) and the metric CSVs.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import a2e
from ..data import WindDataLoader
from ..project_paths import EVALUATION_DIR, model_paths

K = 100                 # ensemble members (softmax top-k)
TEST_FRACTION = 0.3     # validation fraction in training
LOOKBACK = 8760 * 2     # archive length in training (2 years)
METRICS = ["Bias", "RMSE", "CRPS", "SCRPS"]

# AnEn: window t ± 1 with the index convention of build_features (seq_len 3 = 2 * temporal_window + 1)
ANEN = SimpleNamespace(seq_len=3, foresight=1, time_to_target=2, d_vars=2)
ANEN_NO_OBS = SimpleNamespace(seq_len=3, foresight=1, time_to_target=2, d_vars=1)


def location_range(n_locations: int, arg: int = 1):
    """
    Locations from the optional command line arguments `first last` (end exclusive) at positions arg, arg + 1,
    default all.
    """
    first = int(sys.argv[arg]) if len(sys.argv) > arg else 0
    last = int(sys.argv[arg + 1]) if len(sys.argv) > arg + 1 else n_locations
    return range(first, last)


class Data:
    """Raw wind speed forecasts and observations [time, location, 1] with their time axis."""

    def __init__(self):
        self.loader = WindDataLoader()
        self.observations, self.forecasts = self.loader.get_all_data(
            obs_components=["total"],
            fcst_components=["total"]
        )
        self.time_index = self.loader.time_index
        self.n_locations = self.forecasts.shape[1]
        self._nearest = {}

    @property
    def test_size(self) -> int:
        return a2e.io.evaluation_test_size(self.forecasts.shape[0], LOOKBACK, TEST_FRACTION)

    def nearest_locations(self, d_loc: int) -> np.ndarray:
        if d_loc not in self._nearest:
            self._nearest[d_loc] = self.loader.find_nearest_locations(d_loc - 1)
        return self._nearest[d_loc]

    def inputs(self, config, location: int):
        """
        Features and targets of one location for a model config. SA2E: the location and its nearest neighbours,
        targets of the location. Without an observation channel if config.d_vars equals the forecast variables.
        """
        observation_inputs = config.d_vars > self.forecasts.shape[-1]
        if config.model_type == "SA2E":
            neighbours = self.nearest_locations(config.d_loc)[location]
            return a2e.io.build_features(self.forecasts[:, neighbours], self.observations[:, neighbours], config,
                                         target_location=0, observation_inputs=observation_inputs)
        return a2e.io.build_features(self.forecasts[:, location], self.observations[:, location], config,
                                     observation_inputs=observation_inputs)


@dataclass(frozen=True)
class ModelSpec:
    """A trained model: model type, similarity metric and ablation variant (None, "T3", "no_obs")."""
    model_type: str
    similarity_metric: str = "cosine_similarity"
    variant: Optional[str] = None

    @property
    def label(self) -> str:
        """Variant label of the ablation CSV, e.g. cosine_similarity_T3."""
        return self.similarity_metric + (f"_{self.variant}" if self.variant else "")

    def name(self, location: int) -> str:
        """Name of the evaluated model at a location (per-sample file name)."""
        return f"{self.model_type}_{self.label}_Location{location}"

    def paths(self, location: int):
        if self.model_type == "CA2E":  # one model for all locations
            return model_paths(self.model_type, f"{self.model_type}_{self.label}")
        return model_paths(self.model_type, self.name(location), location)


@dataclass
class Ensemble:
    values: tf.Tensor      # (n_samples, k)
    weights: tf.Tensor     # (n_samples, k)
    split: a2e.io.EvaluationSplit

    @property
    def y_t(self):
        return self.split.y_t


class ModelEnsembles:
    """Ensembles of a trained model; loads each model file once."""

    def __init__(self, spec: ModelSpec, data: Data):
        self.spec = spec
        self.data = data
        self._api = None
        self._model_path = None

    def available(self, location: int) -> bool:
        paths = self.spec.paths(location)
        return paths["model"].exists() and paths["config"].exists()

    def api(self, location: int) -> a2e.io.Api:
        paths = self.spec.paths(location)
        if self._model_path != paths["model"]:
            self._api = a2e.io.Api(a2e.io.ModelConfig.load_from_json(paths["config"]))
            self._api.model = tf.keras.models.load_model(paths["model"])
            self._model_path = paths["model"]
        return self._api

    def __call__(self, location: int, test_size: Optional[int] = None, end: Optional[int] = None) -> Ensemble:
        """Ensembles for the `test_size` queries before row `end` (default: the evaluation period)."""
        api = self.api(location)
        config = api.config
        features, targets = self.data.inputs(config, location)
        split = a2e.io.evaluation_split(features, targets, config, test_size or self.data.test_size, end=end)
        embeddings = api.embed(features, verbose=False,
                               location=location if self.spec.model_type == "CA2E" else None)
        values, weights = api.retrieve(embeddings[split.query], embeddings[split.archive], split.y_h, k=K)
        return Ensemble(values, weights, split)


def anen_ensemble(data: Data, location: int, observation_inputs: bool = True,
                  test_size: Optional[int] = None, end: Optional[int] = None) -> Ensemble:
    """AnEn (window t ± 1, k = 100, predictor weights 1) with the queries and archive end of the A2E evaluation."""
    settings = ANEN if observation_inputs else ANEN_NO_OBS
    features, targets = a2e.io.build_features(data.forecasts[:, location], data.observations[:, location], settings,
                                              observation_inputs=observation_inputs)
    split = a2e.io.evaluation_split(features, targets, settings, test_size or data.test_size, end=end)
    inputs = a2e.io.anen_inputs(features, targets, split, temporal_window=settings.foresight)
    model = a2e.model.AnEn(n_analogs=K, temporal_window=settings.foresight)
    model.build(tuple(x.shape for x in inputs))
    values = model(inputs)
    weights = tf.ones(values.shape) / values.shape[-1]
    return Ensemble(values, weights, split)


def scores(ensemble: Ensemble, pit_seed: int) -> Dict[str, np.ndarray]:
    """Per-sample scores: bias of the ensemble mean, CRPS, SCRPS, ensemble mean, randomized PIT."""
    x, w, y = ensemble.values, ensemble.weights, ensemble.y_t
    bias = a2e.metrics.computation.compute_bias(x, w, y).numpy()[:, 0]
    u = np.random.default_rng(pit_seed).uniform(size=bias.shape[0])
    return {
        "y": np.asarray(y, dtype=np.float32),
        "ens_mean": np.asarray(y, dtype=np.float32) + bias,
        "bias": bias,
        "crps": -a2e.metrics.computation.compute_crps(x, w, y).numpy(),
        "scrps": -a2e.metrics.computation.compute_scrps(x, w, y).numpy(),
        "pit": a2e.metrics.computation.compute_pit(x, w, y, u).numpy(),
    }


def summary(sample_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Means over the samples; RMSE of the ensemble mean."""
    return {
        "Bias": float(np.mean(sample_scores["bias"])),
        "RMSE": float(np.sqrt(np.mean(sample_scores["bias"] ** 2))),
        "CRPS": float(np.mean(sample_scores["crps"])),
        "SCRPS": float(np.mean(sample_scores["scrps"])),
    }


def save_per_sample(model_type: str, name: str, data: Data, ensemble: Ensemble,
                    sample_scores: Dict[str, np.ndarray]):
    """evaluation/<model type>/per_sample/<name>.npz with time (target time, UTC), y, ens_mean, bias, crps, scrps, pit."""
    path = EVALUATION_DIR / model_type / "per_sample" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    time = data.time_index[ensemble.split.target_index].tz_localize(None).values.astype("datetime64[s]")
    np.savez(path, time=time, **sample_scores)
    return path


def write_metrics(csv_path, results: Dict[int, Optional[Dict[str, float]]], model: str,
                  variant: Optional[str] = None, n_locations: int = 82):
    """
    Writes the location means into a metric CSV, columns (Metric, Model) or, with `variant`, (Metric, Model, Variant).
    Rows Location_0 ... Location_81; a location without result (None) is set to NaN. Other columns are kept.
    """
    levels = ["Metric", "Model"] + (["Variant"] if variant is not None else [])
    index = [f"Location_{i}" for i in range(n_locations)]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, header=list(range(len(levels))))
        df.columns = df.columns.set_names(levels)
        df = df.reindex(index=sorted(set(df.index) | set(index), key=lambda s: int(s.split("_")[-1])))
    else:
        df = pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples([], names=levels), dtype=float)
    df.index.name = None

    for metric in METRICS:
        column = (metric, model) + ((variant,) if variant is not None else ())
        if column not in df.columns:
            df[column] = np.nan
        for location, result in results.items():
            df.loc[f"Location_{location}", column] = np.nan if result is None else result[metric]

    # Metric-major column order, models in order of appearance
    order = {metric: i for i, metric in enumerate(METRICS)}
    df = df[sorted(df.columns, key=lambda c: (order.get(c[0], len(order)), list(df.columns).index(c)))]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path)
    return df


def print_summary(results: Dict[int, Optional[Dict[str, float]]], label: str):
    done = {loc: r for loc, r in results.items() if r is not None}
    missing = sorted(loc for loc, r in results.items() if r is None)
    if done:
        mean = pd.DataFrame(done).T.mean()
        print(f"\n{label}: mean over {len(done)} locations - " + ", ".join(f"{m}: {mean[m]:.4f}" for m in METRICS))
    if missing:
        print(f"{label}: no model for locations {missing}")
