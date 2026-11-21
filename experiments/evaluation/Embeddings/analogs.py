"""
Analogs of the evaluation queries: the top-k archive windows of A2E/CA2E/SA2E (largest attention logits)
and of AnEn (smallest AnEn dissimilarity, window t ± 1), as raw time indices of their target times.

Queries and archives are those of the evaluation scripts (a2e.io.evaluation_split, a2e.io.anen_inputs). A window
that ends in features row r has its target at raw time index r + time_to_target (see a2e.io.data).
"""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import a2e
from ..common import ANEN, K

BATCH_SIZE = 64


@dataclass
class Analogs:
    query_time: np.ndarray      # (n_queries,) raw time index of each query target
    analog_time: np.ndarray     # (n_queries, k) raw time index of each analog target, best first
    weights: np.ndarray         # (n_queries, k) member weights
    values: np.ndarray          # (n_queries, k) ensemble values (observations at the analog targets)
    y_t: np.ndarray             # (n_queries,) observed targets
    archive_time: np.ndarray    # (n_archive,) raw time index of the target of every archive window


def model_analogs(ensembles, data, location, k=K):
    """
    Top-k analogs of a trained model (ModelEnsembles) at a location, softmax weights as in Table 2.
    Returns (Analogs, embeddings of all windows, config).
    """
    api = ensembles.api(location)
    config = api.config
    features, targets = data.inputs(config, location)
    split = a2e.io.evaluation_split(features, targets, config, data.test_size)
    embeddings = api.embed(features, verbose=False,
                           location=location if ensembles.spec.model_type == "CA2E" else None)
    query, archive = embeddings[split.query], embeddings[split.archive]
    attention = api.model.cross_attention

    indices, weights = [], []
    for start in range(0, int(query.shape[0]), BATCH_SIZE):
        s = attention.scores(query[start:start + BATCH_SIZE][:, None, :], archive[None, :, :])
        s_k, i_k = tf.math.top_k(s, k=k)
        indices.append(i_k.numpy())
        weights.append(tf.nn.softmax(s_k).numpy())
    indices = np.concatenate(indices)

    first_archive_row = split.archive.start + config.seq_len - 1
    archive_time = first_archive_row + config.time_to_target + np.arange(int(archive.shape[0]))
    analogs = Analogs(query_time=np.asarray(split.target_index), analog_time=archive_time[indices],
                      weights=np.concatenate(weights), values=split.y_h.numpy()[indices], y_t=split.y_t.numpy(),
                      archive_time=archive_time)
    _check(analogs, data, location)
    return analogs, embeddings.numpy(), config


def anen_analogs(data, location, k=K):
    """Top-k analogs of AnEn (window t ± 1, predictor weights 1) at a location, equal weights."""
    features, targets = a2e.io.build_features(data.forecasts[:, location], data.observations[:, location], ANEN)
    split = a2e.io.evaluation_split(features, targets, ANEN, data.test_size)
    x_t, x_h, y_h = a2e.io.anen_inputs(features, targets, split, temporal_window=ANEN.foresight)
    model = a2e.model.AnEn(n_analogs=k, temporal_window=ANEN.foresight)
    model.build((x_t.shape, x_h.shape, y_h.shape))

    # Same steps as AnEn.call, keeping the indices of the analogs
    model.compute_feature_statistics(x_h)
    windows_t = model.create_temporal_windows(x_t)
    windows_h = model.create_temporal_windows(x_h)
    indices = []
    for start in range(0, int(windows_t.shape[0]), model.chunk_size):
        dissimilarity = model.compute_dissimilarity_matrix(windows_t[start:start + model.chunk_size], windows_h)
        indices.append(tf.nn.top_k(-dissimilarity, k=k).indices.numpy())
    indices = np.concatenate(indices)

    # Archive window j ends in features row j + 2 * temporal_window
    archive_time = np.arange(int(windows_h.shape[0])) + 2 * ANEN.foresight + ANEN.time_to_target
    y_center = np.asarray(y_h)[ANEN.foresight:-ANEN.foresight, 0]
    analogs = Analogs(query_time=np.asarray(split.target_index), analog_time=archive_time[indices],
                      weights=np.full(indices.shape, 1.0 / k), values=y_center[indices], y_t=split.y_t.numpy(),
                      archive_time=archive_time)
    _check(analogs, data, location)
    return analogs


def _check(analogs, data, location):
    """The ensemble values and targets are the observations at the computed raw time indices."""
    observations = np.asarray(data.observations[:, location, 0])
    np.testing.assert_allclose(analogs.values, observations[analogs.analog_time], rtol=0, atol=1e-5)
    np.testing.assert_allclose(analogs.y_t.reshape(-1), observations[analogs.query_time], rtol=0, atol=1e-5)
