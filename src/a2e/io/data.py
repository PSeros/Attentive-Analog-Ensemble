"""
Feature construction and evaluation split shared by all training and evaluation scripts.

Index convention, with idx = foresight + time_to_target (paper: 1 + 2 = 3) and x/y the raw forecasts/observations:

    features[r] = [x_{r+idx}, y_r]        targets[r] = y_{r+idx}

A window that ends in row r contains the forecasts up to x_{t+foresight} and the observations up to
y_{t-time_to_target}, with target time t = r + time_to_target. Its target value is y_t = targets[r - foresight].
The issue time t - time_to_target is the time of the newest observation in the window (raw time index r).
The same convention holds for every archive window: its ensemble value is the value at its own target time.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf


def build_features(
        forecasts: tf.Tensor,
        observations: tf.Tensor,
        config,
        target_location: Optional[int] = None,
        observation_inputs: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the model inputs and the target values with the index convention of the module docstring.

    Args:
        forecasts: Raw forecasts, shape (time, vars) or (time, locations, vars).
        observations: Raw observations with the same leading shape as `forecasts` and one variable.
        config: ModelConfig (or any object with `foresight`, `time_to_target` and `d_vars`).
        target_location: For inputs with a location axis, return only the targets of this location
            (SA2E: 0, the location the model is trained for). None keeps all locations (CA2E training).
        observation_inputs: If False, the features contain only the forecasts (ablation without observations).

    Returns:
        features: shape (time - idx, [locations,] d_vars)
        targets: shape (time - idx, [locations,] 1), or (time - idx, 1) if `target_location` is given.
    """
    forecasts = tf.cast(forecasts, tf.float32)
    observations = tf.cast(observations, tf.float32)[..., :1]
    idx = config.foresight + config.time_to_target

    if observation_inputs:
        features = tf.concat([forecasts[idx:], observations[:-idx]], axis=-1)
    else:
        features = forecasts[idx:]
    targets = observations[idx:]
    if target_location is not None:
        targets = targets[:, target_location]

    if features.shape[-1] != config.d_vars:
        raise ValueError(f"The features have {features.shape[-1]} variables, but config.d_vars is {config.d_vars}. "
                         f"Check `observation_inputs` and the forecast variables.")
    return features, targets


def evaluation_test_size(num_timesteps: int, lookback: int, test_fraction: float = 0.3) -> int:
    """
    Number of evaluation samples per location as in the submitted evaluation (Table 2: 3720 samples).

    Args:
        num_timesteps: Length of the raw data (before `build_features`).
        lookback: Archive length used in training (config.lookback).
        test_fraction: Fraction of the data used for validation in training.
    """
    return int((num_timesteps - lookback) * test_fraction) - 2


@dataclass(frozen=True)
class EvaluationSplit:
    """
    Queries and static archive for the evaluation of a trained model.

    The embeddings are `Api.embed(features)`: embedding j belongs to the window that ends in row j + seq_len - 1.

    Attributes:
        query: Slice of the embeddings used as queries.
        archive: Slice of the embeddings used as archive keys.
        y_t: Target value of each query, shape (n_queries,).
        y_h: Ensemble value of each archive window, shape (n_archive,).
        target_index: Index of each target time on the raw time axis (row of the data before `build_features`).
        first_query_row: Row of `features` in which the first query window ends.
        last_archive_row: Row of `features` in which the newest archive window ends.
        foresight: config.foresight used for the split.
    """
    query: slice
    archive: slice
    y_t: tf.Tensor
    y_h: tf.Tensor
    target_index: np.ndarray
    first_query_row: int
    last_archive_row: int
    foresight: int


def evaluation_split(
        features: tf.Tensor,
        targets: tf.Tensor,
        config,
        test_size: int,
        end: Optional[int] = None,
) -> EvaluationSplit:
    """
    Splits features and targets into queries and a static archive.

    - Queries: the `test_size` windows that end in rows end - test_size ... end - 1 (default end: last row).
    - Targets: y_t = targets[r - foresight] for the query window ending in row r.
    - Archive: all earlier windows whose ensemble value targets[r - foresight] lies at or before the issue time of
      the first query. The ensemble value of an archive window is its value at its own target time.

    Args:
        features: Output of `build_features`, shape (time, ..., d_vars).
        targets: Output of `build_features`, shape (time, 1).
        config: ModelConfig (or any object with `seq_len`, `foresight` and `time_to_target`).
        test_size: Number of queries.
        end: Row after the last query window. Defaults to len(features).

    Returns:
        EvaluationSplit
    """
    n_rows = int(features.shape[0])
    end = n_rows if end is None else int(end)
    seq_len, fs, ttt = config.seq_len, config.foresight, config.time_to_target
    idx = fs + ttt

    first_query_row = end - test_size
    # Newest archive window: its value (target time r + time_to_target) must not lie after the issue time of the
    # first query (first_query_row); it must also end before the first query window.
    last_archive_row = first_query_row - max(ttt, 1)
    first_archive_row = max(seq_len - 1, fs)
    if first_query_row - (seq_len - 1) < 0 or last_archive_row < first_archive_row:
        raise ValueError("Not enough rows for the requested test_size and seq_len.")

    offset = seq_len - 1  # embedding j <-> window ending in row j + seq_len - 1
    if len(targets.shape) > 2 or (len(targets.shape) == 2 and targets.shape[-1] != 1):
        raise ValueError(f"targets must have shape (time,) or (time, 1), got {tuple(targets.shape)}.")
    y = targets[:, 0] if len(targets.shape) == 2 else targets
    target_rows = np.arange(first_query_row - fs, end - fs)
    return EvaluationSplit(
        query=slice(first_query_row - offset, end - offset),
        archive=slice(first_archive_row - offset, last_archive_row - offset + 1),
        y_t=y[first_query_row - fs: end - fs],
        y_h=y[first_archive_row - fs: last_archive_row - fs + 1],
        target_index=target_rows + idx,
        first_query_row=first_query_row,
        last_archive_row=last_archive_row,
        foresight=fs,
    )


def anen_inputs(features: tf.Tensor, targets: tf.Tensor, split: EvaluationSplit,
                temporal_window: int) -> List[tf.Tensor]:
    """
    Inputs of `a2e.model.AnEn` for the queries and the archive of `split`.

    AnEn compares windows t ± temporal_window and takes the target of the window center. With
    temporal_window = foresight the center row of a window ending in row r is r - foresight, which is the convention
    of `build_features`. The archive ends with the same window as the A2E archive of `split`.

    Returns:
        [current features, archive features, archive targets] for `AnEn(...)(inputs)`.
    """
    if temporal_window != split.foresight:
        raise ValueError(f"temporal_window ({temporal_window}) must equal foresight ({split.foresight}).")
    width = 2 * temporal_window
    n_queries = int(split.y_t.shape[0])
    first = split.first_query_row - width
    return [
        features[first: split.first_query_row + n_queries],
        features[: split.last_archive_row + 1],
        targets[: split.last_archive_row + 1],
    ]
