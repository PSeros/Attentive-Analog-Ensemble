"""
Which analogs A2E and AnEn choose, over all evaluation queries of each location.

Per location and method (model: top-100 by attention logits with softmax weights; AnEn: top-100 by dissimilarity
with equal weights; archive: all archive windows with equal weights as reference):
    overlap_with_AnEn:   mean share of the model's top-100 analogs that are also AnEn analogs (same target time)
    season_distance_d:   weighted mean circular distance in day of year between analog target and query target
    hour_distance_h:     weighted mean circular distance in hour of day (UTC)
    abs_obs_difference:  weighted mean |y_analog - y_t| (m/s)

Run from the repository root (model type A2E, CA2E or SA2E):
    .venv/bin/python -m experiments.evaluation.Embeddings.analog_comparison A2E          # all locations
    .venv/bin/python -m experiments.evaluation.Embeddings.analog_comparison A2E 0 41     # locations 0 ... 40

Result (rows of finished locations are replaced, other rows are kept):
    evaluation/Embeddings/analog_comparison_<model>.csv
"""
import sys
import numpy as np
import pandas as pd
from ..common import Data, ModelSpec, ModelEnsembles, location_range
from .analogs import model_analogs, anen_analogs
from ...project_paths import EVALUATION_DIR

MODEL = sys.argv[1] if len(sys.argv) > 1 else "A2E"
assert MODEL in ("A2E", "CA2E", "SA2E"), f"Model type must be A2E, CA2E or SA2E, not {MODEL}"
SPEC = ModelSpec(MODEL, "cosine_similarity")
OUTPUT_CSV = EVALUATION_DIR / "Embeddings" / f"analog_comparison_{MODEL}.csv"


def circular(a, b, period):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def statistics(analog_time, weights, query_time, y_t, day_of_year, hour, observations, chunk=256):
    """Weighted mean distances between analogs and queries, computed in chunks of queries."""
    values = {"season_distance_d": [], "hour_distance_h": [], "abs_obs_difference": []}
    for start in range(0, len(query_time), chunk):
        sl = slice(start, start + chunk)
        a, w, q = analog_time[sl], weights[sl], query_time[sl, None]
        values["season_distance_d"].append(np.sum(circular(day_of_year[a], day_of_year[q], 365.25) * w, axis=1))
        values["hour_distance_h"].append(np.sum(circular(hour[a], hour[q], 24) * w, axis=1))
        values["abs_obs_difference"].append(np.sum(np.abs(observations[a] - y_t[sl, None]) * w, axis=1))
    return {key: float(np.mean(np.concatenate(v))) for key, v in values.items()}


data = Data()
ensembles = ModelEnsembles(SPEC, data)
day_of_year = data.time_index.dayofyear.to_numpy().astype(float)
hour = data.time_index.hour.to_numpy().astype(float)

rows = []
for location in location_range(data.n_locations, arg=2):
    if not ensembles.available(location):
        print(f"Location {location}: no trained model at {SPEC.paths(location)['model']}, skipped")
        continue
    observations = np.asarray(data.observations[:, location, 0])
    model, _, _ = model_analogs(ensembles, data, location)
    anen = anen_analogs(data, location)
    assert np.array_equal(model.query_time, anen.query_time), "Model and AnEn queries differ"
    y_t = model.y_t.reshape(-1)

    overlap = np.mean([len(np.intersect1d(a, b, assume_unique=True)) / a.shape[0]
                       for a, b in zip(model.analog_time, anen.analog_time)])

    # Reference: every window of the model archive with equal weight
    n_archive = model.archive_time.shape[0]
    archive = np.broadcast_to(model.archive_time, (len(y_t), n_archive))
    archive_row = statistics(archive, np.broadcast_to(1.0 / n_archive, archive.shape), model.query_time, y_t,
                             day_of_year, hour, observations, chunk=64)

    new = [
        {"Location": location, "Method": MODEL, "overlap_with_AnEn": overlap,
         **statistics(model.analog_time, model.weights, model.query_time, y_t, day_of_year, hour, observations)},
        {"Location": location, "Method": "AnEn", "overlap_with_AnEn": 1.0,
         **statistics(anen.analog_time, anen.weights, anen.query_time, y_t, day_of_year, hour, observations)},
        {"Location": location, "Method": "archive", "overlap_with_AnEn": np.nan, **archive_row},
    ]
    rows.extend(new)
    print(f"Location {location}\n" + pd.DataFrame(new).drop(columns="Location").round(3).to_string(index=False))

    df = pd.DataFrame(rows)
    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([old[~old["Location"].isin(df["Location"])], df], ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Location"], kind="stable").to_csv(OUTPUT_CSV, index=False)

if rows:
    print("\nMean over the locations of this run:")
    print(pd.DataFrame(rows).drop(columns="Location").groupby("Method", sort=False).mean().round(3).to_string())
print(f"\nResult: {OUTPUT_CSV}")
