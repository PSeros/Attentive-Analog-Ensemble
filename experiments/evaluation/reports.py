"""
Figures and tables of the auxiliary evaluation scripts (PIT histograms, training/test bias, comparison plots),
built on the ensembles of experiments.evaluation.common.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import a2e
from .common import (Data, ModelSpec, ModelEnsembles, anen_ensemble, LOOKBACK, location_range, scores, summary)
from ..project_paths import FIGURES_DIR, evaluation_file


def ensemble_source(spec, data):
    """Callable location -> Ensemble for a ModelSpec, or for AnEn if spec is None."""
    if spec is None:
        return lambda location, **kwargs: anen_ensemble(data, location, **kwargs)
    return ModelEnsembles(spec, data)


def plot_pit_histogram(ensembles, data, title, ax, locations=None):
    """PIT histogram over all locations (randomized PIT under the weighted ensemble CDF)."""
    y, values, weights = [], [], []
    for location in locations if locations is not None else range(data.n_locations):
        print(f"  {title}: location {location}")
        ensemble = ensembles(location)
        y.append(ensemble.y_t.numpy())
        values.append(ensemble.values.numpy())
        weights.append(ensemble.weights.numpy())
    a2e.io.plotting.pit_histogram(
        y_true=tf.constant(np.concatenate(y)),
        ensemble_observations=tf.constant(np.concatenate(values)),
        member_weights=tf.constant(np.concatenate(weights)),
        ax=ax,
        title=title,
    )


def combined_pit_histogram(spec, label):
    """figures/<label>_pit-histogram.png over all locations (or the `first last` range)."""
    data = Data()
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_pit_histogram(ensemble_source(spec, data), data, label, ax, location_range(data.n_locations))
    plt.tight_layout()
    save_path = FIGURES_DIR / f"{label}_pit-histogram.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def training_queries(data, seq_len):
    """
    Queries of the training period: all windows with a complete 2-year archive before them, up to the validation
    period. Returns (n_queries, end) for evaluation_split.
    """
    idx = 3  # foresight + time_to_target
    end = data.forecasts.shape[0] - idx - data.test_size
    return end - (LOOKBACK + seq_len - 1), end


def training_test_bias(spec, label):
    """
    Mean bias on the training period (archive: the 2 years before it) and the validation metrics without and with
    subtracting this bias from the members. Result: evaluation/<label>/<label lower>_bias_correction_comparison.csv
    """
    data = Data()
    ensembles = ensemble_source(spec, data)
    rows = []
    for location in location_range(data.n_locations):
        if spec is not None and not ensembles.available(location):
            print(f"Location {location}: no trained model, skipped")
            continue
        seq_len = 3 if spec is None else ensembles.api(location).config.seq_len
        n_train, end = training_queries(data, seq_len)
        train = ensembles(location, test_size=n_train, end=end)
        bias_train = float(np.mean(scores(train, pit_seed=location)["bias"]))

        test = ensembles(location)
        uncorrected = summary(scores(test, pit_seed=location))
        test.values = test.values - bias_train
        corrected = summary(scores(test, pit_seed=location))
        rows.append({"Location": location, "Training_Bias": bias_train,
                     **{f"{m}_Uncorrected": uncorrected[m] for m in ["Bias", "RMSE", "CRPS", "SCRPS"]},
                     **{f"{m}_Corrected": corrected[m] for m in ["Bias", "RMSE", "CRPS", "SCRPS"]}})
        print(f"Location {location}: training bias {bias_train:.4f}, "
              f"CRPS {uncorrected['CRPS']:.4f} (uncorrected) / {corrected['CRPS']:.4f} (corrected)")
    df = pd.DataFrame(rows)
    output_path = evaluation_file(label, f"{label.lower()}_bias_correction_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(df.describe())


def comparison_plots(spec):
    """
    Per location: bias, RMSE, CRPS and SCRPS over time, PIT histograms and Q-Q plots of the model and AnEn.
    Saved in the model's _evaluation folder (CA2E: Trained_Models/CA2E/Location<i>/<name>_evaluation).
    """
    data = Data()
    ensembles = ModelEnsembles(spec, data)
    for location in location_range(data.n_locations):
        if not ensembles.available(location):
            print(f"Location {location}: no trained model, skipped")
            continue
        print(f"Location {location}")
        model = ensembles(location)
        anen = anen_ensemble(data, location)
        y_t = model.y_t
        label = spec.model_type
        if spec.model_type == "CA2E":  # one model, figures per location
            name = f"{spec.model_type}_{spec.label}"
            prefix = spec.paths(location)["directory"] / f"Location{location}" / f"{name}_evaluation" / name
        else:
            prefix = spec.paths(location)["evaluation"] / spec.name(location)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle(f"{label} vs. AnEn - Location {location} with {spec.label}")
        for axis in ax.flatten():
            axis.grid()
        for plot, axis in [(a2e.io.plotting.plot_bias, ax[0, 0]), (a2e.io.plotting.plot_rmse, ax[0, 1]),
                           (a2e.io.plotting.plot_crps, ax[1, 0]), (a2e.io.plotting.plot_scrps, ax[1, 1])]:
            plot(y_true=y_t, ensemble_observations=model.values, member_weights=model.weights, label=label, ax=axis)
            plot(y_true=y_t, ensemble_observations=anen.values, label="AnEn", ax=axis)
        plt.tight_layout()
        plt.savefig(f"{prefix}_bias-rmse-crps-scrps-comparison.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
        fig.suptitle(f"PIT Histogram - Location {location}")
        a2e.io.plotting.pit_histogram(y_true=y_t, ensemble_observations=model.values, member_weights=model.weights,
                                      ax=ax[0], title=label)
        a2e.io.plotting.pit_histogram(y_true=y_t, ensemble_observations=anen.values, ax=ax[1], title="AnEn")
        plt.tight_layout()
        plt.savefig(f"{prefix}_pit-histogram-comparison.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        fig.suptitle(f"Q-Q Plot - Location {location}")
        a2e.io.plotting.qq_plot(y_true=y_t, ensemble_observations=model.values, member_weights=model.weights,
                                handle_ties="uniform", ax=ax[0], title=label)
        a2e.io.plotting.qq_plot(y_true=y_t, ensemble_observations=anen.values, handle_ties="uniform", ax=ax[1],
                                title="AnEn")
        plt.tight_layout()
        plt.savefig(f"{prefix}_qq-plot-comparison.png", dpi=300)
        plt.close()
