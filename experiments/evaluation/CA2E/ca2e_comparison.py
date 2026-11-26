"""
CA2E (cosine similarity) vs. AnEn per location: bias, RMSE, CRPS, SCRPS over time, PIT histograms, Q-Q plots.
Saved in the _evaluation folder of the model.

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.CA2E.ca2e_comparison            # all locations
    .venv/bin/python -m experiments.evaluation.CA2E.ca2e_comparison 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..common import ModelSpec
from ..reports import comparison_plots

comparison_plots(ModelSpec("CA2E", "cosine_similarity"))
