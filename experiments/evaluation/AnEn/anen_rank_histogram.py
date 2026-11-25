"""
Rank histogram of AnEn over all locations: evaluation/figures/AnEn_rank-histogram.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.AnEn.anen_rank_histogram            # all locations
    .venv/bin/python -m experiments.evaluation.AnEn.anen_rank_histogram 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..reports import combined_rank_histogram

combined_rank_histogram(None, "AnEn")
