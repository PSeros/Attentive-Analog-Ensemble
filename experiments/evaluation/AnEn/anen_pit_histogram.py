"""
PIT histogram of AnEn over all locations: evaluation/figures/AnEn_pit-histogram.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.AnEn.anen_pit_histogram            # all locations
    .venv/bin/python -m experiments.evaluation.AnEn.anen_pit_histogram 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..reports import combined_pit_histogram

combined_pit_histogram(None, "AnEn")
