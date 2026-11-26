"""
PIT histogram of A2E (cosine similarity) over all locations: evaluation/figures/A2E_pit-histogram.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.A2E.a2e_pit_histogram            # all locations
    .venv/bin/python -m experiments.evaluation.A2E.a2e_pit_histogram 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..common import ModelSpec
from ..reports import combined_pit_histogram

combined_pit_histogram(ModelSpec("A2E", "cosine_similarity"), "A2E")
