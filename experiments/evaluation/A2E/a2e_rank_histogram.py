"""
Rank histogram of A2E (cosine similarity) over all locations: evaluation/figures/A2E_rank-histogram.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.A2E.a2e_rank_histogram            # all locations
    .venv/bin/python -m experiments.evaluation.A2E.a2e_rank_histogram 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..common import ModelSpec
from ..reports import combined_rank_histogram

combined_rank_histogram(ModelSpec("A2E", "cosine_similarity"), "A2E")
