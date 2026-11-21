"""
Rank histogram of CA2E (cosine similarity) over all locations: evaluation/figures/CA2E_rank-histogram.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.CA2E.ca2e_rank_histogram            # all locations
    .venv/bin/python -m experiments.evaluation.CA2E.ca2e_rank_histogram 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..common import ModelSpec
from ..reports import combined_rank_histogram

combined_rank_histogram(ModelSpec("CA2E", "cosine_similarity"), "CA2E")
