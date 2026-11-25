"""
SA2E (cosine similarity): mean bias on the training period (archive: the 2 years before it) and the validation
metrics without and with subtracting it from the members.
Result: evaluation/SA2E/sa2e_bias_correction_comparison.csv

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.SA2E.sa2e_training-test-bias            # all locations
    .venv/bin/python -m experiments.evaluation.SA2E.sa2e_training-test-bias 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..common import ModelSpec
from ..reports import training_test_bias

training_test_bias(ModelSpec("SA2E", "cosine_similarity"), "SA2E")
