"""
AnEn: mean bias on the training period (archive: the 2 years before it) and the validation metrics without and
with subtracting it from the members.
Result: evaluation/AnEn/anen_bias_correction_comparison.csv

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.AnEn.anen_training-test-bias            # all locations
    .venv/bin/python -m experiments.evaluation.AnEn.anen_training-test-bias 0 41       # locations 0 ... 40 (end exclusive)
"""
from ..reports import training_test_bias

training_test_bias(None, "AnEn")
