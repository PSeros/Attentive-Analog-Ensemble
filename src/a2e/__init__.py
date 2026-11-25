"""
Attentive Analog Ensemble (A2E):
End-to-End learning of Analog Similarity by Optimizing
Probabilistic Scoring Rules via Differentiable Retrieval.
"""
import os
# Non-interactive matplotlib backend for scripts: plt.show() no longer blocks (e.g. after each training run).
# Has to be set before matplotlib is imported. Jupyter kernels set MPLBACKEND themselves and are not affected.
os.environ.setdefault("MPLBACKEND", "Agg")

from a2e import metrics
from a2e import io
from a2e import callbacks
from a2e import layer
from a2e import loss
from a2e import model
from a2e import pipeline
from a2e import factory

__version__ = '1.0.0'
__author__ = '<phillip.schlicht@alumni.fh-aachen.de>'
# __license__ = None
# __description__ = 'TODO'
