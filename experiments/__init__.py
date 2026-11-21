import os
# Non-interactive matplotlib backend for all experiment scripts (python -m experiments...): plt.show() does not block.
# Has to be set before matplotlib is imported. Jupyter kernels set MPLBACKEND themselves and are not affected.
os.environ.setdefault("MPLBACKEND", "Agg")

from .data import WindDataLoader