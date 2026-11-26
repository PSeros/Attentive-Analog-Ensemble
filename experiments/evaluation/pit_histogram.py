"""
PIT histograms of AnEn, A2E, CA2E and SA2E (cosine similarity) over all locations in one figure:
evaluation/figures/combined_pit_histograms.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.pit_histogram
"""
import matplotlib.pyplot as plt
from .common import Data, ModelSpec
from .reports import ensemble_source, plot_pit_histogram
from ..project_paths import FIGURES_DIR

data = Data()
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
models = [("AnEn", None), ("A2E", ModelSpec("A2E")), ("CA2E", ModelSpec("CA2E")), ("SA2E", ModelSpec("SA2E"))]
for ax, (title, spec) in zip(axes.flatten(), models):
    print(f"\nProcessing {title}...")
    plot_pit_histogram(ensemble_source(spec, data), data, title, ax)

plt.tight_layout()
save_path = FIGURES_DIR / "combined_pit_histograms.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved {save_path}")
