"""
Observation, NWP forecast and ensembles of AnEn, A2E, CA2E and SA2E (cosine similarity) for one location and a
window of the validation period: evaluation/figures/all_models_comparison_location<i>.png

Run from the repository root:
    .venv/bin/python -m experiments.evaluation.values_comparison
"""
import matplotlib.pyplot as plt
import a2e
from .common import Data, ModelSpec, ModelEnsembles, anen_ensemble
from ..project_paths import FIGURES_DIR

# Select a location to evaluate
location = 0

# ========== PLOT WINDOW CONFIGURATION ==========
plot_start = -300  # Starting index within the validation samples
plot_window = 100  # Number of timesteps to plot (None = plot all)
# ===============================================

print(f"Evaluating Location {location}")
data = Data()
ensembles = {"AnEn": anen_ensemble(data, location)}
for model_type in ["A2E", "CA2E", "SA2E"]:
    print(f"Loading {model_type} model...")
    ensembles[model_type] = ModelEnsembles(ModelSpec(model_type), data)(location)

# Common y and NWP forecast for the target times (identical queries for all models)
split = ensembles["A2E"].split
y_t = split.y_t.numpy()
x_t = data.forecasts[split.target_index, location, 0]

if plot_window is None:
    window = slice(None)
else:
    plot_end = plot_start + plot_window
    window = slice(plot_start, plot_end if plot_end < 0 else None)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
window_info = f" (Window: {plot_start}-{plot_start + plot_window})" if plot_window else ""
fig.suptitle(f"Model Comparison - Location {location}{window_info}", fontsize=16)
for ax, (title, ensemble) in zip(axes.flatten(), ensembles.items()):
    weights = None if title == "AnEn" else ensemble.weights.numpy()[window]
    a2e.io.plotting.plot_values(
        y_true=y_t[window],
        x=x_t[window],
        ensemble_observations=ensemble.values.numpy()[window],
        member_weights=weights,
        title=title,
        y_lim=12,
        legend_loc="upper left",
        ax=ax,
    )
    ax.grid(True)

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
save_path = FIGURES_DIR / f"all_models_comparison_location{location}.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Plot saved to: {save_path}")
