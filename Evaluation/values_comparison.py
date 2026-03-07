import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import tensorflow as tf
from Data.data_loader import WindDataLoader
import A2E

# Load Data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
observations = tf.cast(observations, dtype=tf.float32)
forecasts = tf.cast(forecasts, dtype=tf.float32)

# Select a location to evaluate
location = 0  # Change this to evaluate different locations

# ========== PLOT WINDOW CONFIGURATION ==========
plot_start = -300 # Starting index within test data (0 = start from beginning)
plot_window = 100  # Number of timesteps to plot (None = plot all)
# ===============================================

print(f"Evaluating Location {location}")

# ========== Load A2E Model ==========
print("Loading A2E model...")
a2e_model_type = "A2E"
a2e_similarity_metric = "cosine_similarity"
a2e_model_name = f"{a2e_model_type}_{a2e_similarity_metric}_Location{location}"
a2e_directory = f"Trained_Models/{a2e_model_type}/Location{location}"
a2e_model_path = f"{a2e_directory}/{a2e_model_name}.keras"
a2e_config_path = f"{a2e_directory}/{a2e_model_name}_config.json"

a2e_config = A2E.io.ModelConfig.load_from_json(a2e_config_path)
a2e_api = A2E.io.Api(a2e_config)

# Prepare A2E data
a2e_obs = observations[:, location, :]
a2e_fcst = forecasts[:, location, :]
idx = a2e_config.foresight + a2e_config.time_to_target
a2e_fcst = tf.concat([a2e_fcst[idx:], a2e_obs[:-idx]], axis=-1)
a2e_obs = a2e_obs[idx:, :1]

test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2
t_start = -test_size - a2e_config.foresight if a2e_config.foresight else -test_size
t_end = -a2e_config.foresight if a2e_config.foresight else None
h_start = a2e_config.seq_len - 1
h_end = -test_size

# Common x and y_true for all models
x_t = a2e_fcst[t_start:t_end, 0]
y_t = a2e_obs[t_start:t_end, 0]
y_h_a2e = a2e_obs[h_start:h_end, 0]

a2e_embeddings = a2e_api.embed(forecasts=a2e_fcst, model_path=a2e_model_path, verbose=False)
z_t_a2e = a2e_embeddings[-test_size:]
z_h_a2e = a2e_embeddings[:-test_size]
a2e_ensemble, a2e_weights = a2e_api.retrieve(z_t_a2e, z_h_a2e, y_h_a2e, k=100)

# ========== Load CA2E Model ==========
print("Loading CA2E model...")
ca2e_model_type = "CA2E"
ca2e_model_name = f"{ca2e_model_type}_{a2e_similarity_metric}"
ca2e_directory = f"Trained_Models/{ca2e_model_type}"
ca2e_model_path = f"{ca2e_directory}/{ca2e_model_name}.keras"
ca2e_config_path = f"{ca2e_directory}/{ca2e_model_name}_config.json"

ca2e_config = A2E.io.ModelConfig.load_from_json(ca2e_config_path)
ca2e_api = A2E.io.Api(ca2e_config)

# Prepare CA2E data (same as A2E for single location)
ca2e_obs = observations[:, location, :]
ca2e_fcst = forecasts[:, location, :]
idx = ca2e_config.foresight + ca2e_config.time_to_target
ca2e_fcst = tf.concat([ca2e_fcst[idx:], ca2e_obs[:-idx]], axis=-1)
ca2e_obs = ca2e_obs[idx:, :1]

y_h_ca2e = ca2e_obs[h_start:h_end, 0]

ca2e_embeddings = ca2e_api.embed(forecasts=ca2e_fcst, model_path=ca2e_model_path, verbose=False)
z_t_ca2e = ca2e_embeddings[-test_size:]
z_h_ca2e = ca2e_embeddings[:-test_size]
ca2e_ensemble, ca2e_weights = ca2e_api.retrieve(z_t_ca2e, z_h_ca2e, y_h_ca2e, k=100)

# ========== Load SA2E Model ==========
print("Loading SA2E model...")
sa2e_model_type = "SA2E"
sa2e_model_name = f"{sa2e_model_type}_{a2e_similarity_metric}_Location{location}"
sa2e_directory = f"Trained_Models/{sa2e_model_type}/Location{location}"
sa2e_model_path = f"{sa2e_directory}/{sa2e_model_name}.keras"
sa2e_config_path = f"{sa2e_directory}/{sa2e_model_name}_config.json"

sa2e_config = A2E.io.ModelConfig.load_from_json(sa2e_config_path)
sa2e_api = A2E.io.Api(sa2e_config)

# Get nearest neighbors for SA2E
nearest_locations = loader.find_nearest_locations(sa2e_config.d_loc - 1)
neighbours = nearest_locations[location, :]

sa2e_obs = tf.gather(observations, neighbours, axis=1)
sa2e_fcst = tf.gather(forecasts, neighbours, axis=1)
idx = sa2e_config.foresight + sa2e_config.time_to_target
sa2e_fcst = tf.concat([sa2e_fcst[idx:], sa2e_obs[:-idx]], axis=-1)
sa2e_obs = sa2e_obs[idx:, 0, :1]

y_h_sa2e = sa2e_obs[h_start:h_end, 0]

sa2e_embeddings = sa2e_api.embed(sa2e_fcst, model_path=sa2e_model_path, verbose=False)
z_t_sa2e = sa2e_embeddings[-test_size:]
z_h_sa2e = sa2e_embeddings[:-test_size]
sa2e_ensemble, sa2e_weights = sa2e_api.retrieve(z_t_sa2e, z_h_sa2e, y_h_sa2e, k=100)

# ========== AnEn Model ==========
print("Running AnEn...")
AnEn = A2E.model.AnEn(n_analogs=100, temporal_window=a2e_config.foresight)

anen_inputs = [
    a2e_fcst[-(test_size + 2):],
    a2e_fcst[:-test_size],
    a2e_obs[:-test_size]
]
AnEn.build((anen_inputs[0].shape, anen_inputs[1].shape, anen_inputs[2].shape))
anen_ensemble = AnEn(anen_inputs)

# ========== Apply Plot Window ==========
if plot_window is not None:
    plot_end = min(plot_start + plot_window, len(x_t))

    x_t_plot = x_t[plot_start:plot_end]
    y_t_plot = y_t[plot_start:plot_end]
    a2e_ensemble_plot = a2e_ensemble[plot_start:plot_end]
    a2e_weights_plot = a2e_weights[plot_start:plot_end]
    ca2e_ensemble_plot = ca2e_ensemble[plot_start:plot_end]
    ca2e_weights_plot = ca2e_weights[plot_start:plot_end]
    sa2e_ensemble_plot = sa2e_ensemble[plot_start:plot_end]
    sa2e_weights_plot = sa2e_weights[plot_start:plot_end]
    anen_ensemble_plot = anen_ensemble[plot_start:plot_end]

    print(f"Plotting window: timesteps {plot_start} to {plot_end} (total: {plot_end - plot_start})")
else:
    x_t_plot = x_t
    y_t_plot = y_t
    a2e_ensemble_plot = a2e_ensemble
    a2e_weights_plot = a2e_weights
    ca2e_ensemble_plot = ca2e_ensemble
    ca2e_weights_plot = ca2e_weights
    sa2e_ensemble_plot = sa2e_ensemble
    sa2e_weights_plot = sa2e_weights
    anen_ensemble_plot = anen_ensemble

    print(f"Plotting all timesteps: {len(x_t)}")

# ========== Create Combined Plot ==========
print("Creating combined plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
window_info = f" (Window: {plot_start}-{plot_start + len(x_t_plot)})" if plot_window else ""
fig.suptitle(f"Model Comparison - Location {location}{window_info}", fontsize=16)

# Flatten axes for easier indexing
ax = axes.flatten()

# Plot AnEn
A2E.io.plotting.plot_values(
    y_true=y_t_plot,
    x=x_t_plot,
    ensemble_observations=anen_ensemble_plot,
    title="AnEn",
    y_lim=12,
    legend_loc='upper left',
    ax=ax[0]
)
ax[0].grid(True)

# Plot A2E
A2E.io.plotting.plot_values(
    y_true=y_t_plot,
    x=x_t_plot,
    ensemble_observations=a2e_ensemble_plot,
    member_weights=a2e_weights_plot,
    title="A2E",
    y_lim=12,
    legend_loc='upper left',
    ax=ax[1]
)
ax[1].grid(True)

# Plot CA2E
A2E.io.plotting.plot_values(
    y_true=y_t_plot,
    x=x_t_plot,
    ensemble_observations=ca2e_ensemble_plot,
    member_weights=ca2e_weights_plot,
    title="CA2E",
    y_lim=12,
    legend_loc='upper left',
    ax=ax[2]
)
ax[2].grid(True)

# Plot SA2E
A2E.io.plotting.plot_values(
    y_true=y_t_plot,
    x=x_t_plot,
    ensemble_observations=sa2e_ensemble_plot,
    member_weights=sa2e_weights_plot,
    title="SA2E",
    y_lim=12,
    legend_loc='upper left',
    ax=ax[3]
)
ax[3].grid(True)

plt.tight_layout()

# Save the plot
save_directory = f"Evaluation/figures"
os.makedirs(save_directory, exist_ok=True)
save_path = f"{save_directory}/all_models_comparison_location{location}.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")

plt.show()
plt.close()

print("Finished!")