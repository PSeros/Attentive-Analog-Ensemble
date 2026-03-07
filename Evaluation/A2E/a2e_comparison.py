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

for location in range(62, forecasts.shape[1]):
    print(f"Evaluating Location {location}")

    # Prepare paths
    model_type = "A2E"
    similarity_metric = "cosine_similarity"
    model_name = f"{model_type}_{similarity_metric}_Location{location}"
    directory = f"Trained_Models/{model_type}/Location{location}"
    model_path = f"{directory}/{model_name}.keras"
    config_path = f"{directory}/{model_name}_config.json"

    # Load Model Configs
    print("\rA2E...", end="", flush=True)
    config = A2E.io.ModelConfig.load_from_json(config_path)

    # Initialize Api
    api = A2E.io.Api(config)

    # Set up Data [time, location, variable]
    locational_observations=observations[:,location,:]
    locational_forecasts=forecasts[:,location,:]

    # Add observation as causal feature
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) -2
    t_start = -test_size -config.foresight if config.foresight else -test_size
    t_end = -config.foresight if config.foresight else None
    h_start = config.seq_len -1
    h_end = -test_size

    # Set up evaluation data (feature 0 -> total wind-speed)
    x_t = locational_forecasts   [t_start:t_end, 0]    # NWP for time t
    y_t = locational_observations[t_start:t_end, 0]    # Observation of time t
    y_h = locational_observations[h_start:h_end, 0]    # Observation archive aligned to hist embeddings

    # Compute embeddings (at the same time load model into api)
    embeddings = api.embed(
        forecasts=locational_forecasts,
        model_path=model_path,
        verbose=True
    )

    # Split embeddings
    z_t = embeddings[-test_size:]
    z_h = embeddings[:-test_size]

    # Retrieve top-k A2E ensemble and corresponding weights
    a2e_ensemble, a2e_weights = api.retrieve(z_t, z_h, y_h, k=100)

    # AnEn
    print("\rAnEn...", end="", flush=True)
    AnEn = A2E.model.AnEn(
        n_analogs=100,
        temporal_window=config.foresight
    )

    anen_inputs = [
        locational_forecasts[-(test_size+2):], # +2 to produce test_size many ensembles
        locational_forecasts[:-test_size],
        locational_observations[:-test_size]
    ]
    AnEn.build((anen_inputs[0].shape, anen_inputs[1].shape, anen_inputs[2].shape))

    # Retrieve top-k AnEn ensemble
    anen_ensemble = AnEn(anen_inputs)

    # Plotting
    print("\rInitializing Comparison Plot...", end="", flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"{model_type} vs. AnEn - Location {location} with {similarity_metric}")

    for axis in ax.flatten():
        axis.grid()
    print("\rPlot Bias...", end="", flush=True)
    A2E.io.plotting.plot_bias(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        label="A2E",
        ax=ax[0, 0]
    )
    A2E.io.plotting.plot_bias(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        label="AnEn",
        ax=ax[0, 0]
    )

    print("\rPlot RMSE...", end="", flush=True)
    A2E.io.plotting.plot_rmse(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        label="A2E",
        ax=ax[0, 1]
    )
    A2E.io.plotting.plot_rmse(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        label="AnEn",
        ax=ax[0, 1]
    )

    print("\rPlot CRPS...", end="", flush=True)
    A2E.io.plotting.plot_crps(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        label="A2E",
        ax=ax[1, 0]
    )
    A2E.io.plotting.plot_crps(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        label="AnEn",
        ax=ax[1, 0]
    )

    print("\rPlot SCRPS...", end="", flush=True)
    A2E.io.plotting.plot_scrps(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        label="A2E",
        ax=ax[1, 1]
    )
    A2E.io.plotting.plot_scrps(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        label="AnEn",
        ax=ax[1, 1]
    )

    print("\rSaving Comparison Plot...", end="", flush=True)
    plt.tight_layout()
    plot_type = "bias-rmse-crps-scrps"
    save_path = f"{directory}/{model_name}_evaluation/{model_name}_{plot_type}-comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    #plt.show()
    plt.close()

    print("\rInitializing Rank Histogram Comparison...", end="", flush=True)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle(f"Rank Histogram - Location {location}")

    A2E.io.plotting.rank_histogram(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        handle_ties="uniform",
        ax=ax[0],
        title="A2E",
    )
    A2E.io.plotting.rank_histogram(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        handle_ties="uniform",
        ax=ax[1],
        title="AnEn",
    )
    plt.tight_layout()
    plot_type = "rank-histogram"
    save_path = f"{directory}/{model_name}_evaluation/{model_name}_{plot_type}-comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.close()

    print("\rInitializing Q-Q Plot...", end="", flush=True)
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    fig.suptitle(f"Q-Q Plot - {model_type} Location {location}")

    A2E.io.plotting.qq_plot(
        y_true=y_t,
        ensemble_observations=a2e_ensemble,
        member_weights=a2e_weights,
        handle_ties="uniform",
        ax=ax[0],
        title="A2E",
    )
    A2E.io.plotting.qq_plot(
        y_true=y_t,
        ensemble_observations=anen_ensemble,
        handle_ties="uniform",
        ax=ax[1],
        title="AnEn",
    )

    plt.tight_layout()
    plot_type = "qq-plot"
    save_path = f"{directory}/{model_name}_evaluation/{model_name}_{plot_type}-comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.close()
    print("\rFinished.", end="", flush=True)