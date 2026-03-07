import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
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

# Prepare DataFrame for bias comparison and metrics
bias_comparison = pd.DataFrame(
    columns=['Location', 'Training_Bias', 'Test_Bias_Uncorrected', 'Test_Bias_Corrected',
             'RMSE_Uncorrected', 'RMSE_Corrected', 'CRPS_Uncorrected', 'CRPS_Corrected',
             'SCRPS_Uncorrected', 'SCRPS_Corrected']
)

# Evaluate all locations
for location in range(82):
    print(f"Evaluating Location {location}")

    # Prepare paths
    model_type = "CA2E"
    similarity_metric = "cosine_similarity"
    model_name = f"{model_type}_{similarity_metric}"
    directory = f"Trained_Models/{model_type}"
    model_path = f"{directory}/{model_name}.keras"
    config_path = f"{directory}/{model_name}_config.json"

    # Load Model Configs
    config = A2E.io.ModelConfig.load_from_json(config_path)

    # Initialize Api
    api = A2E.io.Api(config)

    # Set up Data [time, location, variable]
    locational_observations = observations[:, location, :]
    locational_forecasts = forecasts[:, location, :]

    # Add observation as causal feature
    idx = config.foresight + config.time_to_target
    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices based on model configurations (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2

    # Compute embeddings (at the same time load model into api)
    embeddings = api.embed(locational_forecasts, model_path=model_path, verbose=False)

    # ===== TRAINING BIAS =====
    train_size = embeddings.shape[0] - test_size - (8760 * 2)

    t_start_train = 8760 * 2 + (config.seq_len - 1) - config.foresight if config.foresight else 8760 * 2 + (config.seq_len - 1)
    t_end_train = -test_size - config.foresight if config.foresight else -test_size
    h_start_train = config.seq_len - 1
    h_end_train = 8760 * 2 + (config.seq_len - 1)

    # Set up evaluation data for TRAINING (feature 0 -> total wind-speed)
    y_t_train = locational_observations[t_start_train:t_end_train, 0]  # Observation of time t (TRAINING)
    y_h_train = locational_observations[h_start_train:h_end_train, 0]  # Observation archive aligned to hist embeddings (LOOKBACK)

    # Split embeddings for TRAINING
    z_t_train = embeddings[8760 * 2:-test_size]
    z_h_train = embeddings[:8760 * 2]

    # Retrieve top-k A2E ensemble and corresponding weights for TRAINING data
    ca2e_ensemble_train, ca2e_weights_train = api.retrieve(z_t_train, z_h_train, y_h_train, k=100)

    # Compute bias for TRAINING data
    bias_train = A2E.metrics.computation.compute_bias(ca2e_ensemble_train, ca2e_weights_train, y_t_train)
    bias_train_mean = tf.reduce_mean(bias_train).numpy()

    # ===== TEST EVALUATION (UNCORRECTED) =====
    t_start_test = -test_size - config.foresight if config.foresight else -test_size
    t_end_test = -config.foresight if config.foresight else None
    h_start_test = config.seq_len - 1
    h_end_test = -test_size

    # Set up evaluation data for TEST (feature 0 -> total wind-speed)
    y_t_test = locational_observations[t_start_test:t_end_test, 0]  # Observation of time t (TEST)
    y_h_test = locational_observations[h_start_test:h_end_test, 0]  # Observation archive aligned to hist embeddings (LOOKBACK + TRAINING)

    # Split embeddings for TEST
    z_t_test = embeddings[-test_size:]
    z_h_test = embeddings[:-test_size]

    # Retrieve top-k A2E ensemble and corresponding weights for TEST data
    ca2e_ensemble_test, ca2e_weights_test = api.retrieve(z_t_test, z_h_test, y_h_test, k=100)

    # Compute metrics for UNCORRECTED TEST data
    bias_test_uncorrected = A2E.metrics.computation.compute_bias(ca2e_ensemble_test, ca2e_weights_test, y_t_test)
    rmse_test_uncorrected = A2E.metrics.computation.compute_rmse(ca2e_ensemble_test, ca2e_weights_test, y_t_test)
    crps_test_uncorrected = -A2E.metrics.computation.compute_crps(ca2e_ensemble_test, ca2e_weights_test, y_t_test)
    scrps_test_uncorrected = -A2E.metrics.computation.compute_scrps(ca2e_ensemble_test, ca2e_weights_test, y_t_test)

    bias_test_uncorrected_mean = tf.reduce_mean(bias_test_uncorrected).numpy()
    rmse_test_uncorrected_mean = tf.reduce_mean(rmse_test_uncorrected).numpy()
    crps_test_uncorrected_mean = tf.reduce_mean(crps_test_uncorrected).numpy()
    scrps_test_uncorrected_mean = tf.reduce_mean(scrps_test_uncorrected).numpy()

    # ===== BIAS CORRECTION =====
    ca2e_ensemble_test_corrected = ca2e_ensemble_test - bias_train_mean

    # Compute metrics for CORRECTED TEST data
    bias_test_corrected = A2E.metrics.computation.compute_bias(ca2e_ensemble_test_corrected, ca2e_weights_test, y_t_test)
    rmse_test_corrected = A2E.metrics.computation.compute_rmse(ca2e_ensemble_test_corrected, ca2e_weights_test, y_t_test)
    crps_test_corrected = -A2E.metrics.computation.compute_crps(ca2e_ensemble_test_corrected, ca2e_weights_test, y_t_test)
    scrps_test_corrected = -A2E.metrics.computation.compute_scrps(ca2e_ensemble_test_corrected, ca2e_weights_test, y_t_test)

    bias_test_corrected_mean = tf.reduce_mean(bias_test_corrected).numpy()
    rmse_test_corrected_mean = tf.reduce_mean(rmse_test_corrected).numpy()
    crps_test_corrected_mean = tf.reduce_mean(crps_test_corrected).numpy()
    scrps_test_corrected_mean = tf.reduce_mean(scrps_test_corrected).numpy()

    # Store results
    bias_comparison.loc[location] = [
        location,
        bias_train_mean,
        bias_test_uncorrected_mean,
        bias_test_corrected_mean,
        rmse_test_uncorrected_mean,
        rmse_test_corrected_mean,
        crps_test_uncorrected_mean,
        crps_test_corrected_mean,
        scrps_test_uncorrected_mean,
        scrps_test_corrected_mean
    ]

    print(f"Location {location}:")
    print(f"  Training Bias: {bias_train_mean:.4f}")
    print(f"  Test Bias (Uncorrected): {bias_test_uncorrected_mean:.4f} | (Corrected): {bias_test_corrected_mean:.4f}")
    print(f"  RMSE (Uncorrected): {rmse_test_uncorrected_mean:.4f} | (Corrected): {rmse_test_corrected_mean:.4f}")
    print(f"  CRPS (Uncorrected): {crps_test_uncorrected_mean:.4f} | (Corrected): {crps_test_corrected_mean:.4f}")
    print(f"  SCRPS (Uncorrected): {scrps_test_uncorrected_mean:.4f} | (Corrected): {scrps_test_corrected_mean:.4f}")

# Save results
output_path = f"Evaluation/{model_type}/{model_type.lower()}_bias_correction_comparison.csv"
bias_comparison.to_csv(output_path, index=False)
print(f"\nBias correction comparison complete. Results saved to {output_path}")

# Display summary statistics
print("\nBias Correction Comparison Summary:")
print(bias_comparison.describe())