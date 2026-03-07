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

    # Add observation as causal feature
    foresight = 1
    time_to_target = 2
    idx = foresight + time_to_target

    # Set up Data [time, location, variable]
    locational_observations = observations[:, location, :]
    locational_forecasts = forecasts[:, location, :]

    locational_forecasts = tf.concat([locational_forecasts[idx:], locational_observations[:-idx]], axis=-1)
    locational_observations = locational_observations[idx:, :1]

    # Set indices (-2 -> to get the number of ensembles!)
    test_size = int((forecasts.shape[0] - 8760 * 2) * 0.3) - 2

    # ===== TRAINING BIAS =====
    t_start_train = 8760 * 2 - foresight if foresight else 8760 * 2
    t_end_train = -test_size - foresight if foresight else -test_size

    # Set up evaluation data for TRAINING (feature 0 -> total wind-speed)
    y_t_train = locational_observations[t_start_train:t_end_train, 0]    # Observation of time t (TRAINING)

    # Create AnEn Model for TRAINING
    model_type = "AnEn"
    AnEn_train = A2E.model.AnEn(
        n_analogs=100,
        temporal_window=foresight
    )

    # Prepare training inputs
    train_forecast_size = locational_observations.shape[0] - 8760 * 2 - test_size
    anen_inputs_train = [
        locational_forecasts[8760 * 2:(8760 * 2 + train_forecast_size + 2)],  # +2 to produce correct number of ensembles
        locational_forecasts[:8760 * 2],
        locational_observations[:8760 * 2]
    ]
    AnEn_train.build((anen_inputs_train[0].shape, anen_inputs_train[1].shape, anen_inputs_train[2].shape))

    # Retrieve top-k AnEn ensemble for TRAINING
    ensemble_train = AnEn_train(anen_inputs_train)
    weights_train = tf.ones(ensemble_train.shape) / ensemble_train.shape[-1]

    # Compute bias for TRAINING data
    bias_train = A2E.metrics.computation.compute_bias(ensemble_train, weights_train, y_t_train)
    bias_train_mean = tf.reduce_mean(bias_train).numpy()

    # ===== TEST EVALUATION (UNCORRECTED) =====
    t_start_test = -test_size - foresight if foresight else -test_size
    t_end_test = -foresight if foresight else None

    # Set up evaluation data for TEST (feature 0 -> total wind-speed)
    y_t_test = locational_observations[t_start_test:t_end_test, 0]  # Observation of time t (TEST)

    # Create AnEn Model for TEST
    AnEn_test = A2E.model.AnEn(
        n_analogs=100,
        temporal_window=foresight
    )
    anen_inputs_test = [
        locational_forecasts[-(test_size + 2):],  # +2 to produce test_size many ensembles
        locational_forecasts[:-test_size],
        locational_observations[:-test_size]
    ]
    AnEn_test.build((anen_inputs_test[0].shape, anen_inputs_test[1].shape, anen_inputs_test[2].shape))

    # Retrieve top-k AnEn ensemble for TEST
    ensemble_test = AnEn_test(anen_inputs_test)
    weights_test = tf.ones(ensemble_test.shape) / ensemble_test.shape[-1]

    # Compute metrics for UNCORRECTED TEST data
    bias_test_uncorrected = A2E.metrics.computation.compute_bias(ensemble_test, weights_test, y_t_test)
    rmse_test_uncorrected = A2E.metrics.computation.compute_rmse(ensemble_test, weights_test, y_t_test)
    crps_test_uncorrected = -A2E.metrics.computation.compute_crps(ensemble_test, weights_test, y_t_test)
    scrps_test_uncorrected = -A2E.metrics.computation.compute_scrps(ensemble_test, weights_test, y_t_test)

    bias_test_uncorrected_mean = tf.reduce_mean(bias_test_uncorrected).numpy()
    rmse_test_uncorrected_mean = tf.reduce_mean(rmse_test_uncorrected).numpy()
    crps_test_uncorrected_mean = tf.reduce_mean(crps_test_uncorrected).numpy()
    scrps_test_uncorrected_mean = tf.reduce_mean(scrps_test_uncorrected).numpy()

    # ===== BIAS CORRECTION =====
    ensemble_test_corrected = ensemble_test - bias_train_mean

    # Compute metrics for CORRECTED TEST data
    bias_test_corrected = A2E.metrics.computation.compute_bias(ensemble_test_corrected, weights_test, y_t_test)
    rmse_test_corrected = A2E.metrics.computation.compute_rmse(ensemble_test_corrected, weights_test, y_t_test)
    crps_test_corrected = -A2E.metrics.computation.compute_crps(ensemble_test_corrected, weights_test, y_t_test)
    scrps_test_corrected = -A2E.metrics.computation.compute_scrps(ensemble_test_corrected, weights_test, y_t_test)

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
print("\nAnEn Bias Correction Comparison Summary:")
print(bias_comparison.describe())