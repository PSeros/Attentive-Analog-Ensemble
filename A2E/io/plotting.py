import os
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import gaussian_kde
from typing import Literal
from A2E.metrics.computation import compute_crps, compute_scrps, compute_bias, compute_rmse, compute_quantiles

def plot_bias(
    y_true: tf.Tensor,
    ensemble_observations: tf.Tensor,
    member_weights: tf.Tensor = None,
    title: str = "Bias",
    label: str = None,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot Bias over time based on the Ensemble mean.

    Args:
        y_true (tf.Tensor): True values, shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble observations, shape (n_samples, n_members)
        member_weights (tf.Tensor): Weights for ensemble members, shape (n_samples, n_members)
        color (str): Color of the legend, None for no color.
        title (str): Title of the legend, None for no title.
        label (str): Label of the legend, None for no label.
        ax (plt.Axes): Axes object for plotting. If None, creates a new figure.

    Returns:
        plt.Axes: Axes object with the plot.
    """
    # Extract shape
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    # Standard weights if not given
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members

    # Ensure Tensor
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Compute bias values
    bias = compute_bias(ensemble_observations, member_weights, y_true)
    mean_bias = tf.reduce_mean(bias)

    if label:
        label = f"{label} (Mean: {mean_bias:.4f})"
    else:
        label = f"(Mean: {mean_bias:.4f})"

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bias, label=label, color=color)
    ax.set(
        xlabel="Time Step",
        ylabel="Value",
        title=title
    )
    if label:
        ax.legend(loc="upper right")
    return ax

def plot_rmse(
    y_true: tf.Tensor,
    ensemble_observations: tf.Tensor,
    member_weights: tf.Tensor = None,
    title: str = "RMSE",
    label: str = None,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot RMSE over time based on ensemble members and true values.

    Args:
        y_true (tf.Tensor): True values, shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble observations, shape (n_samples, n_members)
        member_weights (tf.Tensor): Weights for ensemble members, shape (n_samples, n_members)
        color (str): Color of the legend, None for no color.
        title (str): Title of the legend, None for no title.
        label (str): Label of the legend, None for no label.
        ax (plt.Axes): Axes object for plotting. If None, creates a new figure.

    Returns:
        plt.Axes: Axes object with the plot.
    """
    # Extract shape
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    # Standard weights if not given
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members

    # Ensure Tensor
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Compute RMSE values
    rmse = compute_rmse(ensemble_observations, member_weights, y_true)
    mean_rmse = tf.reduce_mean(rmse)

    if label:
        label = f"{label} (Mean: {mean_rmse:.4f})"
    else:
        label = f"(Mean: {mean_rmse:.4f})"

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rmse, label=label, color=color)
    ax.set(
        xlabel="Time Step",
        ylabel="Value",
        title=title
    )
    if label:
        ax.legend(loc="upper right")
    return ax

def plot_crps(
    y_true: tf.Tensor,
    ensemble_observations: tf.Tensor,
    member_weights: tf.Tensor = None,
    title: str = "CRPS",
    label: str = None,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot CRPS over time for each time step.

    Args:
        y_true (tf.Tensor): True values, shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble observations, shape (n_samples, n_members)
        member_weights (tf.Tensor): Weights for ensemble members, shape (n_samples, n_members)
        color (str): Color of the legend, None for no color.
        title (str): Title of the legend, None for no title.
        label (str): Label of the legend, None for no label.
        ax (plt.Axes): Axes object for plotting. If None, creates a new figure.

    Returns:
        plt.Axes: Axes object with the plot.
    """
    # Extract shape
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    # Standard weights if not given
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members

    # Ensure Tensor
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Compute CRPS for each time step
    if tf.rank(y_true).numpy() == 1:
        y_true = tf.expand_dims(y_true, axis=1)
    crps = -compute_crps(ensemble_observations, member_weights, y_true) # comes as tf.Tensor
    mean_crps = tf.reduce_mean(crps)

    if label:
        label = f"{label} (Mean: {mean_crps:.4f})"
    else:
        label = f"(Mean: {mean_crps:.4f})"

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(crps, label=label, color=color)
    ax.set(
        xlabel="Time Step",
        ylabel="Value",
        title=title
    )
    if label:
        ax.legend(loc="upper right")
    return ax

def plot_scrps(
    y_true: tf.Tensor,
    ensemble_observations: tf.Tensor,
    member_weights: tf.Tensor = None,
    title: str = "SCRPS",
    label: str = None,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot SCRPS over time for each time step.

    Args:
        y_true (tf.Tensor): True values, shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble observations, shape (n_samples, n_members)
        member_weights (tf.Tensor): Weights for ensemble members, shape (n_samples, n_members)
        color (str): Color of the legend, None for no color.
        title (str): Title of the legend, None for no title.
        label (str): Label of the legend, None for no label.
        ax (plt.Axes): Axes object for plotting. If None, creates a new figure.

    Returns:
        plt.Axes: Axes object with the plot.
    """
    # Extract shape
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    # Standard weights if not given
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members

    # Ensure Tensor
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Compute SCRPS for each time step
    scrps = -compute_scrps(ensemble_observations, member_weights, y_true)
    mean_scrps = tf.reduce_mean(scrps)

    if label:
        label = f"{label} (Mean: {mean_scrps:.4f})"
    else:
        label = f"(Mean: {mean_scrps:.4f})"

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scrps, label=label, color=color)
    ax.set(
        xlabel="Time Step",
        ylabel="Value",
        title=title
    )
    if label:
        ax.legend(loc="upper right")
    return ax

def plot_values(
        y_true: tf.Tensor,
        x: tf.Tensor,
        ensemble_observations: tf.Tensor,
        member_weights: tf.Tensor = None,
        lower_quantiles: list[float] = None,
        upper_quantiles: list[float] = None,
        annotate_metrics: bool = False,
        title: str = None,
        y_lim: float = None,
        legend_loc: str = None,
        ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot of the actual observation, original forecast, and ensemble forecast with color-coded quantile ranges.
    Args:
        y_true (tf.Tensor): Observed values, Shape (n_samples,)
        x (tf.Tensor): Predicted values, Shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble members, Shape (n_samples, member)
        member_weights (tf.Tensor): Weights for ensemble members, Shape (n_samples, member)
        lower_quantiles (list[float]): Lists for the lower quantiles,
        upper_quantiles (list[float]): Lists for the upper quantiles,
        annotate_metrics (bool): Whether or not to annotate metrics,
        title (str): Title of the legend, None for no title.
        y_lim (float): Maximum of the Y-axis for the plot (optional),
        ax (plt.Axes): Axes object for plotting (optional)
    Returns:
        plt.Axes: Axes object with the plot
    """
    # Extract shapes
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()
    if tf.rank(y_true).numpy() == 1:
        y_true = tf.expand_dims(y_true, axis=-1)

    # Default weights if not specified
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members)) / n_members

    # Ensure tf.Tensor in tf.float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)
    ensemble_mean = tf.reduce_sum(ensemble_observations * member_weights, axis=1)

    # Compute quantiles
    if lower_quantiles is None:
        lower_quantiles = [0.005, 0.05]
    if upper_quantiles is None:
        upper_quantiles = [0.995, 0.95]

    lower_quantile_values = compute_quantiles(ensemble_observations, member_weights, tf.constant(lower_quantiles, dtype=tf.float32))
    upper_quantile_values = compute_quantiles(ensemble_observations, member_weights, tf.constant(upper_quantiles, dtype=tf.float32))

    # Compute Metrics
    if annotate_metrics:
        crps = -compute_crps(ensemble_observations, member_weights, y_true)
        scrps = -compute_scrps(ensemble_observations, member_weights, y_true)
        mean_crps = tf.reduce_mean(crps)
        mean_scrps = tf.reduce_mean(scrps)

    # Create plot
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Plot Values
    ax.plot(y_true, label="Observation", color="orange")
    ax.plot(x, label="Forecast", color="red")
    ax.plot(ensemble_mean, label="Ensemble Mean", color="blue")

    # Plot quantiles
    for lqv, uqv, lq, uq in zip(tf.transpose(lower_quantile_values), tf.transpose(upper_quantile_values),
                                lower_quantiles, upper_quantiles):
        ax.fill_between(
            range(n_forecasts),
            lqv,
            uqv,
            color="blue",
            alpha=0.2,
            label=f"Ensemble: {round((uq-lq)*100,0)}% PI"
        )

    # Prepare the rest of the Plot
    if y_lim:
        ax.set_ylim(0, y_lim)
    ax.set(
        xlabel="Sample Index",
        ylabel="Value",
        title=title
    )
    ax.legend(loc=legend_loc or 'upper right')
    if annotate_metrics:
        ax.text(
            0.05, 0.95,
            f"  CRPS: {mean_crps:.4f}\nSCRPS: {mean_scrps:.4f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
        )
    return ax


def rank_histogram(
        y_true: tf.Tensor,
        ensemble_observations: tf.Tensor,
        member_weights: tf.Tensor = None,
        bins: int = 10,
        handle_ties: Literal["uniform", "exclude"] = "uniform",
        title: str = "Rank Histogram",
        ax: plt.Axes = None,
) -> plt.Axes:
    """
    Generates a Rank histogram for probabilistic forecasts using vectorized operations.
    A Rank histogram visualizes the distribution of forecast ranks to assess calibration.

    Args:
        y_true (tf.Tensor): Observed values, shape (n_forecasts,).
        ensemble_observations (tf.Tensor): Ensemble member forecasts,
            shape (n_forecasts, n_members).
        member_weights (tf.Tensor, optional): Weights for each ensemble member,
            shape (n_forecasts, n_members). Defaults to equal weights.
        bins (int, optional): Number of histogram bins. Default is 10.
        handle_ties (["uniform", "exclude"], optional): How to handle ties between observation and members:
            - "uniform": distribute tied mass uniformly within the tie interval.
            - "exclude": omit tied cases.
            Defaults to "uniform".
        title (str, optional): Plot title. Defaults to "Rank Histogram".
        ax (plt.Axes, optional): Matplotlib Axes on which to plot. If None, a new figure is created.

    Returns:
        plt.Axes: The Axes object containing the histogram.
    """
    # Extract shapes and prepare tensors
    n_forecasts, n_members = tf.unstack(tf.shape(ensemble_observations))

    if len(y_true.shape) == 1:
        y_true = tf.expand_dims(y_true, axis=-1)

    # Initialize equal weights if none provided
    if member_weights is None:
        member_weights = tf.ones_like(ensemble_observations) / tf.cast(n_members, tf.float32)

    # Ensure consistent dtypes
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Vectorized rank computation
    # Expand y_true to broadcast with ensemble_observations: (n_forecasts, 1) -> (n_forecasts, n_members)
    y_expanded = tf.expand_dims(y_true[:, 0], axis=1)

    # Create boolean masks for comparisons
    below_mask = ensemble_observations < y_expanded  # (n_forecasts, n_members)
    equal_mask = tf.abs(ensemble_observations - y_expanded) < 1e-8  # (n_forecasts, n_members)

    # Compute weights below and equal for all forecasts at once
    weight_below = tf.reduce_sum(member_weights * tf.cast(below_mask, tf.float32), axis=1)  # (n_forecasts,)
    weight_equal = tf.reduce_sum(member_weights * tf.cast(equal_mask, tf.float32), axis=1)  # (n_forecasts,)

    if handle_ties == "uniform":
        # Generate random uniform values for all forecasts
        u = tf.random.uniform((n_forecasts,), 0, 1)
        ranks = weight_below + weight_equal * u
    elif handle_ties == "exclude":
        # Create mask for non-tied cases
        non_tied_mask = weight_equal < 1e-8
        ranks = tf.boolean_mask(weight_below, non_tied_mask)
    else:
        raise ValueError(f"handle_ties has to be 'uniform' or 'exclude' but is '{handle_ties}'")

    # Create plot
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ranks_np = ranks.numpy()
    ax.hist(ranks_np, bins=bins, density=True, alpha=0.5, edgecolor='k')
    ax.axhline(1, color='k', linestyle='--', label='Uniform Reference')

    # Set labels and title
    ax.set(
        ylim=[0, 2],
        xlim=[0, 1],
        xlabel="Normalized Rank",
        ylabel="Density",
        title=title
    )
    ax.legend(loc="upper right")
    return ax


def qq_plot(
        y_true: tf.Tensor,
        ensemble_observations: tf.Tensor,
        member_weights: tf.Tensor = None,
        handle_ties: Literal["uniform", "exclude"] = "uniform",
        title: str = "QQ Plot",
        ax: plt.Axes = None,
) -> plt.Axes:
    """
    Creates a QQ-Plot for probabilistic forecasts using vectorized operations.
    Compares observed quantiles with theoretical quantiles of a uniform distribution.

    Args:
        y_true (tf.Tensor): Array of observations, Shape (n_forecasts,)
        ensemble_observations (tf.Tensor): Ensemble forecasts, Shape (n_forecasts, n_members)
        member_weights (tf.Tensor, optional): Weights for ensemble members, Shape (n_forecasts, n_members). Default is None.
        handle_ties (["uniform", "exclude"], optional): How to handle ties between observation and members:
            - "uniform": distribute tied mass uniformly within the tie interval.
            - "exclude": omit tied cases.
            Defaults to "uniform".
        title (str, optional): Plot title. Default is "QQ Plot".
        ax (plt.Axes, optional): Axes object for plotting. If None, a new figure is created.

    Returns:
        plt.Axes: Axes object with the QQ-Plot.
    """
    # Extract shapes and prepare tensors
    n_forecasts, n_members = tf.unstack(tf.shape(ensemble_observations))

    if member_weights is None:
        member_weights = tf.ones_like(ensemble_observations) / tf.cast(n_members, tf.float32)

    # Ensure consistent dtypes
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Vectorized rank computation
    y_expanded = tf.expand_dims(y_true, axis=1)  # (n_forecasts, 1)

    # Create boolean masks for comparisons
    below_mask = ensemble_observations < y_expanded  # (n_forecasts, n_members)
    equal_mask = tf.abs(ensemble_observations - y_expanded) < 1e-8  # (n_forecasts, n_members)

    # Compute weights below and equal for all forecasts at once
    weight_below = tf.reduce_sum(member_weights * tf.cast(below_mask, tf.float32), axis=1)  # (n_forecasts,)
    weight_equal = tf.reduce_sum(member_weights * tf.cast(equal_mask, tf.float32), axis=1)  # (n_forecasts,)

    if handle_ties == "uniform":
        # Use forecast index for deterministic but uniform distribution
        indices = tf.range(n_forecasts, dtype=tf.float32)
        u = (indices + 0.5) / tf.cast(n_forecasts, tf.float32)  # Deterministic uniform [0,1]
        ranks = weight_below + weight_equal * u
    elif handle_ties == "exclude":
        # Create mask for non-tied cases
        non_tied_mask = weight_equal < 1e-8
        ranks = tf.boolean_mask(weight_below, non_tied_mask)
    else:
        raise ValueError(f"handle_ties has to be 'uniform' or 'exclude' but is '{handle_ties}'")

    # Sort ranks for QQ plot
    sorted_ranks = tf.sort(ranks)
    n_used = tf.shape(sorted_ranks)[0]

    # Theoretical quantiles (uniform distribution)
    theoretical = (tf.range(n_used, dtype=tf.float32) + 0.5) / tf.cast(n_used, tf.float32)

    # Create plot
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Convert to numpy for plotting
    theoretical_np = theoretical.numpy()
    sorted_ranks_np = sorted_ranks.numpy()

    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    ax.scatter(
        theoretical_np, sorted_ranks_np,
        alpha=0.5,
        label='Data'
    )

    ax.legend(loc="upper right")
    ax.set(
        xlabel="Theoretical Quantile",
        ylabel="Empirical Quantile",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
        aspect='equal',
    )
    ax.grid(True)
    return ax

def cdf_comparison(
    y_true: tf.Tensor,
    ensemble_observations: tf.Tensor,
    member_weights: tf.Tensor = None,
    num_points: int = 100,
    title: str = "CDF Comparison",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Creates a plot comparing the empirical cumulative distribution function (CDF) of observations
    with the aggregated CDF of ensemble forecasts.

    Args:
        y_true (tf.Tensor): Array of observations, Shape (n_forecasts,)
        ensemble_observations (tf.Tensor): Ensemble members, Shape (n_forecasts, n_members)
        member_weights (tf.Tensor): Weights for ensemble members,
            Shape (n_forecasts, n_members) (Default: Equal weighting)
        num_points (int): Number of points at which the CDF is evaluated (Default: 100)
        title (str): Plot title (Default: "CDF Comparison")
        ax (plt.Axes): Axes object for plotting (optional)

    Returns:
        plt.Axes: Axes object with the plot
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    member_weights = tf.cast(member_weights, dtype=tf.float32)

    # Extract shapes
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    # If no weights are specified: equally weighted ensemble members
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members

    # Create a common grid for CDF calculation
    min_val = min(tf.minimum(ensemble_observations), tf.minimum(y_true))
    max_val = max(tf.maximum(ensemble_observations), tf.maximum(y_true))
    x_grid = tf.linspace(min_val, max_val, num_points)

    # Calculate the aggregated CDF of ensemble forecasts
    forecast_vals = ensemble_observations.flatten()
    forecast_weights = member_weights.flatten()
    total_forecast_weight = tf.reduce_sum(forecast_weights)
    forecast_cdf = tf.cast([tf.reduce_sum(forecast_weights[forecast_vals <= x]) / total_forecast_weight for x in x_grid])

    # Calculate the empirical CDF of observations
    total_obs = len(y_true)
    obs_cdf = tf.cast([tf.reduce_sum(y_true <= x) / total_obs for x in x_grid])

    # Create the plot
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    ax.plot(x_grid, forecast_cdf, label="Ensemble CDF", lw=2)
    ax.plot(x_grid, obs_cdf, label="Observation CDF", lw=2, linestyle="--")

    ax.set(
        xlim=[min_val,max_val],
        ylim=[0, 1],
        xlabel="Value",
        ylabel="Cumulative Probability",
        title=title
    )
    ax.legend(loc="upper right")
    return ax

def attention_scatter(
        y_true: tf.Tensor,
        x: tf.Tensor,
        ensemble_observations: tf.Tensor,
        ensemble_forecasts: tf.Tensor = None,
        member_weights: tf.Tensor = None,
        title: str = None,
        y_lim: float = None,
        x_lim: float = None,
        ax: plt.Axes = None,
) -> plt.Axes:
    """
    Real-time update of the scatter plot with vertical lines for Observation and Forecast.
    Args:
        y_true (tf.Tensor): Observed values, Shape (n_samples,)
        x (tf.Tensor): Originally predicted values, Shape (n_samples,)
        ensemble_observations (tf.Tensor): Ensemble members, Shape (n_samples, n_members)
        ensemble_forecasts (tf.Tensor, optional): Ensemble Forecasts, Shape (n_samples, n_members)
        member_weights (tf.Tensor): Weights for ensemble members, Shape (n_samples, n_members)
        title (str): Plot title
        y_lim (float): Maximum Y-axis limit
        x_lim (float): Maximum X-axis limit
        ax (plt.Axes): Axes object for plotting
    Returns:
        plt.Axes: Axes object with the plot
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    # Extract shapes
    n_forecasts, n_members = ensemble_observations.shape
    # Activate live plot
    plt.ion()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    # Initialization of Y limits
    if y_lim:
        y_lower = 0
        y_upper = y_lim
    else:
        if member_weights:
            y_lower = 0
            y_upper = tf.maximum(member_weights)
    x_upper = x_lim if x_lim else tf.maximum(ensemble_observations)
    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members
        if y_lim is None:
            y_lower = tf.minimum(member_weights) * 0.9
            y_upper = tf.maximum(member_weights) * 1.1
    else:
        member_weights = tf.cast(member_weights, dtype=tf.float32)
    # Initial scatter plot for observations
    scatter_observations = ax.scatter([], [], alpha=0.5, label="Observation Member", color="red")
    # Initialize scatter plot for forecasts, if available
    if ensemble_forecasts:
        scatter_forecasts = ax.scatter([], [], alpha=0.5, label="Forecast Member", color="blue")
    # Vertical lines (initially empty)
    observation_line, = ax.plot([], [], color="red", alpha=0.5, label="Target", linestyle='--')
    forecast_line, = ax.plot([], [], color="blue", alpha=0.5, label="Original Forecast", linestyle='--')
    ax.set_title(title or "Scatter-Plot of Attention-Scores")
    ax.set_xlabel("Predicted Values (observations)")
    ax.set_ylabel("Weight")
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlim(0, x_upper)
    ax.legend(loc="upper right")
    fig.canvas.draw()
    n_samples, n_members = member_weights.shape
    try:
        for i in range(n_samples):
            current_observations = ensemble_observations[i, :]
            current_weights = member_weights[i, :]
            current_x = x[i]
            current_y_true = y_true[i]
            # Update observation data in scatter plot
            scatter_observations.set_offsets(tf.stack((current_observations, current_weights), axis=1))
            # Update forecast data in scatter plot, if available
            if ensemble_forecasts:
                current_forecasts = ensemble_forecasts[i, :]
                scatter_forecasts.set_offsets(tf.stack((current_forecasts, current_weights), axis=1))
            # Update vertical lines
            forecast_line.set_data([current_x, current_x], [0, y_upper])
            observation_line.set_data([current_y_true, current_y_true], [0, y_upper])

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.25)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        plt.ioff()
        plt.show()
    return ax

def attention_kde(
        y_true: tf.Tensor,
        x: tf.Tensor,
        ensemble_observations: tf.Tensor,
        ensemble_forecasts: tf.Tensor = None,
        member_weights: tf.Tensor = None,
        title: str = None,
        y_lim: float = None,
        x_lim: float = None,
        ax: plt.Axes = None,
) -> plt.Axes:
    """
    Real-time update of the kernel density estimation for forecasts and observations.
    Args:
        y_true (tf.Tensor): Observed values, Shape (n_samples,)
        x (tf.Tensor): Originally predicted values, Shape (n_samples,)
        ensemble_observations (tf.Tensor): Observation values of ensemble members, Shape (n_samples, n_members)
        ensemble_forecasts (tf.Tensor, optional): Forecast values of ensemble members, Shape (n_samples, n_members)
        member_weights (tf.Tensor, optional): Weights for both ensembles, Shape (n_samples, n_members)
        title (str, optional): Plot title
        y_lim (float, optional): Maximum Y-axis limit for density
        x_lim (float, optional): Maximum X-axis limit
        ax (plt.Axes, optional): Axes object for plotting
    Returns:
        plt.Axes: Axes object with the plot
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    x = tf.cast(x, dtype=tf.float32)
    ensemble_observations = tf.cast(ensemble_observations, dtype=tf.float32)
    ensemble_forecasts = tf.cast(ensemble_forecasts, dtype=tf.float32)

    # Extract shape
    n_forecasts, n_members = tf.shape(ensemble_observations).numpy()

    plt.ion()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    # Initialize Y limits
    if y_lim:
        y_lower = 0
        y_upper = y_lim
    else:
        if member_weights:
            y_lower = 0
            y_upper = tf.maximum(member_weights[0])

    x_upper = x_lim if x_lim else tf.maximum(ensemble_observations)
    x_grid = tf.linspace(0, x_upper, 100, dtype=tf.float32)

    if member_weights is None:
        member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32)/n_members
        if y_lim is None:
            y_lower = tf.minimum(member_weights) * 0.9
            y_upper = tf.maximum(member_weights) * 1.1
    # Uniform weights if not specified
    if member_weights is None:
        if ensemble_forecasts:
            member_weights = tf.ones(ensemble_forecasts.shape, dtype=tf.float32) / ensemble_forecasts.shape[1]
        else:
            member_weights = tf.ones((n_forecasts, n_members), dtype=tf.float32) / n_members
    else:
        member_weights = tf.cast(member_weights, dtype=tf.float32)
    # Line initialization
    if ensemble_forecasts:
        kde_forecast_line, = ax.plot([], [], color="blue", alpha=0.5, label="Forecast-Values KDE")
    kde_observations_line, = ax.plot([], [], color="red", alpha=0.5, label="Observed-Values KDE")
    forecast_line, = ax.plot([], [], color="blue", alpha=0.5, label="Original Forecast", linestyle='--')
    observation_line, = ax.plot([], [], color="red", alpha=0.5, label="Target", linestyle='--')

    ax.set_title(title or "KDE for Forecast- and Observation-Members")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Density")
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlim(0, x_upper)
    ax.legend(loc="upper right")
    fig.canvas.draw()

    n_samples = tf.shape(ensemble_observations).numpy()[0]
    try:
        for i in range(n_samples):
            # Data for the current timestamp
            if ensemble_forecasts:
                current_forecasts = ensemble_forecasts[i, :]
            current_observations_ens = ensemble_observations[i, :]
            current_weights = member_weights[i, :]
            current_x = x[i]
            current_y_true = y_true[i]
            # Calculate KDE
            if ensemble_forecasts:
                kde_forecast = gaussian_kde(current_forecasts, weights=current_weights)
                y_kde_forecast = kde_forecast(x_grid)
            kde_observations = gaussian_kde(current_observations_ens, weights=current_weights)
            y_kde_observations = kde_observations(x_grid)
            # Update lines
            if ensemble_forecasts:
                kde_forecast_line.set_data(x_grid, y_kde_forecast)
            kde_observations_line.set_data(x_grid, y_kde_observations)
            forecast_line.set_data([current_x, current_x], [0, y_upper])
            observation_line.set_data([current_y_true, current_y_true], [0, y_upper])
            # Dynamic Y-axis adjustment
            if y_lim is None:
                current_max = 0
                if ensemble_forecasts:
                    current_max = max(current_max, tf.maximum(y_kde_forecast))
                current_max = max(current_max, tf.maximum(y_kde_observations))
                new_y_upper = current_max * 1.1
                if new_y_upper > y_upper:
                    y_upper = new_y_upper
                    ax.set_ylim(y_lower, y_upper)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.25)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        plt.ioff()
        plt.show()
    return ax

def plot_training_history(history: tf.keras.callbacks.History,
                          model_name: str, save_path: str) -> None:
    """Plot and save training history."""
    fig, ax = plt.subplots(figsize=(12, 8))
    for metric, values in history.history.items():
        if metric != "lr":
            ax.plot(values, label=metric, alpha=0.7)
            min_val = min(values)
            epoch_min = values.index(min_val)
            ax.scatter(epoch_min, min_val, zorder=5)
            ax.annotate(
                f'{min_val:.4f}',
                xy=(epoch_min, min_val),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center'
            )
    ax.set_title(f'Training History: {model_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(os.path.dirname(save_path), f"{model_name}_training.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()