import matplotlib.pyplot as plt
import numpy as np

def multi_model_visualizer(results_df,
                           subplot_num=3,
                           font_size=10,
                           font_color='black',
                           metrics=None,
                           colors=None):
    """
    Visualizes multiple metrics from a results DataFrame using horizontal bar charts.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing the results to visualize. Must contain a 'Model' column and 'Metric' columns.
    metrics : list of str, optional
        List of metric names (column names in results_df) to visualize. If None, all columns except 'Model' are used.
    colors : list of str, optional
        List of colors to use for the plots. If None, a default color palette is used.

    Raises
    ------
    TypeError
        If results_df is not a pandas DataFrame.
        If metrics is not a list of strings.
        If colors is not a list of strings.
    ValueError
        If 'Model' column is not present in results_df.
        If any of the specified metrics are not columns in results_df.

    Examples
    --------
    >>> import pandas as pd
    >>> results = pd.DataFrame({
    ...     'Model': ['Model A', 'Model B', 'Model C'],
    ...     'Accuracy': [0.9, 0.85, 0.88],
    ...     'Precision': [0.8, 0.75, 0.78],
    ...     'Recall': [0.7, 0.65, 0.68]
    ... })
    >>> multi_model_visualizer(results)
    Creator
    -------
    Created by Gary Hutson
    GitHub: https://github.com/StatsGary/modelviz
    """
    import pandas as pd

    # Error handling
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")
    if 'Model' not in results_df.columns:
        raise ValueError("results_df must contain a 'Model' column.")
    if metrics is not None:
        if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
            raise TypeError("metrics must be a list of strings.")
        missing_metrics = [m for m in metrics if m not in results_df.columns]
        if missing_metrics:
            raise ValueError(f"The following metrics are not in results_df: {missing_metrics}")
    else:
        # Use all columns except 'Model' if metrics are not specified
        metrics = [col for col in results_df.columns if col != 'Model']

    if colors is not None:
        if not isinstance(colors, list) or not all(isinstance(c, str) for c in colors):
            raise TypeError("colors must be a list of strings.")
    else:
        # Default color palette
        colors = ["lightcoral", "cornflowerblue", "mediumseagreen", "mediumpurple", "gold", "sandybrown", "skyblue"]

    # Adjust colors to match the number of metrics
    if len(colors) < len(metrics):
        # Repeat colors if not enough
        colors = colors * (len(metrics) // len(colors) + 1)
    colors = colors[:len(metrics)]

    # Determine the layout of subplots
    num_metrics = len(metrics)
    num_cols = subplot_num
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Ceiling division

    fig_height = num_rows * 5  # Adjust the figure height based on rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, fig_height), sharex=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        bars = axes[i].barh(results_df["Model"], results_df[metric], color=colors[i])
        axes[i].set_title(f"{metric} by Model")
        axes[i].set_xlabel(metric)
        axes[i].invert_yaxis()  # Best-performing models appear at the top

        # Add data labels to each bar
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width + 0.02 * max(results_df[metric]), bar.get_y() + bar.get_height() / 2,
                         f"{width:.3f}", ha="left",
                         va="center", color=font_color, fontsize=font_size
                         )

    # Remove any unused subplots
    if len(axes) > num_metrics:
        for j in range(num_metrics, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()