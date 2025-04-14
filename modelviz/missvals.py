import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

def plot_missing_values_heatmap(df, figsize=(10, 8),
                                cmap='BuGn', cbar=True,
                                y_tick_labels=False,
                                x_tick_labels=True,
                                plt_title='Missing values heatmap',
                                plt_x_label='Columns',
                                *args, **kwargs):
    """
    Plots a heatmap showing the location of missing values in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to visualize for missing values.
    figsize : tuple of (float, float), optional
        Width and height of the figure in inches. Default is (10, 8).
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use for the heatmap. Default is 'BuGn'.
    cbar : bool, optional
        Whether to draw a colorbar. Default is True.
    y_tick_labels : bool or list, optional
        Whether to display y-axis tick labels. Default is False.
    x_tick_labels : bool or list, optional
        Whether to display x-axis tick labels. Default is True.
    plt_title : str, optional
        Title of the plot. Default is 'Missing values heatmap'.
    plt_x_label : str, optional
        Label for the x-axis. Default is 'Columns'.
    *args
        Additional positional arguments passed to `seaborn.heatmap`.
    **kwargs
        Additional keyword arguments passed to `seaborn.heatmap`. If any of these
        keys overlap with the function parameters, the values in `**kwargs` will
        take precedence.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
        If `figsize` is not a tuple of two numbers.
        If `cbar` is not a boolean.
        If `y_tick_labels` or `x_tick_labels` are not of expected types.
        If `cmap` is not a valid colormap name or instance.
        If `plt_title` or `plt_x_label` are not strings.
    ValueError
        If `figsize` does not have exactly two elements.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    >>> plot_missing_values_heatmap(df, plt_title='My Missing Data', plt_x_label='Features')

    Creator
    -------
    Created by Gary Hutson
    GitHub: https://github.com/StatsGary/modelviz
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(figsize, tuple):
        raise TypeError("figsize must be a tuple of two numbers.")
    if len(figsize) != 2:
        raise ValueError("figsize must be a tuple of two numbers.")
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise TypeError("figsize dimensions must be numbers.")

    if not isinstance(cbar, bool):
        raise TypeError("cbar must be a boolean value.")

    if not isinstance(y_tick_labels, (bool, list)):
        raise TypeError("y_tick_labels must be a boolean or a list.")

    if not isinstance(x_tick_labels, (bool, list)):
        raise TypeError("x_tick_labels must be a boolean or a list.")

    if not isinstance(plt_title, str):
        raise TypeError("plt_title must be a string.")

    if not isinstance(plt_x_label, str):
        raise TypeError("plt_x_label must be a string.")
    if isinstance(cmap, str):
        if cmap not in plt.colormaps():
            raise ValueError(f"'{cmap}' is not a valid colormap name.")
    elif not isinstance(cmap, Colormap):
        raise TypeError("cmap must be a valid colormap name or Colormap instance.")

    # Prepare parameters for sns.heatmap
    heatmap_params = {
        'data': df.isnull(),
        'cmap': cmap,
        'cbar': cbar,
        'yticklabels': y_tick_labels,
        'xticklabels': x_tick_labels,
    }

    overlapping_keys = set(heatmap_params.keys()) & set(kwargs.keys())
    if overlapping_keys:
        print(f"Warning: Overriding default parameters with user-provided values for {overlapping_keys}")

    heatmap_params.update(kwargs)

    # Create the figure with the specified figsize
    plt.figure(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(*args, **heatmap_params)
    plt.title(plt_title)
    plt.xlabel(plt_x_label)
    plt.show()