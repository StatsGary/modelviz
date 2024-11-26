import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, columns=None, method='pearson',
                            figsize=(10, 8),
                            annot=True, cmap='BuGn', *args, **kwargs):
    """
    Plots a correlation matrix heatmap of numerical columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    columns : list or None, optional
        A list of column names or indices to include in the correlation matrix.
        If None, all numerical columns are used. Default is None.
    method : str, optional
        Method of correlation: 'pearson', 'spearman', or 'kendall'. Default is 'pearson'.
    figsize : tuple of (float, float), optional
        Width and height of the figure in inches. Default is (10, 8).
    annot : bool, optional
        Whether to annotate the heatmap with correlation coefficients. Default is True.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use for the heatmap. Default is 'BuGn'.
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
        If `columns` is not a list, tuple, or None.
        If `figsize` is not a tuple of two numbers.
        If `annot` is not a boolean.
        If `method` is not a string.
        If `cmap` is neither a string nor a valid colormap instance.
    ValueError
        If `figsize` does not have exactly two elements.
        If `method` is not one of 'pearson', 'spearman', or 'kendall'.
        If there are less than two numerical columns in `df`.
        If any of the specified `columns` are not in `df`.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [4, 3, 2, 1],
    ...     'C': [5, 6, 7, 8]
    ... })
    >>> plot_correlation_matrix(df, columns=['A', 'B'])
    """

    # Error handling
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if columns is not None and not isinstance(columns, (list, tuple)):
        raise TypeError("columns must be a list, tuple, or None.")
    if not isinstance(figsize, tuple):
        raise TypeError("figsize must be a tuple of two numbers.")
    if len(figsize) != 2:
        raise ValueError("figsize must be a tuple of two numbers.")
    if not all(isinstance(dim, (int, float)) for dim in figsize):
        raise TypeError("figsize dimensions must be numbers.")
    if not isinstance(annot, bool):
        raise TypeError("annot must be a boolean value.")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'.")


    if isinstance(cmap, str):
        if cmap not in plt.colormaps():
            raise ValueError(f"'{cmap}' is not a valid colormap name.")
    elif not callable(cmap):
        raise TypeError("cmap must be a string or a valid matplotlib colormap instance.")

    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numerical columns to compute correlation.")

    if columns is not None:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        numeric_df = numeric_df[columns]
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least two numerical columns to compute correlation.")
    
    
    corr_mat = numeric_df.corr(method=method)

    # Prepare parameters for sns.heatmap
    heatmap_params = {
        'data': corr_mat,
        'cmap': cmap,
        'annot': annot,
        'fmt': ".2f",
        'square': True,
    }

    # Handle overlapping keys
    overlapping_keys = set(heatmap_params.keys()) & set(kwargs.keys())
    if overlapping_keys:
        print(f"Warning: Overriding default parameters with user-provided values for {overlapping_keys}")

    # Update heatmap_params with kwargs, user-provided values take precedence
    heatmap_params.update(kwargs)

    # Create the figure
    plt.figure(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(*args, **heatmap_params)

    # Set plot title
    plt.title(f'Correlation Matrix ({method.capitalize()} Method)')

    # Display the plot
    plt.show()
