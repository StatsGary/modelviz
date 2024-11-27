import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_correlation_matrix(df, method='pearson',
                            figsize=(10, 8),
                            annot=True, cmap='BuGn',
                            max_columns=None, 
                            fmt=".2f", square=True, 
                            title='Correlation Matrix',
                            title_fontsize=14, title_y=1.03,
                            subtitle_fontsize=10, subtitle_y=0.01, subtitle_ha='center',
                            *args, **kwargs):
    """
    Plots a correlation matrix heatmap of numerical columns in a DataFrame.

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
    if not isinstance(annot, bool):
        raise TypeError("annot must be a boolean value.")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'.")

    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numerical columns to compute correlation.")

    if max_columns is not None and numeric_df.shape[1] > max_columns:
        numeric_df = numeric_df.iloc[:, :max_columns]
        subtitle = f"Filter applied: showing first {max_columns} columns"
    else:
        subtitle = None

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numerical columns to compute correlation.")

    corr_mat = numeric_df.corr(method=method)
    heatmap_params = {
        'data': corr_mat,
        'cmap': cmap,
        'annot': annot,
        'fmt': fmt,
        'square': square,
    }

    # Explicitly remove parameters specific to plot_correlation_matrix
    sns_compatible_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_columns']}

    overlapping_keys = set(heatmap_params.keys()) & set(sns_compatible_kwargs.keys())
    if overlapping_keys:
        print(f"Warning: Overriding default parameters with user-provided values for {overlapping_keys}")
    heatmap_params.update(sns_compatible_kwargs)

    plt.figure(figsize=figsize)
    sns.heatmap(*args, **heatmap_params)
    plt.title(title, fontsize=title_fontsize, y=title_y)
    if subtitle:
        plt.figtext(0.5, subtitle_y, subtitle, ha=subtitle_ha, fontsize=subtitle_fontsize, wrap=True)
    plt.show()