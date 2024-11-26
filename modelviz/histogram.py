import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def plot_feature_histograms(df, hist_bins=30, edge_col='black', hist_col='skyblue',
                            exclude_bin_encode=True, library='matplotlib',
                            y_label='Frequency'):
    """
    Plots histograms for numeric columns in a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be visualized.
    hist_bins : int, optional, default=30
        Number of bins to use in the histograms.
    edge_col : str, optional, default='black'
        Color of the edges of the histogram bars.
    hist_col : str, optional, default='skyblue'
        Color of the histogram bars.
    exclude_bin_encode : bool, optional, default=True
        Whether to exclude binary encoded columns (values 0 and 1) from the numeric columns.
    library : str, optional, default='matplotlib'
        Visualization library to use. Options are 'matplotlib' or 'plotly'.

    Raises:
    ------
    ValueError
        If an invalid library is specified.
        If no numeric columns are found.
        If the DataFrame is empty.
        If `hist_bins` is not a positive integer.
    TypeError
        If `df` is not a pandas DataFrame.

    Notes:
    ------
    - When `library='matplotlib'`, histograms for all numeric columns are displayed in a grid.
    - When `library='plotly'`, each numeric column is displayed in an individual histogram.

    Examples:
    --------
    >>> import pandas as pd
    >>> data = {'col1': [1, 2, 3], 'col2': [4.0, 5.5, 6.7], 'col3': [0, 1, 0]}
    >>> df = pd.DataFrame(data)
    >>> plot_feature_histograms(df, library='matplotlib')

    Author:
    -------
    Gary Hutson
    """
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Validate hist_bins
    if not isinstance(hist_bins, int) or hist_bins <= 0:
        raise ValueError("`hist_bins` must be a positive integer.")

    # Validate color inputs
    if not isinstance(hist_col, str) or not isinstance(edge_col, str):
        raise ValueError("`hist_col` and `edge_col` must be valid color strings.")

    # Ensure numeric_cols is a list
    if exclude_bin_encode:
        numeric_cols = [col for col in df.select_dtypes(include='number').columns
                        if not df[col].dropna().isin([0, 1]).all()]
    else:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Check if numeric_cols is empty
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to plot.")

    # Generate histograms
    if library == 'matplotlib':
        num_cols = len(numeric_cols)
        fig, axes = plt.subplots(nrows=(num_cols // 3) + 1, ncols=3, figsize=(15, num_cols * 2.5))
        axes = axes.flatten()

        for ax_loc, col in enumerate(numeric_cols):
            df[col].hist(bins=hist_bins, color=hist_col, edgecolor=edge_col, ax=axes[ax_loc])
            axes[ax_loc].set_title(f"Histogram: {col}")
            axes[ax_loc].set_xlabel(col)
            axes[ax_loc].set_ylabel(y_label)

        for j in range(ax_loc + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    elif library == 'plotly':
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=hist_bins, title=f"Histogram: {col}")
            fig.update_traces(marker=dict(color=hist_col, line=dict(color=edge_col, width=1)))
            fig.update_layout(xaxis_title=col, yaxis_title=y_label)
            fig.show()
    else:
        raise ValueError("Invalid library specified. Choose 'matplotlib' or 'plotly'.")
