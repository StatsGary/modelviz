import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


def plot_similarity(
    data, point_of_interest, mode="gaussian", 
    std_range=3, perplexity=30, random_state=42, 
    pca_components=2, seaborn_style="darkgrid",
    scatter_title="Original Multi-Dimensional Space", 
    gaussian_title="Gaussian Similarity Distribution",
    tsne_title="t-SNE Projection", 
    pca_title="PCA Projection",
    data_color="blue", reference_color="red", 
    similarity_color="green", curve_color="purple",
    line_style="--", line_width=2, scatter_size=100
):
    """
    Computes and visualizes either Gaussian similarity, t-SNE, or PCA.
    """

    # Apply Seaborn style
    sns.set_style(seaborn_style)

    if mode == "gaussian":
        def gaussian_similarity(data, point_of_interest):
            distances = np.linalg.norm(data - point_of_interest, axis=1)
            mu, sigma = np.mean(distances), np.std(distances)
            similarity_scores = norm.pdf(distances, mu, sigma)
            return distances, similarity_scores, mu, sigma

        distances, similarity_scores, mu, sigma = gaussian_similarity(data, point_of_interest)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Scatter plot of original space
        ax[0].scatter(data[:, 0], data[:, 1], c=data_color, alpha=0.7, label="Data Points")
        ax[0].scatter(point_of_interest[0], point_of_interest[1], c=reference_color, s=130, edgecolors='black', label="Point of Interest")
        for i, d in enumerate(distances):
            ax[0].plot([point_of_interest[0], data[i, 0]], [point_of_interest[1], data[i, 1]], 'k--', alpha=0.4)

        ax[0].set_title(scatter_title, fontsize=14, fontweight="bold")
        ax[0].set_xlabel("Feature 1")  
        ax[0].set_ylabel("Feature 2")  
        ax[0].legend()

        # Right: Gaussian Bell Curve
        x_vals = np.linspace(mu - std_range * sigma, mu + std_range * sigma, 100)
        y_vals = norm.pdf(x_vals, mu, sigma)

        ax[1].plot(x_vals, y_vals, color=curve_color, linestyle=line_style, linewidth=line_width, label="Gaussian PDF")
        ax[1].scatter(distances, similarity_scores, c=similarity_color, s=scatter_size, edgecolors="black", zorder=3, label="Similarity Scores")

        ax[1].set_title(gaussian_title, fontsize=14, fontweight="bold")
        ax[1].set_xlabel("Distance from Reference Point")  
        ax[1].set_ylabel("Similarity Score")  
        ax[1].legend()

        plt.show()
        return distances, similarity_scores

    elif mode == "tsne":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        transformed_data = tsne.fit_transform(data)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=data_color, alpha=0.7, label="t-SNE Data")
        ax.scatter(transformed_data[0, 0], transformed_data[0, 1], c=reference_color, s=130, edgecolors="black", label="Reference Point")
        ax.set_title(tsne_title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Component 1")  
        ax.set_ylabel("t-SNE Component 2")  
        ax.legend()

        plt.show()
        return transformed_data

    elif mode == "pca":
        pca = PCA(n_components=pca_components)
        transformed_data = pca.fit_transform(data)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=data_color, alpha=0.7, label="PCA Data")
        ax.scatter(transformed_data[0, 0], transformed_data[0, 1], c=reference_color, s=130, edgecolors="black", label="Reference Point")
        ax.set_title(pca_title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Principal Component 1")  
        ax.set_ylabel("Principal Component 2")  
        ax.legend()

        plt.show()
        return transformed_data

    else:
        raise ValueError("Invalid mode. Choose 'gaussian', 'tsne', or 'pca'.")
    
def plot_model_probs_scatter(
    probs_pairs,
    labels=None,
    palette="Set2",
    markers=None,
    xlabel="Model X Probability",
    ylabel="Model Y Probability",
    title=None,
    x_thresh=None,
    y_thresh=None,
    thresh_linewidth=1.5,
    x_thresh_color="blue",
    y_thresh_color="green",
    x_thresh_label=None,
    y_thresh_label=None,
    show_diagonal=False,
    diagonal_color="gray",
    diagonal_style="--",
    alpha=0.7,
    figsize=(6,6),
    ax=None,
    show=True,
    show_legend=True,
    legend_loc='best'
):
    """
    Seaborn-style scatter plot for one or more pairs of probability arrays (e.g., model predictions).

    Parameters
    ----------
    probs_pairs : list of tuple of np.ndarray
        List of (probsX, probsY) pairs, each a 1D array of the same length.
    labels : list of str, optional
        List of labels for each pair (for the legend).
    palette : str or list, optional
        Seaborn color palette name or list of colors.
    markers : list, optional
        List of marker styles for each pair.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str, optional
        Plot title.
    x_thresh, y_thresh : float, optional
        Draw a vertical/horizontal threshold line at this value.
    x_thresh_color, y_thresh_color : str, optional
        Colors for threshold lines.
    x_thresh_label, y_thresh_label : str, optional
        Labels for threshold lines.
    show_diagonal : bool, optional
        Whether to show the x=y diagonal.
    alpha : float, optional
        Marker transparency.
    figsize : tuple, optional
        Figure size.
    ax : matplotlib.axes.Axes, optional
        If provided, plot on these axes.
    show : bool, optional
        If True, call plt.show().
    show_legend : bool, optional
        If True, show the legend.
    legend_loc : str, optional
        Legend location.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.

    Example
    -------
    >>> import numpy as np
    >>> from modelviz.relationships import plot_model_probs_scatter
    >>> probs1 = np.random.rand(100)
    >>> probs2 = 0.5 * np.random.rand(100) + 0.5 * probs1
    >>> plot_model_probs_scatter(
    ...     [(probs1, probs2)],
    ...     labels=['Model A vs B'],
    ...     x_thresh=0.5,
    ...     y_thresh=0.5,
    ...     show_diagonal=True
    ... )
    """
    sns.set(style="whitegrid")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_pairs = len(probs_pairs)
    colors = sns.color_palette(palette, n_colors=n_pairs)
    if markers is None:
        markers = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "*"] * ((n_pairs//10)+1)
    if labels is None:
        labels = [f"Pair {i+1}" for i in range(n_pairs)]

    for i, ((probs1, probs2), label, color, marker) in enumerate(zip(probs_pairs, labels, colors, markers)):
        probs1 = np.asarray(probs1).flatten()
        probs2 = np.asarray(probs2).flatten()
        if probs1.shape != probs2.shape:
            raise ValueError(f"Shape mismatch for pair {i}: {probs1.shape} vs {probs2.shape}")
        sns.scatterplot(
            x=probs1,
            y=probs2,
            label=label,
            color=color,
            marker=marker,
            alpha=alpha,
            ax=ax,
            s=70,  # slightly larger points
            edgecolor='w'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not title:
        title = f"{xlabel} vs {ylabel}"
    ax.set_title(title)

    if show_diagonal:
        ax.plot([0, 1], [0, 1], diagonal_style, color=diagonal_color, label='Diagonal (Agreement)')
    if x_thresh is not None:
        xl = x_thresh_label if x_thresh_label else f'X Threshold: {x_thresh:.2f}'
        ax.axvline(x=x_thresh, color=x_thresh_color, linestyle=':', linewidth=thresh_linewidth, label=xl)
    if y_thresh is not None:
        yl = y_thresh_label if y_thresh_label else f'Y Threshold: {y_thresh:.2f}'
        ax.axhline(y=y_thresh, color=y_thresh_color, linestyle=':', linewidth=thresh_linewidth, label=yl)
    if show_legend:
        ax.legend(loc=legend_loc)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    ax.grid(True)
    if show:
        plt.show()
    return ax