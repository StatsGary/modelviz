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

