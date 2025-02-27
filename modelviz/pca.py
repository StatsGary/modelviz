import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

def plot_pca_projection(
    df_original, df_projected, 
    figsize=(8, 6), alpha=0.5, 
    original_label="Original Data", 
    projected_label="Projected Data", 
    original_color="gray", 
    projected_color="blue", 
    line_color="red", 
    title_2d="PCA Projection (2D)", 
    title_3d="PCA Projection (3D)", 
    title_nD="PCA Pairwise Scatter Plot",
    labelpad=10  
):
    """
    Plots a PCA projection of data in 2D, 3D, or higher dimensions.

    Parameters:
    - df_original: DataFrame with original data.
    - df_projected: DataFrame with PCA-transformed data.
    - figsize: Tuple, figure size (default: (8,6)).
    - alpha: Transparency level for points and lines (default: 0.5).
    - original_label: Label for original dataset points (default: "Original Data").
    - projected_label: Label for projected dataset points (default: "Projected Data").
    - original_color: Color for original data points (default: 'gray').
    - projected_color: Color for projected data points (default: 'blue').
    - line_color: Color for connecting lines (default: 'red').
    - title_2d: Title for 2D PCA plots (default: "PCA Projection (2D)").
    - title_3d: Title for 3D PCA plots (default: "PCA Projection (3D)").
    - title_nD: Title for nD (4+ components) pairwise scatter plot (default: "PCA Pairwise Scatter Plot").
    - labelpad: Padding for axis labels to avoid overlap (default: 10).
    """

    num_components = df_projected.shape[1]

    if num_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(df_original.iloc[:, 0], df_original.iloc[:, 1], 
                   color=original_color, alpha=alpha, label=original_label)
        ax.scatter(df_projected.iloc[:, 0], df_projected.iloc[:, 1], 
                   color=projected_color, alpha=alpha + 0.2, label=projected_label)

        for i in range(len(df_original)):
            ax.plot([df_original.iloc[i, 0], df_projected.iloc[i, 0]], 
                    [df_original.iloc[i, 1], df_projected.iloc[i, 1]], 
                    linestyle="--", color=line_color, alpha=alpha)

        ax.set_xlabel("Principal Component 1", labelpad=labelpad)
        ax.set_ylabel("Principal Component 2", labelpad=labelpad)
        ax.set_title(title_2d)
        ax.legend()
        plt.show()

    elif num_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(df_original.iloc[:, 0], df_original.iloc[:, 1], df_original.iloc[:, 2], 
                   color=original_color, alpha=alpha, label=original_label)
        ax.scatter(df_projected.iloc[:, 0], df_projected.iloc[:, 1], df_projected.iloc[:, 2], 
                   color=projected_color, alpha=alpha + 0.2, label=projected_label)

        for i in range(len(df_original)):
            ax.plot([df_original.iloc[i, 0], df_projected.iloc[i, 0]], 
                    [df_original.iloc[i, 1], df_projected.iloc[i, 1]],
                    [df_original.iloc[i, 2], df_projected.iloc[i, 2]], 
                    linestyle="--", color=line_color, alpha=alpha)

        ax.set_xlabel("Principal Component 1", labelpad=labelpad)
        ax.set_ylabel("Principal Component 2", labelpad=labelpad)
        ax.set_zlabel("Principal Component 3", labelpad=labelpad)
        ax.set_title(title_3d)

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)

        plt.show()

    else:
        df_combined = df_projected.copy()
        df_combined['Category'] = projected_label
        df_original_copy = df_original.iloc[:, :num_components].copy()
        df_original_copy['Category'] = original_label
        df_all = pd.concat([df_combined, df_original_copy])

        sns.pairplot(df_all, hue='Category', diag_kind="hist", height=1.5, 
                     plot_kws={'alpha': alpha})
        plt.suptitle(title_nD, y=1.02)
        plt.show()
