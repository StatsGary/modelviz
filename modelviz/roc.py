import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, 
                                           adjusted_threshold=None, youden_threshold=None, 
                                           figsize=(8, 6), annotation_offset=(0.02, 0.02),
                                           scatter_size=100, annotation_fontsize=9,
                                           line_style_random='k--', legend_loc='lower right',
                                           show_grid=True, title_fontsize=10, threshold_col='red',
                                           youden_col='green', save_path=None,
                                           xlabel="False Positive Rate", ylabel="True Positive Rate",
                                           title=None):
    """
    Plot an ROC curve with annotations for adjusted and Youden's J thresholds.

    Parameters
    ----------
    fpr : array-like
        False positive rates from the ROC analysis.
    tpr : array-like
        True positive rates from the ROC analysis.
    thresholds : array-like
        Thresholds corresponding to the ROC curve.
    roc_auc : float
        Area under the ROC curve (AUC).
    model_name : str
        Name of the model for the plot title.
    adjusted_threshold : float, optional
        Threshold chosen manually for annotation. If None, no adjusted threshold is plotted.
    youden_threshold : float, optional
        Threshold calculated using Youden's J statistic. If None, no Youden's threshold is plotted.
    figsize : tuple, optional
        Size of the figure. Default is (8, 6).
    annotation_offset : tuple, optional
        Offset for annotation text near the threshold points (x_offset, y_offset). Default is (0.02, 0.02).
    scatter_size : int, optional
        Size of the scatter plot points for threshold annotations. Default is 100.
    annotation_fontsize : int, optional
        Font size of the annotation text. Default is 9.
    line_style_random : str, optional
        Line style for the random guess line. Default is 'k--'.
    legend_loc : str, optional
        Location of the legend on the plot. Default is 'lower right'.
    show_grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    save_path : str, optional
        File path to save the plot. If None, the plot is not saved. Default is None.
    xlabel : str, optional
        Label for the X-axis. Default is "False Positive Rate".
    ylabel : str, optional
        Label for the Y-axis. Default is "True Positive Rate".
    title : str, optional
        Title of the plot. Default is "ROC Curve for {model_name}".

    Raises
    ------
    ValueError
        If `fpr`, `tpr`, or `thresholds` have inconsistent lengths.
        If `adjusted_threshold` or `youden_threshold` is out of range.
    TypeError
        If inputs are not of the correct type.

    Creator
    -------
    Created by Gary Hutson
    GitHub: https://github.com/StatsGary/modelviz

    Example
    -------
    >>> fpr = [0.0, 0.1, 0.2, 0.3]
    >>> tpr = [0.0, 0.4, 0.6, 1.0]
    >>> thresholds = [1.0, 0.8, 0.5, 0.2]
    >>> roc_auc = 0.85
    >>> plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, "My Model", adjusted_threshold=0.5, youden_threshold=0.8)
    """
    # Error handling
    if not (isinstance(fpr, (list, np.ndarray)) and isinstance(tpr, (list, np.ndarray)) and isinstance(thresholds, (list, np.ndarray))):
        raise TypeError("fpr, tpr, and thresholds must be array-like.")
    
    if len(fpr) != len(tpr) or len(fpr) != len(thresholds):
        raise ValueError("fpr, tpr, and thresholds must have the same length.")
    
    if adjusted_threshold is not None and (adjusted_threshold < thresholds.min() or adjusted_threshold > thresholds.max()):
        raise ValueError("adjusted_threshold must be within the range of provided thresholds.")
    
    if youden_threshold is not None and (youden_threshold < thresholds.min() or youden_threshold > thresholds.max()):
        raise ValueError("youden_threshold must be within the range of provided thresholds.")


    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], line_style_random, label='No predictive power')

    # Annotation for adjusted threshold
    if adjusted_threshold is not None:
        idx_adj = np.argmin(np.abs(thresholds - adjusted_threshold))
        adj_fpr = fpr[idx_adj]
        adj_tpr = tpr[idx_adj]
        plt.scatter(adj_fpr, adj_tpr, color=threshold_col, s=scatter_size, label=f'Adjusted Threshold = {adjusted_threshold:.4f}')
        plt.text(adj_fpr + annotation_offset[0], adj_tpr - annotation_offset[1],
                 f'({adj_fpr:.2f}, {adj_tpr:.2f})', color=threshold_col, fontsize=annotation_fontsize)

    # Annotation for Youden's J threshold
    if youden_threshold is not None:
        idx_youden = np.argmin(np.abs(thresholds - youden_threshold))
        youden_fpr = fpr[idx_youden]
        youden_tpr = tpr[idx_youden]
        plt.scatter(youden_fpr, youden_tpr, color=youden_col, s=scatter_size, label=f"Youden's J Threshold = {youden_threshold:.4f}")
        plt.text(youden_fpr + annotation_offset[0], youden_tpr - annotation_offset[1],
                 f'({youden_fpr:.2f}, {youden_tpr:.2f})', color=youden_col, fontsize=annotation_fontsize)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = f"ROC Curve for {model_name}"
    plt.title(title, fontsize=title_fontsize)
    plt.legend(loc=legend_loc)
    if show_grid:
        plt.grid(True)

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()