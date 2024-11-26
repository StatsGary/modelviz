import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve_with_threshold(fpr, tpr, thresholds, roc_auc, model_name=None,
                                  custom_threshold=0.5, custom_thresh_color='black',
                                  cust_thresh_label='Custom Threshold',
                                  x_label='False Positive Rate (FPR)',
                                  y_label='True Positive Rate (TPR)',
                                  show_grid=True):
    """
    Plots the ROC curve and annotates a custom threshold point.

    Parameters
    ----------
    fpr : array-like
        False Positive Rates obtained from `roc_curve`.
    tpr : array-like
        True Positive Rates obtained from `roc_curve`.
    thresholds : array-like
        Thresholds used to compute `fpr` and `tpr`.
    roc_auc : float
        Area Under the ROC Curve (AUC) score.
    model_name : str, optional
        Name of the model to display in the plot title.
    custom_threshold : float, optional
        The custom threshold value to annotate on the ROC curve. Default is 0.5.
    custom_thresh_color : str, optional
        Color of the annotated custom threshold point. Default is 'black'.
    cust_thresh_label : str, optional
        Label for the custom threshold point in the legend. Default is 'Custom Threshold'.
    x_label : str, optional
        Label for the X-axis. Default is 'False Positive Rate (FPR)'.
    y_label : str, optional
        Label for the Y-axis. Default is 'True Positive Rate (TPR)'.
    show_grid : bool, optional
        Whether to display a grid in the plot. Default is True.

    Raises
    ------
    ValueError
        If `fpr`, `tpr`, or `thresholds` are not array-like or are empty.
        If `custom_threshold` is not within the range of `thresholds`.
        If `roc_auc` is not a float between 0 and 1.
    TypeError
        If input types are not as expected.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import roc_curve, auc
    >>> y_true = [0, 0, 1, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8]
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    >>> roc_auc = auc(fpr, tpr)
    >>> plot_roc_curve_with_threshold(fpr, tpr, thresholds, roc_auc,
                                      model_name='Logistic Regression',
                                      custom_threshold=0.5)
    """
    # Input validation
    if not isinstance(fpr, (list, np.ndarray)) or len(fpr) == 0:
        raise ValueError("`fpr` must be a non-empty array-like object.")
    if not isinstance(tpr, (list, np.ndarray)) or len(tpr) == 0:
        raise ValueError("`tpr` must be a non-empty array-like object.")
    if not isinstance(thresholds, (list, np.ndarray)) or len(thresholds) == 0:
        raise ValueError("`thresholds` must be a non-empty array-like object.")
    if not isinstance(roc_auc, (float, int)) or not (0 <= roc_auc <= 1):
        raise ValueError("`roc_auc` must be a float between 0 and 1.")
    if not isinstance(custom_threshold, (float, int)):
        raise TypeError("`custom_threshold` must be a numeric value.")
    if not isinstance(custom_thresh_color, str):
        raise TypeError("`custom_thresh_color` must be a string representing a color.")
    if not isinstance(cust_thresh_label, str):
        raise TypeError("`cust_thresh_label` must be a string.")
    if not isinstance(x_label, str) or not isinstance(y_label, str):
        raise TypeError("`x_label` and `y_label` must be strings.")
    if not isinstance(show_grid, bool):
        raise TypeError("`show_grid` must be a boolean.")

    thresholds_array = np.array(thresholds)
    if custom_threshold < thresholds_array.min() or custom_threshold > thresholds_array.max():
        raise ValueError("`custom_threshold` is outside the range of `thresholds`.")

    # Find the index of the threshold closest to the custom threshold
    idx = np.argmin(np.abs(thresholds_array - custom_threshold))
    adjusted_fpr = fpr[idx]
    adjusted_tpr = tpr[idx]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.scatter(adjusted_fpr, adjusted_tpr, color=custom_thresh_color, s=100,
                label=f'{cust_thresh_label} = {custom_threshold}')
    plt.text(adjusted_fpr + 0.02, adjusted_tpr - 0.02,
             f'({adjusted_fpr:.2f}, {adjusted_tpr:.2f})', color=custom_thresh_color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if model_name:
        plt.title(f'ROC Curve for {model_name}')
    else:
        plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(show_grid)
    plt.show()


def plot_roc_curve_with_thresholds(fpr, tpr, thresholds, roc_auc, model_name=None, adjusted_threshold=0.5, youden_threshold=0.3):
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # Adjusted Threshold Annotation
    idx_adj = np.argmin(np.abs(thresholds - adjusted_threshold))
    adj_fpr = fpr[idx_adj]
    adj_tpr = tpr[idx_adj]
    plt.scatter(adj_fpr, adj_tpr, color='red', s=100, label=f'Adjusted Threshold = {adjusted_threshold}')
    plt.text(adj_fpr + 0.02, adj_tpr - 0.02, f'({adj_fpr:.2f}, {adj_tpr:.2f})', color='red', fontsize=9)
    
    # Youden's J Threshold Annotation
    idx_youden = np.argmin(np.abs(thresholds - youden_threshold))
    youden_fpr = fpr[idx_youden]
    youden_tpr = tpr[idx_youden]
    plt.scatter(youden_fpr, youden_tpr, color='green', s=100, label=f"Youden's J Threshold = {youden_threshold:.4f}")
    plt.text(youden_fpr + 0.02, youden_tpr - 0.02, f'({youden_fpr:.2f}, {youden_tpr:.2f})', color='green', fontsize=9)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


