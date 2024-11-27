import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score
)

def plot_confusion_matrix(
    cm, classes, model_name, 
    normalize=False, cmap='Blues',
    figsize=(10, 10), annot_fontsize=12, title_fontsize=16, label_fontsize=12,
    cell_fontsize=10, table_fontsize=10, proportions_color='black',
    bbox_table=[0.0, -0.4, 1.0, 0.3], adjust_bottom=0.4, label_positions_color='gray'
):
    """
    Plot a confusion matrix with metrics displayed as a table below the visualization.

    Parameters
    ----------
    cm : array-like
        Confusion matrix as a NumPy array.
    classes : list
        List of class names.
    model_name : str
        Name of the model for the plot title.
    normalize : bool, optional
        Whether to normalize the confusion matrix by row (true class). Default is False.
    cmap : str, optional
        Colormap for the heatmap. Default is 'Blues'.
    figsize : tuple, optional
        Size of the figure. Default is (10, 10).
    annot_fontsize : int, optional
        Font size for annotations inside the confusion matrix. Default is 12.
    title_fontsize : int, optional
        Font size for the plot title. Default is 16.
    label_fontsize : int, optional
        Font size for axis labels. Default is 12.
    cell_fontsize : int, optional
        Font size for proportions inside confusion matrix cells. Default is 10.
    table_fontsize : int, optional
        Font size for the metrics table. Default is 10.
    proportions_color : str, optional
        Color for the proportions appended inside the confusion matrix cells. Default is 'black'.
    bbox_table : list, optional
        Bounding box for the table [left, bottom, width, height]. Default is [0.0, -0.4, 1.0, 0.3].
    adjust_bottom : float, optional
        Adjustment for the plot bottom to make space for the table. Default is 0.4.
    label_positions_color : str, optional
        Color for TP, FN, FP, TN labels in binary confusion matrix. Default is 'gray'.

    Returns
    -------
    None. Displays the confusion matrix plot and metrics summary as a table.

    Creator
    -------
    Created by Gary Hutson
    GitHub: https://github.com/StatsGary/modelviz
    
    """
    # Normalize confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f'Normalized Confusion Matrix for {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix for {model_name}'

    # Calculate proportions
    cm_total = cm.sum()
    proportions = cm / cm_total

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": annot_fontsize}, ax=ax)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel('True Label', fontsize=label_fontsize)
    ax.set_xlabel('Predicted Label', fontsize=label_fontsize)

    # Add proportions to the confusion matrix cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            prop_text = f"({proportions[i, j]:.2%})"
            ax.text(j + 0.5, i + 0.7, prop_text, ha='center', va='center', color=proportions_color, fontsize=cell_fontsize)

    # Add TP, FP, FN, TN for binary classification
    if len(classes) == 2:
        tp_fp_fn_tn = np.array([["TN", "FP"], ["FN", "TP"]])
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.3, tp_fp_fn_tn[i, j], ha='center', va='center', color=label_positions_color, fontsize=cell_fontsize)

    # Metrics Calculation
    metrics_table = metrics_summary(cm, classes)

    # Add the table below the confusion matrix
    table = plt.table(cellText=metrics_table.values,
                      colLabels=metrics_table.columns,
                      loc='bottom',
                      bbox=bbox_table,
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    plt.subplots_adjust(bottom=adjust_bottom)  # Adjust space for the table
    plt.show()

def metrics_summary(cm, classes):
    """
    Calculate confusion matrix summary metrics.

    Parameters
    ----------
    cm : array-like
        Confusion matrix as a NumPy array.
    classes : list
        List of class names.

    Returns
    -------
    metrics_table : pandas.DataFrame
        DataFrame containing calculated metrics for display as a table.
    """
    import pandas as pd

    # Flatten the confusion matrix into y_true and y_pred for multi-class metrics
    y_true = []
    y_pred = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            y_true.extend([i] * cm[i, j])
            y_pred.extend([j] * cm[i, j])

    # Calculate metrics
    metrics = {
        "Metric": [],
        "Value": []
    }

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics["Metric"].append("Accuracy")
    metrics["Value"].append(f"{accuracy:.4f}")

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    metrics["Metric"].append("Balanced Accuracy")
    metrics["Value"].append(f"{balanced_acc:.4f}")

    # Cohen's Kappa
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    metrics["Metric"].append("Cohen's Kappa")
    metrics["Value"].append(f"{cohen_kappa:.4f}")

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics["Metric"].append("Matthews Correlation Coefficient")
    metrics["Value"].append(f"{mcc:.4f}")

    # Precision, Recall, F1 Scores (Micro/Macro)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    metrics["Metric"].extend(["Micro Precision", "Micro Recall", "Micro F1 Score",
                              "Macro Precision", "Macro Recall", "Macro F1 Score"])
    metrics["Value"].extend([f"{precision_micro:.4f}", f"{recall_micro:.4f}", f"{f1_micro:.4f}",
                              f"{precision_macro:.4f}", f"{recall_macro:.4f}", f"{f1_macro:.4f}"])

    # Specificity for binary classification
    if len(classes) == 2:
        TN, FP, FN, TP = cm.ravel()

        specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
        metrics["Metric"].append("Specificity")
        metrics["Value"].append(f"{specificity:.4f}")

    return pd.DataFrame(metrics)
