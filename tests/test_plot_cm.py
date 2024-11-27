import pytest
import numpy as np
from unittest.mock import patch
from matplotlib import pyplot as plt
from modelviz.confusion_matrix import plot_confusion_matrix, metrics_summary  

@pytest.fixture
def binary_confusion_matrix():
    cm = np.array([[50, 10], [5, 35]])
    classes = ["Negative", "Positive"]
    model_name = "Logistic Regression"
    return cm, classes, model_name

@pytest.fixture
def multi_class_confusion_matrix():
    cm = np.array([[30, 2, 3], [5, 25, 4], [3, 2, 40]])
    classes = ["Class A", "Class B", "Class C"]
    model_name = "Random Forest"
    return cm, classes, model_name

def test_metrics_summary_binary(binary_confusion_matrix):
    cm, classes, _ = binary_confusion_matrix
    metrics_table = metrics_summary(cm, classes)
    assert "Accuracy" in metrics_table["Metric"].values
    assert "Micro Precision" in metrics_table["Metric"].values
    assert "Specificity" in metrics_table["Metric"].values
    assert metrics_table.loc[metrics_table["Metric"] == "Accuracy", "Value"].iloc[0] == "0.8500"

def test_metrics_summary_multi_class(multi_class_confusion_matrix):
    cm, classes, _ = multi_class_confusion_matrix
    metrics_table = metrics_summary(cm, classes)
    assert "Accuracy" in metrics_table["Metric"].values
    assert "Micro Precision" in metrics_table["Metric"].values
    assert "Macro Recall" in metrics_table["Metric"].values

def test_plot_confusion_matrix_binary(binary_confusion_matrix):
    cm, classes, model_name = binary_confusion_matrix
    with patch.object(plt, "show") as mock_show:
        plot_confusion_matrix(cm, classes, model_name, proportions_color="blue", label_positions_color="red")
        mock_show.assert_called_once()

def test_plot_confusion_matrix_multi_class(multi_class_confusion_matrix):
    cm, classes, model_name = multi_class_confusion_matrix
    with patch.object(plt, "show") as mock_show:
        plot_confusion_matrix(cm, classes, model_name, proportions_color="green")
        mock_show.assert_called_once()

