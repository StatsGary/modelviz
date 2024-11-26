import pytest
import pandas as pd
from unittest import mock
from modelviz.model_eval import multi_model_visualizer

def test_default_usage():
    results = pd.DataFrame({
        'Model': ['A', 'B', 'C'],
        'Accuracy': [0.9, 0.85, 0.88],
        'Precision': [0.8, 0.75, 0.78],
        'Recall': [0.7, 0.65, 0.68]
    })
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        multi_model_visualizer(results)
        mock_show.assert_called_once()


def test_missing_model_column():
    results = pd.DataFrame({
        'Accuracy': [0.9, 0.85],
        'Precision': [0.8, 0.75]
    })
    with pytest.raises(ValueError):
        multi_model_visualizer(results)

def test_invalid_metrics():
    results = pd.DataFrame({
        'Model': ['A', 'B'],
        'Accuracy': [0.9, 0.85]
    })
    with pytest.raises(ValueError):
        multi_model_visualizer(results, metrics=['NonExistentMetric'])

def test_invalid_colors():
    results = pd.DataFrame({
        'Model': ['A', 'B'],
        'Accuracy': [0.9, 0.85]
    })
    with pytest.raises(TypeError):
        multi_model_visualizer(results, colors='red')

def test_not_enough_colors():
    results = pd.DataFrame({
        'Model': ['A', 'B'],
        'Metric1': [0.5, 0.6],
        'Metric2': [0.7, 0.8],
        'Metric3': [0.9, 1.0],
        'Metric4': [0.4, 0.3]
    })
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        multi_model_visualizer(results, colors=['red', 'green'])
        mock_show.assert_called_once()
