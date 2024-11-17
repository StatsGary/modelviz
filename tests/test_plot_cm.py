import pytest
import pandas as pd
from unittest import mock
from modelviz.relationships import plot_correlation_matrix

def test_plot_with_defaults():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [4, 3, 2, 1],
        'C': [5, 6, 7, 8]
    })
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_correlation_matrix(df)
        mock_show.assert_called_once()

def test_invalid_df_type():
    with pytest.raises(TypeError):
        plot_correlation_matrix([1, 2, 3])

def test_invalid_method():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    with pytest.raises(ValueError):
        plot_correlation_matrix(df, method='invalid_method')

def test_not_enough_numeric_columns():
    df = pd.DataFrame({
        'A': ['x', 'y', 'z'],
        'B': ['a', 'b', 'c']
    })
    with pytest.raises(ValueError):
        plot_correlation_matrix(df)


def test_custom_colormap():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [2, 3, 4],
        'C': [3, 4, 5]
    })
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['red', 'green', 'blue'])
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_correlation_matrix(df, cmap=custom_cmap)
        mock_show.assert_called_once()

def test_invalid_cmap():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [2, 3, 4]
    })
    with pytest.raises(ValueError):
        plot_correlation_matrix(df, cmap='not_a_cmap')
    with pytest.raises(TypeError):
        plot_correlation_matrix(df, cmap=123)

def test_invalid_figsize():
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })
    with pytest.raises(TypeError):
        plot_correlation_matrix(df, figsize='invalid')
    with pytest.raises(ValueError):
        plot_correlation_matrix(df, figsize=(10,))
    with pytest.raises(TypeError):
        plot_correlation_matrix(df, figsize=(10, '8'))
