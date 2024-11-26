import pytest
import pandas as pd
from unittest import mock
import seaborn as sns
from modelviz.missvals import plot_missing_values_heatmap

def test_plot_with_defaults():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df)
        #Mock show assert called once
        mock_show.assert_called_once()

def test_custom_title_and_xlabel():
    df = pd.DataFrame({'A': [1, 2, None]})
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df, plt_title='Custom Title', plt_x_label='Custom X Label')
        mock_show.assert_called_once()

def test_invalid_df_type():
    with pytest.raises(TypeError):
        plot_missing_values_heatmap([1, 2, 3])

def test_invalid_figsize():
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, figsize='invalid')
    with pytest.raises(ValueError):
        plot_missing_values_heatmap(df, figsize=(10,))
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, figsize=(10, '8'))

def test_invalid_cmap():
    df = pd.DataFrame({'A': [1, 2, None]})
    with pytest.raises(ValueError):
        plot_missing_values_heatmap(df, cmap='not_a_cmap')
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, cmap=123)

def test_invalid_cbar():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, cbar='True')

def test_invalid_tick_labels():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, y_tick_labels='yes')
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, x_tick_labels=123)

def test_invalid_plt_title():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, plt_title=123)

def test_invalid_plt_x_label():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, plt_x_label=None)


def test_custom_figsize():
    df = pd.DataFrame({'A': [1, 2, None]})
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df, figsize=(8, 6))
        mock_show.assert_called_once()

def test_valid_cmap_instance():
    df = pd.DataFrame({'A': [1, None, 3]})
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['white', 'black'])
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df, cmap=custom_cmap)
        mock_show.assert_called_once()

def test_y_tick_labels_list():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, None, 6]})
    y_labels = ['Row1', 'Row2', 'Row3']
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df, y_tick_labels=y_labels)
        mock_show.assert_called_once()

def test_x_tick_labels_list():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, None, 6]})
    x_labels = ['Feature A', 'Feature B']
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_missing_values_heatmap(df, x_tick_labels=x_labels)
        mock_show.assert_called_once()

def test_invalid_y_tick_labels():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, y_tick_labels={'label1', 'label2'})

def test_invalid_x_tick_labels():
    df = pd.DataFrame({'A': [1, None, 3]})
    with pytest.raises(TypeError):
        plot_missing_values_heatmap(df, x_tick_labels=(1, 2, 3))
