import pytest
import pandas as pd
from unittest.mock import patch
from modelviz.histogram import plot_feature_histograms  

@pytest.fixture
def sample_df():
    """Fixture for creating a sample DataFrame."""
    return pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [2.5, 3.6, 4.7, 5.8, 6.9],
        'bin_col': [0, 1, 0, 1, 0],
        'string_col': ['a', 'b', 'c', 'd', 'e']
    })

@patch("matplotlib.pyplot.show")
def test_matplotlib_histograms(mock_show, sample_df):
    """Test matplotlib histogram generation."""
    plot_feature_histograms(sample_df, library='matplotlib')
    assert mock_show.called, "The matplotlib figure was not displayed."

@patch("plotly.graph_objects.Figure.show")
def test_plotly_histograms(mock_show, sample_df):
    """Test plotly histogram generation."""
    plot_feature_histograms(sample_df, library='plotly')
    assert mock_show.called, "The plotly figures were not displayed."

def test_exclude_bin_columns(sample_df):
    """Test excluding binary columns."""
    numeric_cols = [col for col in sample_df.select_dtypes(include=['float64', 'int64']).columns
                    if not sample_df[col].dropna().isin([0, 1]).all()]
    assert numeric_cols == ['int_col', 'float_col'], "Binary columns were not correctly excluded."

def test_invalid_library(sample_df):
    """Test invalid library option."""
    with pytest.raises(ValueError, match="Invalid library specified."):
        plot_feature_histograms(sample_df, library='invalid_lib')
