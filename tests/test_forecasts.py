import pytest
import pandas as pd
from matplotlib import pyplot as plt
from unittest import mock
from modelviz.forecast import plot_forecast_subplots
@pytest.fixture
def mock_data():
    dates_train = pd.date_range(start='2023-01-01', periods=5, freq='D')
    dates_test = pd.date_range(start='2023-01-06', periods=5, freq='D')

    y_train = pd.Series([1, 2, 3, 4, 5], index=dates_train)
    y_test = pd.Series([6, 7, 8, 9, 10], index=dates_test)

    combined_index = dates_test
    combined_df = pd.DataFrame({
        'Model_A': [5.5, 6.5, 7.5, 8.5, 9.5],
        'Model_B': [6.2, 7.1, 8.3, 9.0, 10.1],
    }, index=combined_index)

    return y_train, y_test, combined_df

def test_plot_runs_without_error(mock_data):
    y_train, y_test, combined_df = mock_data
    # Just check that the plot function runs without raising an exception
    try:
        plot_forecast_subplots(y_train, y_test, combined_df)
    except Exception as e:
        pytest.fail(f"plot_forecast_subplots raised an exception: {e}")

def test_plot_labels(mock_data):
    y_train, y_test, combined_df = mock_data
    with mock.patch("matplotlib.pyplot.suptitle") as mock_suptitle:
        plot_forecast_subplots(y_train, y_test, combined_df, title="Test Plot Title", actual_label="Real Data")
        mock_suptitle.assert_called_once_with("Test Plot Title", fontsize=16)