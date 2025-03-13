"""Module providing testing scripts for timeseries.py"""
import pytest
import pandas as pd
import numpy as np
from modelviz.timeseries import analyze_and_preprocess_time_series

def generate_mock_data():
    """Generates mock time series data."""
    date_rng = pd.date_range(start='1/1/2020', end='12/1/2025', freq='ME')
    np.random.seed(42)
    data = pd.DataFrame({'date': date_rng, 'value': np.random.randn(len(date_rng)) + 100})
    data.set_index('date', inplace=True)
    return data

def test_valid_input():
    """Tests the function with valid data."""
    data = generate_mock_data()
    result = analyze_and_preprocess_time_series(data, 'value', period=12)
    assert isinstance(result, dict)
    assert "Best Model Type" in result
    assert "Is Stationary" in result

def test_non_datetime_index():
    """Tests if a RuntimeError is raised when index is not DatetimeIndex."""
    data = generate_mock_data().reset_index()
    with pytest.raises(RuntimeError, match="The DataFrame index must be a DatetimeIndex"):
        analyze_and_preprocess_time_series(data, 'value', period=12)

def test_missing_column():
    """Tests if a RuntimeError is raised when the column does not exist."""
    data = generate_mock_data()
    with pytest.raises(RuntimeError, match="Column 'wrong_column' not found in DataFrame"):
        analyze_and_preprocess_time_series(data, 'wrong_column', period=12)

def test_all_nan_column():
    """Tests if a RuntimeError is raised when the column contains only NaN values."""
    data = generate_mock_data()
    data['value'] = np.nan
    with pytest.raises(RuntimeError, match="The time series column contains only NaN values"):
        analyze_and_preprocess_time_series(data, 'value', period=12)

def test_short_series():
    """Tests if the function handles a short time series gracefully."""
    data = generate_mock_data().iloc[:5]
    with pytest.raises(RuntimeError):
        analyze_and_preprocess_time_series(data, 'value', period=12)