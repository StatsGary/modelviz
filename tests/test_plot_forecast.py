import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.linear_model import LinearRegression
import importlib.util
from modelviz.forecast import plot_forecast


@pytest.fixture
def sample_series():
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    return pd.Series(np.random.randn(20), index=dates)


def test_statsmodels_forecast(sample_series):
    mock_model = MagicMock()
    forecast_mean = pd.Series([10, 11, 12])
    conf_int = pd.DataFrame({
        0: [9, 10, 11],
        1: [11, 12, 13]
    })

    forecast_result = MagicMock()
    forecast_result.predicted_mean = forecast_mean
    forecast_result.conf_int.return_value = conf_int
    mock_model.get_forecast.return_value = forecast_result

    result = plot_forecast(sample_series, mock_model, model_type='statsmodels', horizon=3, plot=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['forecast', 'lower_ci', 'upper_ci']
    assert len(result) == 3

has_prophet = importlib.util.find_spec("prophet") is not None
@pytest.mark.skipif(not has_prophet, reason="Prophet is not installed")
def test_prophet_forecast(sample_series):
    mock_model = MagicMock()
    mock_model.make_future_dataframe.return_value = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=23)})
    mock_model.predict.return_value = pd.DataFrame({
        'ds': pd.date_range('2023-01-21', periods=3),
        'yhat': [1.0, 2.0, 3.0],
        'yhat_lower': [0.5, 1.5, 2.5],
        'yhat_upper': [1.5, 2.5, 3.5],
    })

    result = plot_forecast(sample_series, mock_model, model_type='prophet', horizon=3, plot=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['forecast', 'lower_ci', 'upper_ci']
    assert len(result) == 3


def test_ml_forecast(sample_series):
    X_train = np.arange(0, 20).reshape(-1, 1)
    y_train = np.arange(0, 20)
    X_future = np.array([[20], [21], [22]])

    model = LinearRegression().fit(X_train, y_train)

    result = plot_forecast(
        sample_series, model, model_type='ml', horizon=3,
        X_train=X_train, y_train=y_train, X_future=X_future, plot=False
    )

    assert isinstance(result, pd.DataFrame)
    assert 'forecast' in result.columns
    assert len(result) == 3


def test_missing_data_for_ml_raises_error(sample_series):
    with pytest.raises(ValueError, match="X_train, y_train, and X_future are required"):
        plot_forecast(sample_series, model=None, model_type='ml', plot=False)


def test_invalid_model_type_raises_error(sample_series):
    with pytest.raises(ValueError, match="`model_type` must be 'statsmodels', 'prophet', or 'sk-learn-ml'"):
        plot_forecast(sample_series, model=None, model_type='bad_model', plot=False)


def test_infer_freq_failure_raises_error():
    index = pd.Index([0, 1, 2])  # Not a datetime index
    data = pd.Series([1, 2, 3], index=index)
    mock_model = MagicMock()
    forecast_result = MagicMock()
    forecast_result.predicted_mean = pd.Series([1, 2, 3])
    forecast_result.conf_int.return_value = pd.DataFrame({0: [0.8, 1.8, 2.8], 1: [1.2, 2.2, 3.2]})
    mock_model.get_forecast.return_value = forecast_result

    with pytest.raises(TypeError, match="cannot infer freq from a non-convertible index"):
        plot_forecast(data, mock_model, model_type='statsmodels', plot=False)


def test_save_fig_creates_file(tmp_path, sample_series):
    mock_model = MagicMock()
    forecast_mean = pd.Series([10, 11, 12])
    conf_int = pd.DataFrame({0: [9, 10, 11], 1: [11, 12, 13]})
    forecast_result = MagicMock()
    forecast_result.predicted_mean = forecast_mean
    forecast_result.conf_int.return_value = conf_int
    mock_model.get_forecast.return_value = forecast_result

    save_path = tmp_path / "forecast_plot.png"
    plot_forecast(sample_series, mock_model, model_type='statsmodels', horizon=3, save_fig=str(save_path))

    assert save_path.exists()
