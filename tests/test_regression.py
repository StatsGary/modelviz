import pytest
import numpy as np
from modelviz.regression import regression_diagnostics_panel

# Generate fake regression data
@pytest.fixture
def sample_data():
    np.random.seed(42)
    y_true = np.random.normal(loc=100, scale=10, size=100)
    noise = np.random.normal(loc=0, scale=5, size=100)
    y_pred = y_true + noise
    return y_true, y_pred

def test_regression_diagnostics_runs(sample_data):
    y_true, y_pred = sample_data
    # Should run without error
    regression_diagnostics_panel(y_true, y_pred)

def test_custom_plot_args(sample_data):
    y_true, y_pred = sample_data
    # Try changing style params
    regression_diagnostics_panel(
        y_test=y_true,
        y_pred=y_pred,
        font_size=10,
        figsize=(12, 4),
        hist_bins=20,
        hist_color='skyblue',
        scatter_color='black',
        qq_point_color='green',
        show_grid=False
    )

def test_handles_different_input_shapes():
    # Column vector inputs
    y_true = np.array([[10], [20], [30]])
    y_pred = np.array([[12], [19], [29]])
    regression_diagnostics_panel(y_true.flatten(), y_pred.flatten())

def test_fails_with_mismatched_shapes():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 19])  # wrong shape
    with pytest.raises(ValueError):
        regression_diagnostics_panel(y_true, y_pred)
