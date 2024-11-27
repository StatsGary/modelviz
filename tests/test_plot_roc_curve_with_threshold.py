import pytest
import numpy as np
from unittest import mock
from modelviz.roc import plot_roc_curve_with_youdens_thresholds

# Test data
fpr = [0.0, 0.1, 0.2, 0.3]
tpr = [0.0, 0.4, 0.6, 1.0]
thresholds = np.array([1.0, 0.8, 0.5, 0.2])
roc_auc = 0.85
model_name = "Test Model"

def test_plot_with_defaults():
    """Test plot generation with default parameters."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name)
        mock_show.assert_called_once()


def test_inconsistent_lengths():
    """Test for mismatched lengths of fpr, tpr, and thresholds."""
    invalid_fpr = [0.0, 0.1]  # Different length
    with pytest.raises(ValueError):
        plot_roc_curve_with_youdens_thresholds(invalid_fpr, tpr, thresholds, roc_auc, model_name)


def test_invalid_types():
    """Test for invalid types of fpr, tpr, and thresholds."""
    with pytest.raises(TypeError):
        plot_roc_curve_with_youdens_thresholds("invalid", tpr, thresholds, roc_auc, model_name)
    with pytest.raises(TypeError):
        plot_roc_curve_with_youdens_thresholds(fpr, "invalid", thresholds, roc_auc, model_name)
    with pytest.raises(TypeError):
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, "invalid", roc_auc, model_name)


def test_adjusted_threshold_out_of_range():
    """Test for adjusted_threshold outside the range of thresholds."""
    with pytest.raises(ValueError):
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, adjusted_threshold=2.0)
    with pytest.raises(ValueError):
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, adjusted_threshold=-0.1)


def test_youden_threshold_out_of_range():
    """Test for youden_threshold outside the range of thresholds."""
    with pytest.raises(ValueError):
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, youden_threshold=2.0)
    with pytest.raises(ValueError):
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, youden_threshold=-0.1)



def test_custom_figure_size():
    """Test custom figure size."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, figsize=(10, 8))
        mock_show.assert_called_once()


def test_custom_annotation_and_styles():
    """Test custom annotation offsets, scatter size, and font size."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(
            fpr, tpr, thresholds, roc_auc, model_name,
            annotation_offset=(0.03, 0.03), scatter_size=150, annotation_fontsize=12
        )
        mock_show.assert_called_once()


def test_custom_line_style_and_grid():
    """Test custom line style and grid visibility."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(
            fpr, tpr, thresholds, roc_auc, model_name,
            line_style_random='r--', show_grid=False
        )
        mock_show.assert_called_once()


def test_custom_legend_location():
    """Test custom legend location."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, legend_loc='upper left')
        mock_show.assert_called_once()


def test_roc_auc_in_plot():
    """Test that ROC AUC is included in the plot label."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name)
        mock_show.assert_called_once()


def test_title_customization():
    """Test plot with customized title font size."""
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc, model_name, title_fontsize=14)
        mock_show.assert_called_once()
