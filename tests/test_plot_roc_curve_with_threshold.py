import unittest
import numpy as np
from matplotlib import pyplot as plt
from modelviz.roc import plot_roc_curve_with_threshold 

# Sample data for testing
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
fpr = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
tpr = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
thresholds = np.array([1.8, 0.8, 0.4, 0.35, 0.1])
roc_auc = 0.75

def test_valid_input():
    """Test that the function works with valid input."""
    try:
        plot_roc_curve_with_threshold(
            fpr, tpr, thresholds, roc_auc,
            model_name='Test Model', custom_threshold=0.5
        )
        print("test_valid_input: Passed")
    except Exception as e:
        print(f"test_valid_input: Failed - {e}")

def test_invalid_fpr():
    """Test that the function raises ValueError with invalid `fpr`."""
    try:
        plot_roc_curve_with_threshold(
            fpr=[], tpr=tpr, thresholds=thresholds, roc_auc=roc_auc
        )
    except ValueError:
        print("test_invalid_fpr: Passed")
    except Exception as e:
        print(f"test_invalid_fpr: Failed - {e}")
    else:
        print("test_invalid_fpr: Failed - No exception raised")

def test_custom_threshold_out_of_range():
    """Test that the function raises ValueError when `custom_threshold` is out of range."""
    try:
        plot_roc_curve_with_threshold(
            fpr, tpr, thresholds, roc_auc,
            custom_threshold=2.0  # Outside the range of thresholds
        )
    except ValueError:
        print("test_custom_threshold_out_of_range: Passed")
    except Exception as e:
        print(f"test_custom_threshold_out_of_range: Failed - {e}")
    else:
        print("test_custom_threshold_out_of_range: Failed - No exception raised")

def test_invalid_roc_auc():
    """Test that the function raises ValueError when `roc_auc` is invalid."""
    try:
        plot_roc_curve_with_threshold(
            fpr, tpr, thresholds, roc_auc=1.5  # Invalid roc_auc
        )
    except ValueError:
        print("test_invalid_roc_auc: Passed")
    except Exception as e:
        print(f"test_invalid_roc_auc: Failed - {e}")
    else:
        print("test_invalid_roc_auc: Failed - No exception raised")

def test_invalid_types():
    """Test that the function raises TypeError with invalid types."""
    try:
        plot_roc_curve_with_threshold(
            fpr, tpr, thresholds, roc_auc,
            custom_threshold='0.5'  # Should be numeric
        )
    except TypeError:
        print("test_invalid_types: Passed")
    except Exception as e:
        print(f"test_invalid_types: Failed - {e}")
    else:
        print("test_invalid_types: Failed - No exception raised")

# if __name__ == '__main__':
#     print("Running tests...")
#     test_valid_input()
#     test_invalid_fpr()
#     test_custom_threshold_out_of_range()
#     test_invalid_roc_auc()
#     test_invalid_types()
