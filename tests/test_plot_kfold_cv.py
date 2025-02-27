import pytest
import os
from unittest import mock
from modelviz.kfold import plot_kfold_cv

def test_default_parameters():
    # Should run without exceptions
    plot_kfold_cv()

def test_specified_N_K():
    plot_kfold_cv(N=100, K=10)

def test_non_integer_N_K():
    with pytest.raises(TypeError):
        plot_kfold_cv(N=10.5, K=5)
    with pytest.raises(TypeError):
        plot_kfold_cv(N=10, K='5')

def test_negative_zero_N_K():
    with pytest.raises(ValueError):
        plot_kfold_cv(N=0, K=5)
    with pytest.raises(ValueError):
        plot_kfold_cv(N=-10, K=5)
    with pytest.raises(ValueError):
        plot_kfold_cv(N=10, K=0)
    with pytest.raises(ValueError):
        plot_kfold_cv(N=10, K=-5)

def test_K_greater_than_N():
    with pytest.raises(ValueError):
        plot_kfold_cv(N=5, K=10)

def test_invalid_color_type():
    with pytest.raises(TypeError):
        plot_kfold_cv(val_color=123)
    with pytest.raises(TypeError):
        plot_kfold_cv(train_color=None)

