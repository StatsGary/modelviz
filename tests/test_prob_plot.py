import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  
from modelviz.relationships import plot_model_probs_scatter

def test_single_pair_runs():
    probs1 = np.random.rand(50)
    probs2 = np.random.rand(50)
    ax = plot_model_probs_scatter([(probs1, probs2)], show=False)
    assert hasattr(ax, "plot")  # Axes object

def test_multiple_pairs_runs():
    probs1 = np.random.rand(30)
    probs2 = np.random.rand(30)
    probs3 = np.random.rand(30)
    probs4 = np.random.rand(30)
    ax = plot_model_probs_scatter([(probs1, probs2), (probs3, probs4)],
                                  labels=["A vs B", "C vs D"],
                                  show=False)
    assert hasattr(ax, "plot")

def test_labels_and_markers():
    probs1 = np.random.rand(20)
    probs2 = np.random.rand(20)
    ax = plot_model_probs_scatter([(probs1, probs2)],
                                  labels=["My Pair"],
                                  markers=["X"],
                                  show=False)
    legend = ax.get_legend()
    assert legend is not None

def test_thresholds_drawn():
    probs1 = np.random.rand(15)
    probs2 = np.random.rand(15)
    ax = plot_model_probs_scatter(
        [(probs1, probs2)],
        x_thresh=0.5,
        y_thresh=0.7,
        x_thresh_label="X-50%",
        y_thresh_label="Y-70%",
        show=False
    )
    # Should not raise, and legend should have thresh labels
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "X-50%" in texts and "Y-70%" in texts

def test_show_diagonal():
    probs1 = np.random.rand(10)
    probs2 = np.random.rand(10)
    ax = plot_model_probs_scatter([(probs1, probs2)],
                                  show_diagonal=True,
                                  show=False)
    # Diagonal line is drawn as a Line2D in ax.lines
    assert any(np.allclose(line.get_xydata(), [[0,0],[1,1]], atol=1e-2) or
               np.allclose(line.get_xydata(), [[1,1],[0,0]], atol=1e-2)
               for line in ax.lines)

def test_shape_mismatch_raises():
    probs1 = np.random.rand(10)
    probs2 = np.random.rand(11)  # Different length!
    with pytest.raises(ValueError):
        plot_model_probs_scatter([(probs1, probs2)], show=False)

def test_default_labels_assigned():
    probs1 = np.random.rand(8)
    probs2 = np.random.rand(8)
    ax = plot_model_probs_scatter([(probs1, probs2)], show=False)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Pair 1" in texts

def test_returns_ax_object():
    probs1 = np.random.rand(6)
    probs2 = np.random.rand(6)
    ax = plot_model_probs_scatter([(probs1, probs2)], show=False)
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)

