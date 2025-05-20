import pytest
from matplotlib import pyplot as plt
from modelviz.deep_learning import draw_mlp  

@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

def test_basic_network_runs():
    """Check a simple 3-layer MLP runs without error."""
    draw_mlp(layer_sizes=[3, 4, 2])

def test_single_layer_input_output():
    """Check a network with input and output only (no hidden layers)."""
    draw_mlp(layer_sizes=[5, 1])

def test_deep_network():
    """Check a deeper MLP with several hidden layers."""
    draw_mlp(layer_sizes=[3, 5, 4, 3, 2, 1])

def test_with_bias_disabled():
    """Ensure function runs with bias disabled."""
    draw_mlp(layer_sizes=[4, 4, 2], show_bias=False)

def test_custom_styling_runs():
    """Test that passing various styling options doesn't break the function."""
    draw_mlp(
        layer_sizes=[2, 3, 1],
        activation='ReLU',
        neuron_radius=0.3,
        input_color='#ffeecc',
        edge_color='#333333',
        conn_color='gray',
        weight_color='blue',
        weight_fontsize=8,
        weight_box_color='#eeeeee',
        bias_color='red',
        bias_edge_color='darkred',
        activation_text_color='purple'
    )
