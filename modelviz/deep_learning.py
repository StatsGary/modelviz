import matplotlib.pyplot as plt
from typing import List, Tuple


def draw_mlp(
    layer_sizes: List[int],
    show_bias: bool = True,
    activation: str = 'σ',
    figsize: Tuple[int, int] = (11, 7),
    neuron_radius: float = 0.24,
    h_spacing: float = 2.7,
    v_spacing: float = 1.5,
    input_color: str = '#eef4fa',
    edge_color: str = '#336699',
    hidden_text_color: str = '#336699',
    conn_color: str = '#444',
    conn_alpha: float = 0.4,
    weight_color: str = '#ba2222',
    weight_fontsize: int = 10,
    weight_box_color: str = 'white',
    bias_color: str = '#fcf7cd',
    bias_edge_color: str = '#998a26',
    bias_line_color: str = '#998a26',
    bias_box_alpha: float = 0.7,
    activation_text_color: str = '#008488'
) -> None:
    """
    Visualize a multilayer perceptron (MLP) architecture.

    Parameters:
    - layer_sizes: A list of integers indicating the number of neurons per layer.
    - show_bias: Whether to show bias nodes and their connections.
    - activation: Activation function symbol to display between layers.
    - figsize: Size of the matplotlib figure.
    - neuron_radius: Radius of each neuron circle.
    - h_spacing: Horizontal spacing between layers.
    - v_spacing: Vertical spacing between neurons in a layer.
    - input_color: Fill color for neuron circles.
    - edge_color: Edge color for neuron circles.
    - hidden_text_color: Font color for hidden neurons.
    - conn_color: Color of the lines connecting neurons.
    - conn_alpha: Transparency of connection lines.
    - weight_color: Color of the weight text.
    - weight_fontsize: Font size for weight labels.
    - weight_box_color: Background color for weight label boxes.
    - bias_color: Color of bias nodes.
    - bias_edge_color: Edge color of bias nodes.
    - bias_line_color: Color of dashed lines from bias to neuron.
    - bias_box_alpha: Transparency of bias label box.
    - activation_text_color: Color of activation function label."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    n_layers = len(layer_sizes)

    # Calculate vertical positions for neurons in each layer
    y_offset = []
    for n in layer_sizes:
        total_h = v_spacing * (n - 1)
        y_offset.append([i * v_spacing - total_h / 2 for i in range(n)])

    # Draw neurons
    for i, layer in enumerate(layer_sizes):
        for j in range(layer):
            circle = plt.Circle((i * h_spacing, y_offset[i][j]), neuron_radius,
                                color=input_color, ec=edge_color, lw=2.2, zorder=3, alpha=0.97)
            ax.add_patch(circle)
            # Add labels
            if i == 0:
                ax.text(i * h_spacing - 0.5, y_offset[i][j], f"$x_{{{j+1}}}$", fontsize=16,
                        va='center', ha='right')
            elif i == n_layers - 1:
                ax.text(i * h_spacing + 0.5, y_offset[i][j], f"$y_{{{j+1}}}$", fontsize=16,
                        va='center', ha='left', fontweight='bold')
            else:
                ax.text(i * h_spacing, y_offset[i][j], f"$h_{{{i},{j+1}}}$", fontsize=16,
                        ha='center', va='center', color=hidden_text_color, fontweight='bold')

    # Draw connections and weights
    for i in range(n_layers - 1):
        for j, y1 in enumerate(y_offset[i]):
            for k, y2 in enumerate(y_offset[i + 1]):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing], [y1, y2],
                                  color=conn_color, lw=1, alpha=conn_alpha, zorder=1)
                ax.add_line(line)
                # Add weight label
                x_mid = (i * h_spacing + (i + 1) * h_spacing) / 2
                y_mid = (y1 + y2) / 2
                ax.text(x_mid, y_mid + 0.18,
                        f"$w^{{({i+1})}}_{{{k+1},{j+1}}}$",
                        fontsize=weight_fontsize, color=weight_color, alpha=0.95,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.12", fc=weight_box_color,
                                  ec='none', alpha=bias_box_alpha))

        # Activation function label
        if i < n_layers - 2:
            ax.text((i + 0.5) * h_spacing, max(y_offset[i + 1]) + 0.7,
                    f"Activation: ${activation}$", fontsize=13,
                    ha='center', color=activation_text_color, alpha=0.7)

        # Bias nodes
        if show_bias:
            bias_y = max(y_offset[i + 1]) + 0.75
            ax.scatter((i + 1) * h_spacing, bias_y, s=200, marker='s',
                       color=bias_color, edgecolors=bias_edge_color, zorder=4)
            ax.text((i + 1) * h_spacing + 0.3, bias_y,
                    f"$b^{{({i+1})}}$", fontsize=13, color=bias_edge_color, va='center')
            for y2 in y_offset[i + 1]:
                ax.plot([(i + 1) * h_spacing, (i + 1) * h_spacing],
                        [bias_y, y2 - 0.1],
                        color=bias_line_color, lw=1.2, ls='dashed', alpha=0.7, zorder=1)

    # Layer labels
    ax.text(-0.1, max(y_offset[0]) + 1.1, "Input\nLayer", ha='center',
            fontsize=15, fontweight='bold', color='#222')
    for i in range(1, n_layers - 1):
        ax.text(i * h_spacing, max(y_offset[i]) + 1.1,
                f"Hidden\nLayer {i}", ha='center', fontsize=15,
                color='#0084e6', fontweight='bold')
    ax.text((n_layers - 1) * h_spacing, max(y_offset[-1]) + 1.1,
            "Output\nLayer", ha='center', fontsize=15,
            color='#222', fontweight='bold')

    # Set axis limits
    ax.set_xlim(-1.5, n_layers * h_spacing)
    ax.set_ylim(-max(layer_sizes) * v_spacing / 1.5 - 1,
                max(layer_sizes) * v_spacing / 1.5 + 2)

    plt.tight_layout()
    plt.show()