import matplotlib.pyplot as plt

def plot_kfold_cv(N=10000, K=5,
                  val_color='#432f8e',
                  train_color='#86277b',
                  save_path=None
                  ):
    """
    Visualizes K-Fold Cross Validation splits using a bar plot.

    Parameters
    ----------
    N : int, optional
        Total number of samples. Default is 10000.
    K : int, optional
        Number of folds. Default is 5.
    val_color : str, optional
        Color for the validation set bars. Default is '#432f8e'.
    train_color : str, optional
        Color for the training set bars. Default is '#86277b'.
    save_path : str or None, optional
        File path to save the plot. If None, the plot is displayed. Default is None.

    Raises
    ------
    TypeError
        If N or K are not integers.
        If val_color or train_color are not strings.
        If save_path is not a string or None.
    ValueError
        If N or K are not positive integers.
        If K is greater than N.
        If val_color or train_color are not valid matplotlib colors.

    """
    # Error handling
    if not isinstance(N, int) or not isinstance(K, int):
        raise TypeError("N and K must be integers.")
    if N <= 0 or K <= 0:
        raise ValueError("N and K must be positive integers.")
    if K > N:
        raise ValueError("K cannot be greater than N.")
    if not isinstance(val_color, str):
        raise TypeError("val_color must be a string.")
    if not isinstance(train_color, str):
        raise TypeError("train_color must be a string.")
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string or None.")

    # Check if val_color and train_color are valid colors
    try:
        plt.figure()
        plt.plot([0], [0], color=val_color)
        plt.close()
    except ValueError:
        raise ValueError("val_color is not a valid matplotlib color.")

    try:
        plt.figure()
        plt.plot([0], [0], color=train_color)
        plt.close()
    except ValueError:
        raise ValueError("train_color is not a valid matplotlib color.")

    # Calculate fold sizes
    fold_sizes = [N // K] * K
    for i in range(N % K):
        fold_sizes[i] += 1

    indices = list(range(N))
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        fold_indices.append(indices[current:current + fold_size])
        current += fold_size

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, K))

    for i in range(K):
        y = K - i - 1
        current = 0
        for j, fold_size in enumerate(fold_sizes):
            color = val_color if j == i else train_color
            rect = plt.Rectangle((current, y), fold_size, 0.8, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            current += fold_size
        ax.text(-5, y + 0.4, f'Fold {i + 1}', va='center', ha='right')

    ax.set_xlim(0, N)
    ax.set_ylim(-0.5, K + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Sample Index')
    ax.set_title(f'K-Fold Cross Validation Visualization (K={K})')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
