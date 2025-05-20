import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def regression_diagnostics_panel(y_test,
                                  y_pred,
                                  figsize=(18, 5),
                                  font_size=12,
                                  hist_bins=30,
                                  hist_alpha=0.7,
                                  hist_color='grey',
                                  hist_edgecolor='black',
                                  vline_color='black',
                                  vline_style='--',
                                  vline_width=1,
                                  scatter_alpha=0.5,
                                  scatter_color='grey',
                                  line_color='black',
                                  line_style='--',
                                  line_width=1,
                                  qq_line_color='red',
                                  qq_point_color='blue',
                                  qq_point_size=20,
                                  qq_line_style='-',
                                  show_grid=True):
    """
    Creates a 3-panel diagnostic plot:
    - Histogram of residuals
    - Actual vs. Predicted
    - Q-Q plot of residuals

    Parameters:
    - y_test, y_pred: true and predicted values
    - figsize: tuple of figure size
    - font_size: int for axis and title labels
    - hist_bins: number of bins in histogram
    - hist_alpha: alpha for histogram bars
    - hist_color: fill color for histogram
    - hist_edgecolor: edge color for histogram bars
    - vline_color, vline_style, vline_width: vertical line over histogram at 0
    - scatter_alpha, scatter_color: actual vs. predicted plot
    - line_color, line_style, line_width: y=x reference line in actual vs. predicted
    - qq_line_color, qq_point_color, qq_point_size, qq_line_style: Q-Q plot styling
    - show_grid: bool for showing grid on all subplots

    Returns:
    - None: displays the plots  
    Example:
    >>> from sklearn.model_selection import train_test_split    
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> regression_diagnostics_panel(y_test, y_pred)
    >>> # This will display the diagnostic plots for the regression model.
    >>> # Note: Ensure that you have matplotlib and scipy installed in your environment.
    >>> # You can customize the appearance of the plots using the parameters.
    """
   

    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Distribution of Residuals
    axes[0].hist(residuals, bins=hist_bins, alpha=hist_alpha,
                 color=hist_color, edgecolor=hist_edgecolor)
    axes[0].axvline(x=0, color=vline_color, linestyle=vline_style, lw=vline_width)
    axes[0].set_title('Distribution of Residuals', fontsize=font_size)
    axes[0].set_xlabel('Residuals', fontsize=font_size)
    axes[0].set_ylabel('Frequency', fontsize=font_size)
    if show_grid:
        axes[0].grid(True)

    # 2. Actual vs. Predicted
    axes[1].scatter(y_test, y_pred, alpha=scatter_alpha, color=scatter_color)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 color=line_color, linestyle=line_style, lw=line_width)
    axes[1].set_title('Actual vs. Predicted', fontsize=font_size)
    axes[1].set_xlabel('Actual', fontsize=font_size)
    axes[1].set_ylabel('Predicted', fontsize=font_size)
    if show_grid:
        axes[1].grid(True)

    # 3. Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[2].plot(osm, slope * np.array(osm) + intercept,
                 qq_line_style, color=qq_line_color, label='Q-Q Line')
    axes[2].scatter(osm, osr, color=qq_point_color, s=qq_point_size, alpha=0.6, label='Residuals')
    axes[2].set_title('Q-Q Plot of Residuals', fontsize=font_size)
    axes[2].set_xlabel('Theoretical Quantiles', fontsize=font_size)
    axes[2].set_ylabel('Ordered Values', fontsize=font_size)
    axes[2].legend()
    if show_grid:
        axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()