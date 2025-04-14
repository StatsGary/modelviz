import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_forecast_subplots(y_train, y_test, 
                           combined_df, 
                           title="Forecasts by Model",
                           actual_label='Actual',
                           actual_label_color='black',
                           actual_line_width=1.8,
                           axis_cutoff_col='grey',
                           title_font_size=16,
                           x_label_title='Date'):
    """
    Plots a series of subplots comparing actual values to model forecast values over time.

    Parameters:
        y_train (pd.Series): Historical actual values (before forecast).
        y_test (pd.Series): Actual values for the forecast period.
        combined_df (pd.DataFrame): DataFrame where each column (except 'actual') 
                                    contains forecasted values from different models. 
                                    The index should be datetime-like.
        title (str): Overall plot title.
        actual_label (str): Label for the actual series line.
        actual_label_color (str): Color for the actual series line.
        actual_line_width (float): Line width for the actual series line.
        axis_cutoff_col (str): Color for the vertical line marking start of forecast.
        title_font_size (int): Font size for subplot and overall titles.
        x_label_title (str): X-axis label.

    Returns:
        None. Displays a matplotlib figure.
    """
    
    full_actual = pd.concat([y_train, y_test])
    full_actual.index = [i.to_timestamp() if hasattr(i, "to_timestamp") else pd.to_datetime(i)
                         for i in full_actual.index]
    full_actual = full_actual.astype(float)

    combined_df_plot = combined_df.copy()
    combined_df_plot.index = [i.to_timestamp() if hasattr(i, "to_timestamp") else pd.to_datetime(i)
                              for i in combined_df_plot.index]
    for col in combined_df_plot.columns:
        combined_df_plot[col] = combined_df_plot[col].astype(float)

    models = [col for col in combined_df_plot.columns if col != 'actual']
    n_models = len(models)

    cols = 2
    rows = math.ceil(n_models / cols)

    _, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), sharex=True)
    axes = axes.flatten()  

    for ax, model in zip(axes, models):
        ax.plot(full_actual.index, full_actual.values, label=actual_label, 
                color=actual_label_color, linewidth=actual_line_width)
        ax.plot(combined_df_plot.index, combined_df_plot[model].values, 
                label=f'Forecast ({model})', linestyle='--')
        ax.axvline(combined_df_plot.index[0], color=axis_cutoff_col, 
                   linestyle=':', label='Forecast Start')
        ax.set_title(f"{model}", fontsize=title_font_size)
        ax.legend()
        ax.grid(True)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=title_font_size)
    plt.xlabel(x_label_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()