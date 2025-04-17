import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from scipy.stats import norm

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



def plot_forecast(data, model, horizon=3, model_type='statsmodels', alpha=0.05, 
                  plot=True,X_train=None, y_train=None, X_future=None, save_fig=None, **plot_kwargs):
    """
    Generates forecasts, calculates confidence intervals, and optionally plots results.

    Parameters:
        data (pd.Series or pd.DataFrame): The original full time series data used for modeling.
        model: A fitted model object (statsmodels, Prophet, or scikit-learn ML model).
        horizon (int): Number of steps ahead to forecast.
        model_type (str): Type of the model ('statsmodels', 'prophet', or 'ml').
        alpha (float): Significance level for confidence intervals (default 0.05).
        plot (bool): Whether to plot the forecast and confidence intervals.
        X_train (pd.DataFrame or array-like): Required for 'ml' model to estimate residuals.
        y_train (pd.Series or array-like): Required for 'ml' model to estimate residuals.
        X_future (pd.DataFrame or array-like): Required for 'ml' model to predict the future.
        save_fig (bool): If boolean passed saves the plot to a path you provide
         **plot_kwargs: Optional keyword arguments to customize the plot when plot=True.
            Supported keys:
                - figsize (tuple): Figure size (default: (12, 6))
                - historical_label (str): Label for historical data line (default: 'Historical Data')
                - historical_color (str): Color for historical data line (default: 'blue')
                - forecast_label (str): Label for forecast line (default: 'Forecast')
                - ci_label (str): Label for confidence interval shading (default: 'Confidence Interval')
                - forecast_color (str): Color for forecast line (default: 'orange')
                - ci_color (str): Color for confidence interval shading (default: 'orange')
                - ci_alpha (float): Opacity of confidence interval (default: 0.3)
                - xlabel (str): X-axis label (default: 'Time')
                - ylabel (str): Y-axis label (default: 'Value')
                - title (str): Plot title (default: 'Forecast with Confidence Intervals')
                - grid (bool): Whether to show grid (default: True)

    Returns:
        pd.DataFrame: A DataFrame with forecasts and confidence intervals.
    """
    
    if model_type == 'statsmodels':
        forecast_result = model.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        ci = forecast_result.conf_int(alpha=alpha)
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower_ci': ci.iloc[:, 0],
            'upper_ci': ci.iloc[:, 1]
        })
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq is None:
            raise ValueError("Could not infer frequency from data index. Please set the frequency explicitly.")
        forecast_df.index = pd.date_range(start=data.index[-1], periods=horizon + 1, freq=inferred_freq)[1:]

    elif model_type == 'prophet':
        future = model.make_future_dataframe(periods=horizon, freq=pd.infer_freq(data.index))
        forecast_result = model.predict(future)
        forecast_df = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
        forecast_df = forecast_df.rename(columns={
            'ds': 'date',
            'yhat': 'forecast',
            'yhat_lower': 'lower_ci',
            'yhat_upper': 'upper_ci'
        }).set_index('date')

    elif model_type == 'ml':
        if X_future is None or X_train is None or y_train is None:
            raise ValueError("X_train, y_train, and X_future are required for 'ml' model type.")
        
        forecast = model.predict(X_future)
        train_pred = model.predict(X_train)
        residuals = y_train - train_pred
        std = np.std(residuals)
        z = norm.ppf(1 - alpha / 2)
        lower = forecast - z * std
        upper = forecast + z * std
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq is None:
            raise ValueError("Could not infer frequency from data index. Please set the frequency explicitly.")
        
        forecast_idx = pd.date_range(start=data.index[-1], periods=horizon + 1, freq=inferred_freq)[1:]
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower_ci': lower,
            'upper_ci': upper
        }, index=forecast_idx)

    else:
        raise ValueError("`model_type` must be 'statsmodels', 'prophet', or 'sk-learn-ml'.")

    if plot:
        plt.figure(figsize=plot_kwargs.get('figsize', (12, 6)))
        plt.plot(data, label=plot_kwargs.get('historical_label', 'Historical Data'),
                 color=plot_kwargs.get('historical_color', 'blue'))
        plt.plot(forecast_df['forecast'], 
                 label=plot_kwargs.get('forecast_label', 'Forecast'),
                 color=plot_kwargs.get('forecast_color', 'orange'))
        plt.fill_between(forecast_df.index, 
                     forecast_df['lower_ci'], forecast_df['upper_ci'],
                     color=plot_kwargs.get('ci_color', 'orange'), 
                     alpha=plot_kwargs.get('ci_alpha', 0.3),
                     label=plot_kwargs.get('ci_label', 'Confidence Interval'))
        plt.legend()
        plt.xlabel(plot_kwargs.get('xlabel', 'Time'))
        plt.ylabel(plot_kwargs.get('ylabel', 'Value'))
        plt.title(plot_kwargs.get('title', 'Forecast with Confidence Intervals'))
        plt.grid(plot_kwargs.get('grid', True))
        if save_fig is not None:
            plt.savefig(save_fig)
        plt.show()
        
    return forecast_df
