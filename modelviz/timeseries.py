import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox

def analyze_and_preprocess_time_series(data: pd.DataFrame, 
                        column: str, 
                        period: int = 12,
                        add_color:str = "green",
                        mult_color: str = "blue",
                        generate_plot: bool = True,
                        **decomp_args):
    """
    Fully preprocesses a time series by:
    1. Determining if it follows an additive or multiplicative model.
    2. Performing stationarity checks (ADF test).
    3. Applying the correct transformation (Box-Cox or log if necessary).
    
    Parameters:
    - data (pd.DataFrame): DataFrame with a DateTime index and numeric column.
    - column (str): Column name of the time series.
    - period (int): Seasonal period (default=12 for monthly data).

    Returns:
    - dict: Contains preprocessing decisions, key metrics, and transformed series.
    """
    try:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")       
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("The DataFrame index must be a DatetimeIndex")
        if data[column].isnull().all():
            raise ValueError("The time series column contains only NaN values")
        
        data = data.dropna(subset=[column])
        
        decomposition_add = seasonal_decompose(data[column], model="additive", period=period, **decomp_args)
        decomposition_mul = seasonal_decompose(data[column], model="multiplicative", period=period, **decomp_args)

        var_add = np.var(decomposition_add.resid.dropna())
        var_mul = np.var(decomposition_mul.resid.dropna())

        model_type = "additive" if var_add < var_mul else "multiplicative"

        adf_result = adfuller(data[column].dropna())
        is_stationary = adf_result[1] < 0.05  

        transformed_series = data[column].copy()
        transformation_applied = "None"

        if model_type == "multiplicative":
            try:
                transformed_series, lambda_value = boxcox(data[column].dropna() + 1)  
                transformation_applied = f"Box-Cox (λ={lambda_value:.3f})"
            except ValueError:
                transformed_series = np.log1p(data[column])  
                transformation_applied = "Log Transformation"


        if generate_plot:
            _, axes = plt.subplots(4, 2, figsize=(12, 10))
            axes[0, 0].plot(decomposition_add.observed, label="Observed", color=add_color)
            axes[0, 0].set_title("Additive: Observed")
            axes[1, 0].plot(decomposition_add.trend, label="Trend", color=add_color)
            axes[1, 0].set_title("Additive: Trend")
            axes[2, 0].plot(decomposition_add.seasonal, label="Seasonal", color=add_color)
            axes[2, 0].set_title("Additive: Seasonal")
            axes[3, 0].plot(decomposition_add.resid, label="Residuals", color=add_color)
            axes[3, 0].set_title("Additive: Residuals")
            axes[0, 1].plot(decomposition_mul.observed, label="Observed", color=mult_color)
            axes[0, 1].set_title("Multiplicative: Observed")
            axes[1, 1].plot(decomposition_mul.trend, label="Trend", color=mult_color)
            axes[1, 1].set_title("Multiplicative: Trend")
            axes[2, 1].plot(decomposition_mul.seasonal, label="Seasonal", color=mult_color)
            axes[2, 1].set_title("Multiplicative: Seasonal")
            axes[3, 1].plot(decomposition_mul.resid, label="Residuals", color=mult_color)
            axes[3, 1].set_title("Multiplicative: Residuals")

            for ax in axes.flat:
                ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        # Return key metrics and transformed series
        results = {
            "Best Model Type": model_type,
            "Additive Residual Variance": var_add,
            "Multiplicative Residual Variance": var_mul,
            "Trend Strength": np.var(decomposition_add.trend.dropna()) / np.var(data[column]),
            "Seasonality Strength": np.var(decomposition_add.seasonal.dropna()) / np.var(data[column]),
            "Is Stationary": is_stationary,
            "ADF Test p-value": adf_result[1],
            "Transformation Applied": transformation_applied,
            "Transformed Series": transformed_series
        }

        return results
    
    except Exception as e:
        raise RuntimeError(f"Error in preprocessing time series: {str(e)}")
