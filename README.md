# modelviz

**modelviz** is a Python package designed to simplify the visualization of data distributions, making it a powerful tool for exploratory data analysis (EDA). With customizable histogram plotting and support for popular visualization libraries like Matplotlib and Plotly, `modelviz` helps users quickly understand their data.

## Installation

Install `modelviz` via pip:

```bash
pip install modelviz
```

## Features
Plot Feature Histograms:

1. Generate histograms for all numeric columns in a pandas DataFrame.
2. Exclude binary-encoded columns for cleaner visualizations.
3. Customize the number of bins, colors, and labels.

## Importing the Package

```python
import pandas as pd
from modelviz import plot_feature_histograms
```

