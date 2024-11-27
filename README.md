
# 🤖🤖 modelviz - Python package to make visualizations a breeze 🤖🤖

![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**modelviz** is a Python package designed for comprehensive and customizable data visualization and model evaluation. With modules for visualizing relationships, confusion matrices, ROC curves, data distributions, and handling missing values, `modelviz` simplifies exploratory data analysis (EDA) and model performance evaluation.

## Installation

Install `modelviz` via pip:

```bash
pip install modelviz
```

## Features

### 1. Confusion Matrix (`confusion_matrix.py`)
- **Visualize Confusion Matrices**:
  - Supports both binary and multi-class confusion matrices.
  - Displays proportions, TP, FP, FN, and TN labels.
  - Includes detailed metrics like Accuracy, Precision, Recall, F1 Score, MCC, and Cohen's Kappa.
  - Option to normalize the confusion matrix.

#### Example Usage:
```python
from modelviz.confusion_matrix import plot_confusion_matrix
import numpy as np

cm = np.array([[50, 10], [5, 35]])  # Binary confusion matrix
classes = ["Negative", "Positive"]
plot_confusion_matrix(cm, classes, "Logistic Regression")
```

---

### 2. Histogram (`histogram.py`)
- **Feature Histograms**:
  - Automatically generate histograms for all numeric columns in a pandas DataFrame.
  - Skip binary columns for cleaner visualizations.
  - Customize bins, colors, and titles.

#### Example Usage:
```python
from modelviz.histogram import plot_feature_histograms
import pandas as pd

df = pd.DataFrame({
    'Age': [25, 30, 35, 40],
    'Income': [40000, 50000, 60000, 70000],
    'Gender': [0, 1, 0, 1]
})
plot_feature_histograms(df, exclude_binary=True, bins=10, color='blue')
```

---

### 3. ROC Curve (`roc.py`)
- **ROC Curve Visualization**:
  - Plot Receiver Operating Characteristic (ROC) curves.
  - Highlight thresholds like Youden's J and adjusted thresholds.
  - Display key metrics like AUC (Area Under Curve).

#### Example Usage:
```python
from modelviz.roc import plot_roc_curve_with_youdens_thresholds

fpr = [0.0, 0.1, 0.2, 0.3]
tpr = [0.0, 0.4, 0.6, 1.0]
thresholds = [1.0, 0.8, 0.5, 0.2]
plot_roc_curve_with_youdens_thresholds(fpr, tpr, thresholds, roc_auc=0.85, model_name="My Model")
```

---

### 4. Relationships (`relationships.py`)
- **Correlation Matrix**:
  - Generate and visualize correlation matrices for numeric features.
  - Customize heatmaps with annotations, colormap, and figure size.

#### Example Usage:
```python
from modelviz.relationships import plot_correlation_matrix
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [4, 3, 2, 1],
    'C': [5, 6, 7, 8]
})
plot_correlation_matrix(df, method='pearson')
```

---

### 5. K-Fold Visualization (`kfold.py`)
- **Visualize K-Fold Splits**:
  - Display data distribution across training and validation sets for K-Fold Cross-Validation.
  - Easy visualization for understanding fold assignments.

---

### 6. Handling Missing Values (`missvals.py`)
- **Missing Value Analysis**:
  - Visualize missing data in a DataFrame.
  - Quickly identify patterns and percentage of missing values.

---

### 7. Model Evaluation (`model_eval.py`)
- **Aggregate Model Metrics**:
  - Summarize key evaluation metrics for multiple models.
  - Compare performance across models.

---

## Importing the Package

Each module in the package is designed to be imported separately. For example:

```python
from modelviz.confusion_matrix import plot_confusion_matrix
from modelviz.histogram import plot_feature_histograms
from modelviz.roc import plot_roc_curve_with_youdens_thresholds
```

## Contributing
Contributions are welcome! If you have suggestions or new feature ideas, feel free to open an issue or create a pull request on GitHub.
