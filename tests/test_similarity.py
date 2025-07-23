import pytest
import numpy as np
from modelviz.relationships import plot_similarity  
import matplotlib.pyplot as plt 

# Sample test data
np.random.seed(42)
num_points = 10
dimensions = 3
data = np.random.rand(num_points, dimensions)
point_of_interest = data[0]

def test_gaussian_similarity():
    distances, similarity_scores = plot_similarity(data, point_of_interest, mode="gaussian", seaborn_style="whitegrid")
    plt.close("all")
    assert isinstance(distances, np.ndarray), "Distances should be a NumPy array"
    assert isinstance(similarity_scores, np.ndarray), "Similarity scores should be a NumPy array"
    assert len(distances) == len(similarity_scores), "Distances and similarity scores should have the same length"
    assert distances.shape[0] == num_points, "Output should match number of data points"

def test_tsne():
    transformed_data = plot_similarity(data, point_of_interest, mode="tsne", perplexity=5, seaborn_style="darkgrid")
    plt.close("all")
    assert isinstance(transformed_data, np.ndarray), "t-SNE output should be a NumPy array"
    assert transformed_data.shape[1] == 2, "t-SNE output should have 2 dimensions"
    assert transformed_data.shape[0] == num_points, "t-SNE output should match number of data points"

def test_pca():
    transformed_data = plot_similarity(data, point_of_interest, mode="pca", pca_components=2, seaborn_style="ticks")
    plt.close("all")
    assert isinstance(transformed_data, np.ndarray), "PCA output should be a NumPy array"
    assert transformed_data.shape[1] == 2, "PCA output should have 2 dimensions"
    assert transformed_data.shape[0] == num_points, "PCA output should match number of data points"

def test_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode. Choose 'gaussian', 'tsne', or 'pca'"):
        plot_similarity(data, point_of_interest, mode="invalid_mode")

