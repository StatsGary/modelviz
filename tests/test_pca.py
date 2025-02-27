import unittest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from modelviz.pca import plot_pca_projection  

class TestPCAProjection(unittest.TestCase):
    
    def setUp(self):
        """Create dummy data for testing different PCA scenarios."""
        self.df_2d_original = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
        self.df_2d_projected = pd.DataFrame({'PC1': [1.1, 2.1, 3.1], 'PC2': [3.9, 4.9, 5.9]})

        self.df_3d_original = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6], 'Feature3': [7, 8, 9]})
        self.df_3d_projected = pd.DataFrame({'PC1': [1.1, 2.1, 3.1], 'PC2': [3.9, 4.9, 5.9], 'PC3': [6.9, 7.9, 8.9]})

        self.df_4d_original = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6], 'Feature3': [7, 8, 9], 'Feature4': [10, 11, 12]})
        self.df_4d_projected = pd.DataFrame({'PC1': [1.1, 2.1, 3.1], 'PC2': [3.9, 4.9, 5.9], 'PC3': [6.9, 7.9, 8.9], 'PC4': [9.9, 10.9, 11.9]})

    @patch("matplotlib.pyplot.show")  # Prevent actual plot display
    def test_plot_2d(self, mock_show):
        """Test if 2D PCA plot runs without errors."""
        result = plot_pca_projection(self.df_2d_original, self.df_2d_projected)
        self.assertIsNone(result)  # Function should return None

    @patch("matplotlib.pyplot.show")  
    def test_plot_3d(self, mock_show):
        """Test if 3D PCA plot runs without errors."""
        result = plot_pca_projection(self.df_3d_original, self.df_3d_projected)
        self.assertIsNone(result)  # Function should return None

    @patch("seaborn.pairplot")  
    @patch("matplotlib.pyplot.show")  
    def test_plot_4d(self, mock_show, mock_pairplot):
        """Test if 4D+ PCA plot runs without errors and uses pairplot."""
        result = plot_pca_projection(self.df_4d_original, self.df_4d_projected)
        self.assertIsNone(result)  # Function should return None
        mock_pairplot.assert_called()  # Ensure Seaborn pairplot was used for 4D data

    @patch("matplotlib.pyplot.show")  
    def test_custom_parameters(self, mock_show):
        """Test if the function accepts custom labels, colors, and titles."""
        result = plot_pca_projection(
            self.df_3d_original, self.df_3d_projected,
            original_label="Custom Original",
            projected_label="Custom Projected",
            original_color="black",
            projected_color="green",
            line_color="purple",
            title_2d="Custom 2D Title",
            title_3d="Custom 3D Title",
            title_nD="Custom nD Title"
        )
        self.assertIsNone(result)  # Function should return None

if __name__ == "__main__":
    unittest.main()
