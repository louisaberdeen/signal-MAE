"""
Geographic coordinate generation utilities.

SyntheticGeoGenerator creates synthetic lat/long coordinates for
datasets that don't have real location data.
"""

from typing import Tuple, Dict

import numpy as np
import pandas as pd


class SyntheticGeoGenerator:
    """
    Generate synthetic lat/long coordinates for visualization.

    Creates geographic patterns where similar categories are clustered
    in specific regions for meaningful map visualization.

    Args:
        seed: Random seed for reproducibility
    """

    # Default category centers (major cities)
    DEFAULT_CENTERS: Dict[str, Tuple[float, float]] = {
        'Animals': (40.7128, -74.0060),  # New York
        'Natural soundscapes and water sounds': (47.6062, -122.3321),  # Seattle
        'Human, non-speech sounds': (51.5074, -0.1278),  # London
        'Interior/domestic sounds': (35.6762, 139.6503),  # Tokyo
        'Exterior/urban noises': (34.0522, -118.2437)  # Los Angeles
    }

    def __init__(
        self,
        seed: int = 42,
        category_centers: Dict[str, Tuple[float, float]] = None
    ):
        self.seed = seed
        self.category_centers = category_centers or self.DEFAULT_CENTERS

    def generate(
        self,
        category: str,
        spread: float = 0.5
    ) -> Tuple[float, float]:
        """
        Generate synthetic lat/long for a category.

        Args:
            category: Category name
            spread: Standard deviation of offset (~55km per 0.5 degrees)

        Returns:
            Tuple of (latitude, longitude)
        """
        # Set seed based on category for consistency
        np.random.seed(self.seed + hash(category) % 1000)

        # Get center for this category
        center_lat, center_lon = self.category_centers.get(
            category,
            (0.0, 0.0)  # Default to equator/prime meridian
        )

        # Add random offset
        lat_offset = np.random.normal(0, spread)
        lon_offset = np.random.normal(0, spread)

        latitude = center_lat + lat_offset
        longitude = center_lon + lon_offset

        return latitude, longitude

    def generate_for_dataframe(
        self,
        df: pd.DataFrame,
        category_column: str = 'category',
        spread: float = 0.5
    ) -> pd.DataFrame:
        """
        Add synthetic lat/long columns to dataframe.

        Args:
            df: DataFrame with category column
            category_column: Name of category column
            spread: Standard deviation of offset

        Returns:
            DataFrame with added 'latitude' and 'longitude' columns
        """
        # Reset random state for reproducibility
        np.random.seed(self.seed)

        lats = []
        lons = []

        for idx, row in df.iterrows():
            category = row[category_column]
            lat, lon = self.generate(category, spread)
            lats.append(lat)
            lons.append(lon)

        df = df.copy()
        df['latitude'] = lats
        df['longitude'] = lons

        return df

    def add_center(
        self,
        category: str,
        lat: float,
        lon: float
    ) -> None:
        """
        Add or update a category center.

        Args:
            category: Category name
            lat: Center latitude
            lon: Center longitude
        """
        self.category_centers[category] = (lat, lon)
