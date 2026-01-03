"""
ESC-50 Environmental Sound Classification dataset loader.

ESC-50 is a labeled collection of 2000 environmental audio recordings
suitable for benchmarking environmental sound classification methods.

Dataset structure:
- data_root/
  - audio/
    - *.wav (2000 files, 5 seconds each)
  - meta/
    - esc50.csv (metadata)

Reference: https://github.com/karolpiczak/ESC-50
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from src.registry import data_loader_registry
from src.data.base import BaseDataLoader


@data_loader_registry.register("esc50", version="2.0")
class ESC50DataLoader(BaseDataLoader):
    """
    Data loader for ESC-50 environmental sound dataset.

    ESC-50 contains 2000 5-second audio clips organized into:
    - 50 semantic classes
    - 5 major categories (Animals, Nature, Human, Interior, Urban)
    - 5 cross-validation folds

    Metadata columns:
    - filename: Audio filename
    - fold: Cross-validation fold (1-5)
    - target: Class label (0-49)
    - category: Semantic category name
    - esc10: Boolean, part of ESC-10 subset
    - src_file: Original Freesound filename
    - take: Letter indicating fragment

    Args:
        data_root: Root directory of ESC-50 dataset
        add_synthetic_geo: Whether to add synthetic lat/long coordinates
        geo_seed: Random seed for synthetic geo generation
    """

    # ESC-50 category centers for synthetic geo (major cities)
    CATEGORY_CENTERS = {
        'Animals': (40.7128, -74.0060),  # New York
        'Natural soundscapes and water sounds': (47.6062, -122.3321),  # Seattle
        'Human, non-speech sounds': (51.5074, -0.1278),  # London
        'Interior/domestic sounds': (35.6762, 139.6503),  # Tokyo
        'Exterior/urban noises': (34.0522, -118.2437)  # Los Angeles
    }

    def __init__(
        self,
        data_root: Path,
        add_synthetic_geo: bool = False,
        geo_seed: int = 42,
        **kwargs
    ):
        super().__init__(data_root, **kwargs)
        self.audio_dir = self.data_root / "audio"
        self.metadata_csv = self.data_root / "meta" / "esc50.csv"
        self.add_synthetic_geo = add_synthetic_geo
        self.geo_seed = geo_seed

    def load_metadata(self) -> pd.DataFrame:
        """
        Load ESC-50 metadata with standard schema.

        Returns:
            DataFrame with ESC-50 metadata mapped to standard schema
        """
        # Load CSV
        df = pd.read_csv(self.metadata_csv)

        # Add full file paths
        df['filepath'] = df['filename'].apply(
            lambda x: str(self.audio_dir / x)
        )

        # Map 'target' to 'label' for standard schema
        df['label'] = df['target'].astype(str)

        # Add geographic coordinates
        if self.add_synthetic_geo:
            df = self._add_synthetic_geo(df)
        else:
            df['lat'] = np.nan
            df['lon'] = np.nan

        return df

    def _add_synthetic_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic geographic coordinates based on category.

        Args:
            df: DataFrame with 'category' column

        Returns:
            DataFrame with added 'lat' and 'lon' columns
        """
        np.random.seed(self.geo_seed)

        lats = []
        lons = []

        for idx, row in df.iterrows():
            category = row['category']
            center_lat, center_lon = self.CATEGORY_CENTERS.get(
                category, (0.0, 0.0)
            )

            # Add random offset (~50km radius)
            lat_offset = np.random.normal(0, 0.5)
            lon_offset = np.random.normal(0, 0.5)

            lats.append(center_lat + lat_offset)
            lons.append(center_lon + lon_offset)

        df['lat'] = lats
        df['lon'] = lons

        return df

    def get_sample_paths(self) -> List[Path]:
        """
        Get list of all ESC-50 audio file paths.

        Returns:
            List of paths to .wav files
        """
        return sorted(self.audio_dir.glob("*.wav"))

    def get_info(self) -> Dict[str, Any]:
        """
        Get ESC-50 dataset information.

        Returns:
            Dict with dataset statistics
        """
        metadata = self.load_metadata()

        return {
            'dataset': 'ESC-50',
            'data_root': str(self.data_root),
            'total_samples': len(metadata),
            'num_classes': metadata['target'].nunique(),
            'num_categories': metadata['category'].nunique(),
            'num_folds': metadata['fold'].nunique(),
            'categories': metadata['category'].unique().tolist(),
            'has_geo': not metadata['lat'].isna().all(),
            'esc10_count': metadata['esc10'].sum() if 'esc10' in metadata else 0,
        }

    def get_fold_split(
        self,
        val_fold: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/validation split based on fold.

        Args:
            val_fold: Which fold to use for validation (1-5)

        Returns:
            Tuple of (train_df, val_df)
        """
        metadata = self.load_metadata()
        train_df = metadata[metadata['fold'] != val_fold]
        val_df = metadata[metadata['fold'] == val_fold]
        return train_df, val_df

    def get_category_mapping(self) -> Dict[int, str]:
        """
        Get mapping from target ID to category name.

        Returns:
            Dict mapping target (0-49) to category string
        """
        metadata = self.load_metadata()
        return dict(zip(metadata['target'], metadata['category']))
