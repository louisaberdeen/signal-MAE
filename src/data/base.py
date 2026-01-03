"""
Abstract base class for dataset loaders.

All data loader plugins must inherit from BaseDataLoader and implement
the required abstract methods for metadata loading and path discovery.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd
import numpy as np


class BaseDataLoader(ABC):
    """
    Abstract base class for dataset loaders.

    All loaders must return metadata in a standard schema:
    - filepath: str (path to audio/signal file)
    - label: str or int (class/category label)
    - lat: float (optional, latitude)
    - lon: float (optional, longitude)

    Subclasses must implement:
        - load_metadata: Load and standardize dataset metadata
        - get_sample_paths: Return list of all sample file paths

    Args:
        data_root: Root directory of the dataset
        **kwargs: Additional configuration options
    """

    def __init__(self, data_root: Path, **kwargs):
        self.data_root = Path(data_root)
        self.config = kwargs

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata with standard schema.

        Returns:
            DataFrame with at minimum:
            - filepath: Full path to sample file
            - label: Class/category label
            - lat: Latitude (optional, can be NaN)
            - lon: Longitude (optional, can be NaN)
        """
        pass

    @abstractmethod
    def get_sample_paths(self) -> List[Path]:
        """
        Get list of all sample file paths.

        Returns:
            List of Path objects to sample files
        """
        pass

    def get_spectrogram_paths(
        self,
        spectrogram_dir: Path
    ) -> Dict[str, Path]:
        """
        Get mapping from sample filename to spectrogram path.

        Args:
            spectrogram_dir: Directory containing precomputed spectrograms

        Returns:
            Dict mapping sample filename to spectrogram path
        """
        sample_paths = self.get_sample_paths()
        spectrogram_dir = Path(spectrogram_dir)

        mapping = {}
        for sample_path in sample_paths:
            # Replace extension with .npy
            spec_filename = sample_path.stem + '.npy'
            spec_path = spectrogram_dir / spec_filename

            if spec_path.exists():
                mapping[sample_path.name] = spec_path

        return mapping

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset integrity.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check data root exists
        if not self.data_root.exists():
            errors.append(f"Data root not found: {self.data_root}")
            return False, errors

        # Try to load metadata
        try:
            metadata = self.load_metadata()
        except Exception as e:
            errors.append(f"Failed to load metadata: {e}")
            return False, errors

        # Check required columns
        required_cols = ['filepath', 'label']
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check file existence (sample check)
        sample_size = min(10, len(metadata))
        missing_files = []
        for filepath in metadata['filepath'].head(sample_size):
            if not Path(filepath).exists():
                missing_files.append(filepath)

        if missing_files:
            errors.append(f"{len(missing_files)} sample files not found")
            if len(missing_files) <= 5:
                errors.append(f"Missing files: {missing_files}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_info(self) -> Dict[str, Any]:
        """
        Get dataset information.

        Returns:
            Dict with dataset statistics
        """
        try:
            metadata = self.load_metadata()
            return {
                'dataset': self.__class__.__name__,
                'data_root': str(self.data_root),
                'total_samples': len(metadata),
                'num_classes': metadata['label'].nunique(),
                'has_geo': not metadata.get('lat', pd.Series([np.nan])).isna().all(),
            }
        except Exception as e:
            return {
                'dataset': self.__class__.__name__,
                'data_root': str(self.data_root),
                'error': str(e),
            }

    @classmethod
    def get_plugin_info(cls) -> Dict[str, Any]:
        """Return plugin registration information."""
        return {
            'name': cls.__name__,
            'registry_key': getattr(cls, '_registry_key', None),
            'registry_name': getattr(cls, '_registry_name', None),
        }
