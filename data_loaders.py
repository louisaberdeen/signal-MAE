"""
Data loader abstractions for audio datasets.

This module provides a flexible interface for loading different audio datasets
with FiftyOne integration. Designed for extensibility to any IQ data (audio, RF signals).

Key components:
- AudioDataLoader: Abstract base class defining the interface
- ESC50DataLoader: Concrete implementation for ESC-50 dataset
- CustomAudioDataLoader: Generic loader with configurable column mapping
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np


class AudioDataLoader(ABC):
    """Abstract base class for audio dataset loaders.

    All loaders must return metadata in a standard schema:
    - filepath: str (path to audio file)
    - label: str or int (class/category label)
    - lat: float (optional, latitude)
    - lon: float (optional, longitude)
    - Additional columns are preserved as-is
    """

    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Root directory of the dataset
        """
        self.data_root = Path(data_root)

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata with standard schema.

        Returns:
            DataFrame with at minimum:
            - filepath: Full path to audio file
            - label: Class/category label
            - lat: Latitude (optional, can be NaN)
            - lon: Longitude (optional, can be NaN)
        """
        pass

    @abstractmethod
    def get_audio_paths(self) -> List[Path]:
        """Get list of all audio file paths.

        Returns:
            List of Path objects to audio files
        """
        pass

    def get_spectrogram_paths(self, spectrogram_dir: Path) -> Dict[str, Path]:
        """Get mapping from audio filename to spectrogram path.

        Args:
            spectrogram_dir: Directory containing precomputed spectrograms

        Returns:
            Dict mapping audio filename to spectrogram path
        """
        audio_paths = self.get_audio_paths()
        spectrogram_dir = Path(spectrogram_dir)

        mapping = {}
        for audio_path in audio_paths:
            audio_filename = audio_path.name
            spec_filename = audio_filename.replace('.wav', '.npy')
            spec_path = spectrogram_dir / spec_filename

            if spec_path.exists():
                mapping[audio_filename] = spec_path

        return mapping

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate dataset integrity.

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

        # Check file existence
        missing_files = []
        for filepath in metadata['filepath']:
            if not Path(filepath).exists():
                missing_files.append(filepath)

        if missing_files:
            errors.append(f"{len(missing_files)} audio files not found")
            if len(missing_files) <= 5:
                errors.append(f"Missing files: {missing_files}")

        is_valid = len(errors) == 0
        return is_valid, errors


class ESC50DataLoader(AudioDataLoader):
    """Data loader for ESC-50 environmental sound dataset.

    ESC-50 structure:
    - data_root/
      - audio/
        - *.wav (2000 files)
      - meta/
        - esc50.csv (metadata)

    Metadata columns:
    - filename: Audio filename
    - fold: Cross-validation fold (1-5)
    - target: Class label (0-49)
    - category: Semantic category
    - esc10: Boolean, part of ESC-10 subset
    - src_file: Original Freesound filename
    - take: Letter indicating fragment
    """

    def __init__(
        self,
        data_root: Path,
        add_synthetic_geo: bool = False,
        geo_seed: int = 42
    ):
        """
        Args:
            data_root: Root directory of ESC-50 dataset
            add_synthetic_geo: Whether to add synthetic lat/long
            geo_seed: Random seed for synthetic geo generation
        """
        super().__init__(data_root)
        self.audio_dir = self.data_root / "audio"
        self.metadata_csv = self.data_root / "meta" / "esc50.csv"
        self.add_synthetic_geo = add_synthetic_geo
        self.geo_seed = geo_seed

    def load_metadata(self) -> pd.DataFrame:
        """Load ESC-50 metadata with standard schema.

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

        # Add synthetic geographic coordinates if requested
        if self.add_synthetic_geo:
            from embeddings_utils import SyntheticGeoGenerator
            geo_gen = SyntheticGeoGenerator(seed=self.geo_seed)
            df = geo_gen.generate_for_dataframe(df, category_column='category')

            # Rename to standard schema
            df['lat'] = df['latitude']
            df['lon'] = df['longitude']
        else:
            # Add empty lat/lon columns
            df['lat'] = np.nan
            df['lon'] = np.nan

        return df

    def get_audio_paths(self) -> List[Path]:
        """Get list of all ESC-50 audio file paths.

        Returns:
            List of paths to .wav files
        """
        return sorted(self.audio_dir.glob("*.wav"))

    def get_info(self) -> Dict[str, any]:
        """Get dataset information.

        Returns:
            Dict with dataset statistics
        """
        metadata = self.load_metadata()

        return {
            'dataset': 'ESC-50',
            'total_samples': len(metadata),
            'num_classes': metadata['target'].nunique(),
            'num_categories': metadata['category'].nunique(),
            'num_folds': metadata['fold'].nunique(),
            'categories': metadata['category'].unique().tolist(),
            'has_geo': not metadata['lat'].isna().all(),
        }


class CustomAudioDataLoader(AudioDataLoader):
    """Generic data loader for custom audio datasets.

    Flexible loader that can adapt to different metadata formats via
    column mapping configuration.

    Example usage:
    ```python
    loader = CustomAudioDataLoader(
        data_root=Path("path/to/dataset"),
        metadata_file="metadata.csv",
        audio_dir="audio",
        column_mapping={
            'filepath': 'audio_path',    # Map 'audio_path' column to 'filepath'
            'label': 'class_id',          # Map 'class_id' column to 'label'
            'lat': 'latitude',            # Map 'latitude' column to 'lat'
            'lon': 'longitude'            # Map 'longitude' column to 'lon'
        },
        audio_extension='.wav'
    )
    ```
    """

    def __init__(
        self,
        data_root: Path,
        metadata_file: str,
        audio_dir: str = "audio",
        column_mapping: Optional[Dict[str, str]] = None,
        audio_extension: str = ".wav",
        add_path_prefix: bool = True
    ):
        """
        Args:
            data_root: Root directory of dataset
            metadata_file: Path to metadata CSV (relative to data_root)
            audio_dir: Directory containing audio files (relative to data_root)
            column_mapping: Dict mapping standard schema to dataset columns
                           e.g., {'filepath': 'audio_path', 'label': 'class'}
            audio_extension: Audio file extension (e.g., '.wav', '.mp3')
            add_path_prefix: Whether to prepend audio_dir to filepaths
        """
        super().__init__(data_root)
        self.metadata_file = self.data_root / metadata_file
        self.audio_dir = self.data_root / audio_dir
        self.column_mapping = column_mapping or {}
        self.audio_extension = audio_extension
        self.add_path_prefix = add_path_prefix

    def load_metadata(self) -> pd.DataFrame:
        """Load custom metadata with column mapping.

        Returns:
            DataFrame with metadata mapped to standard schema
        """
        # Load CSV
        df = pd.read_csv(self.metadata_file)

        # Apply column mapping
        mapped_df = pd.DataFrame()

        # Map filepath
        if 'filepath' in self.column_mapping:
            filepath_col = self.column_mapping['filepath']
            if filepath_col in df.columns:
                filepaths = df[filepath_col]
            else:
                raise ValueError(f"Filepath column '{filepath_col}' not found")
        else:
            # Assume first column is filepath
            filepaths = df.iloc[:, 0]

        # Add audio_dir prefix if requested
        if self.add_path_prefix:
            mapped_df['filepath'] = filepaths.apply(
                lambda x: str(self.audio_dir / x)
            )
        else:
            mapped_df['filepath'] = filepaths.apply(str)

        # Map label
        if 'label' in self.column_mapping:
            label_col = self.column_mapping['label']
            if label_col in df.columns:
                mapped_df['label'] = df[label_col].astype(str)
            else:
                raise ValueError(f"Label column '{label_col}' not found")
        else:
            # Try common column names
            for col in ['label', 'class', 'target', 'category']:
                if col in df.columns:
                    mapped_df['label'] = df[col].astype(str)
                    break
            else:
                raise ValueError("No label column found. Specify in column_mapping.")

        # Map lat/lon if available
        for geo_field in ['lat', 'lon']:
            if geo_field in self.column_mapping:
                geo_col = self.column_mapping[geo_field]
                if geo_col in df.columns:
                    mapped_df[geo_field] = df[geo_col]
                else:
                    mapped_df[geo_field] = np.nan
            else:
                # Try common names
                common_names = {
                    'lat': ['lat', 'latitude', 'Latitude'],
                    'lon': ['lon', 'long', 'longitude', 'Longitude']
                }
                for col in common_names[geo_field]:
                    if col in df.columns:
                        mapped_df[geo_field] = df[col]
                        break
                else:
                    mapped_df[geo_field] = np.nan

        # Copy all other columns
        for col in df.columns:
            if col not in self.column_mapping.values() and col not in mapped_df.columns:
                mapped_df[col] = df[col]

        return mapped_df

    def get_audio_paths(self) -> List[Path]:
        """Get list of all audio file paths.

        Returns:
            List of paths to audio files
        """
        pattern = f"*{self.audio_extension}"
        return sorted(self.audio_dir.glob(pattern))


class RFSignalDataLoader(AudioDataLoader):
    """Data loader for RF/IQ signal datasets.

    Placeholder for future RF signal support. RF signals can be treated
    similarly to audio with appropriate preprocessing.

    Expected format:
    - IQ data stored as .npy files (complex samples)
    - Metadata CSV with signal parameters
    - Optional: Precomputed spectrograms
    """

    def __init__(
        self,
        data_root: Path,
        metadata_file: str,
        signal_dir: str = "signals",
        signal_extension: str = ".npy"
    ):
        """
        Args:
            data_root: Root directory of dataset
            metadata_file: Path to metadata CSV
            signal_dir: Directory containing IQ signal files
            signal_extension: Signal file extension (.npy, .dat, etc.)
        """
        super().__init__(data_root)
        self.metadata_file = self.data_root / metadata_file
        self.signal_dir = self.data_root / signal_dir
        self.signal_extension = signal_extension

    def load_metadata(self) -> pd.DataFrame:
        """Load RF signal metadata.

        Returns:
            DataFrame with standard schema for RF signals
        """
        # Load CSV
        df = pd.read_csv(self.metadata_file)

        # Map to standard schema
        # (Implementation depends on specific RF dataset format)
        # This is a template - customize based on your RF data

        df['filepath'] = df['filename'].apply(
            lambda x: str(self.signal_dir / x)
        )

        # RF-specific fields might include:
        # - center_frequency
        # - sample_rate
        # - modulation_type
        # etc.

        return df

    def get_audio_paths(self) -> List[Path]:
        """Get list of all RF signal file paths.

        Returns:
            List of paths to signal files
        """
        pattern = f"*{self.signal_extension}"
        return sorted(self.signal_dir.glob(pattern))


def create_data_loader(
    dataset_type: str,
    data_root: Path,
    **kwargs
) -> AudioDataLoader:
    """Factory function to create appropriate data loader.

    Args:
        dataset_type: Type of dataset ('esc50', 'custom', 'rf')
        data_root: Root directory of dataset
        **kwargs: Additional arguments for specific loader

    Returns:
        AudioDataLoader instance

    Example:
    ```python
    # ESC-50
    loader = create_data_loader('esc50', Path('data/ESC-50-master'))

    # Custom dataset
    loader = create_data_loader(
        'custom',
        Path('data/my_dataset'),
        metadata_file='metadata.csv',
        column_mapping={'filepath': 'audio_path', 'label': 'class'}
    )
    ```
    """
    loaders = {
        'esc50': ESC50DataLoader,
        'custom': CustomAudioDataLoader,
        'rf': RFSignalDataLoader
    }

    if dataset_type not in loaders:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available: {list(loaders.keys())}"
        )

    loader_class = loaders[dataset_type]
    return loader_class(data_root, **kwargs)
