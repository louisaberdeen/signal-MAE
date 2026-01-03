"""
Generic data loader for custom audio datasets.

Flexible loader that can adapt to different metadata formats via
column mapping configuration.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np

from src.registry import data_loader_registry
from src.data.base import BaseDataLoader


@data_loader_registry.register("custom", version="2.0")
class CustomAudioDataLoader(BaseDataLoader):
    """
    Generic data loader for custom audio datasets.

    Flexible loader that can adapt to different metadata formats via
    column mapping configuration. Useful for quick prototyping with
    new datasets that have non-standard column names.

    Args:
        data_root: Root directory of dataset
        metadata_file: Path to metadata CSV (relative to data_root)
        audio_dir: Directory containing audio files (relative to data_root)
        column_mapping: Dict mapping standard schema to dataset columns
            e.g., {'filepath': 'audio_path', 'label': 'class'}
        audio_extension: Audio file extension (e.g., '.wav', '.mp3')
        add_path_prefix: Whether to prepend audio_dir to filepaths

    Example:
        loader = CustomAudioDataLoader(
            data_root=Path("data/my_dataset"),
            metadata_file="metadata.csv",
            audio_dir="audio",
            column_mapping={
                'filepath': 'audio_path',
                'label': 'class_id',
                'lat': 'latitude',
                'lon': 'longitude'
            }
        )
    """

    def __init__(
        self,
        data_root: Path,
        metadata_file: str,
        audio_dir: str = "audio",
        column_mapping: Optional[Dict[str, str]] = None,
        audio_extension: str = ".wav",
        add_path_prefix: bool = True,
        **kwargs
    ):
        super().__init__(data_root, **kwargs)
        self.metadata_file = self.data_root / metadata_file
        self.audio_dir = self.data_root / audio_dir
        self.column_mapping = column_mapping or {}
        self.audio_extension = audio_extension
        self.add_path_prefix = add_path_prefix

    def load_metadata(self) -> pd.DataFrame:
        """
        Load custom metadata with column mapping.

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
            if filepath_col not in df.columns:
                raise ValueError(f"Filepath column '{filepath_col}' not found")
            filepaths = df[filepath_col]
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
            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found")
            mapped_df['label'] = df[label_col].astype(str)
        else:
            # Try common column names
            for col in ['label', 'class', 'target', 'category']:
                if col in df.columns:
                    mapped_df['label'] = df[col].astype(str)
                    break
            else:
                raise ValueError(
                    "No label column found. Specify in column_mapping."
                )

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

    def get_sample_paths(self) -> List[Path]:
        """
        Get list of all audio file paths.

        Returns:
            List of paths to audio files
        """
        pattern = f"*{self.audio_extension}"
        return sorted(self.audio_dir.glob(pattern))


@data_loader_registry.register("rf", version="1.0")
class RFSignalDataLoader(BaseDataLoader):
    """
    Data loader for RF/IQ signal datasets.

    Placeholder for future RF signal support. RF signals can be treated
    similarly to audio with appropriate preprocessing.

    Expected format:
    - IQ data stored as .npy files (complex samples)
    - Metadata CSV with signal parameters
    - Optional: Precomputed spectrograms

    Args:
        data_root: Root directory of dataset
        metadata_file: Path to metadata CSV
        signal_dir: Directory containing IQ signal files
        signal_extension: Signal file extension (.npy, .dat, etc.)
    """

    def __init__(
        self,
        data_root: Path,
        metadata_file: str,
        signal_dir: str = "signals",
        signal_extension: str = ".npy",
        **kwargs
    ):
        super().__init__(data_root, **kwargs)
        self.metadata_file = self.data_root / metadata_file
        self.signal_dir = self.data_root / signal_dir
        self.signal_extension = signal_extension

    def load_metadata(self) -> pd.DataFrame:
        """
        Load RF signal metadata.

        Returns:
            DataFrame with standard schema for RF signals
        """
        # Load CSV
        df = pd.read_csv(self.metadata_file)

        # Map to standard schema
        df['filepath'] = df['filename'].apply(
            lambda x: str(self.signal_dir / x)
        )

        # Ensure label column exists
        if 'label' not in df.columns:
            if 'modulation' in df.columns:
                df['label'] = df['modulation']
            elif 'class' in df.columns:
                df['label'] = df['class']

        # Add geo columns
        if 'lat' not in df.columns:
            df['lat'] = np.nan
        if 'lon' not in df.columns:
            df['lon'] = np.nan

        return df

    def get_sample_paths(self) -> List[Path]:
        """
        Get list of all RF signal file paths.

        Returns:
            List of paths to signal files
        """
        pattern = f"*{self.signal_extension}"
        return sorted(self.signal_dir.glob(pattern))
