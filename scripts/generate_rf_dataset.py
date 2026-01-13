#!/usr/bin/env python3
"""
CLI script for generating RF datasets using TorchSig.

This script provides a command-line interface for generating synthetic
RF datasets with comprehensive configuration options.

Usage:
    # Generate with preset
    python scripts/generate_rf_dataset.py --preset classification --samples 10000

    # Generate with custom modulations
    python scripts/generate_rf_dataset.py --name my_dataset \
        --modulations bpsk qpsk 16qam 64qam \
        --samples 5000 \
        --snr-min -5 --snr-max 25

    # Generate from config file
    python scripts/generate_rf_dataset.py --config config.json

    # List available modulations
    python scripts/generate_rf_dataset.py --list-modulations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.torchsig_config import (
    TorchSigConfig,
    ImpairmentLevel,
    MODULATION_TYPES,
    get_all_modulations,
)
from src.data.torchsig_generator import TorchSigGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic RF datasets using TorchSig',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test dataset
  %(prog)s --preset test

  # Classification dataset with 10k samples
  %(prog)s --preset classification --samples 10000 --name my_rf_data

  # Custom modulations with specific SNR range
  %(prog)s --name custom_rf \\
      --modulations bpsk qpsk 8psk 16qam 64qam \\
      --samples 5000 \\
      --snr-min 0 --snr-max 20 \\
      --impairment wireless

  # Detection dataset (multiple signals)
  %(prog)s --preset detection --max-signals 5

  # From config file
  %(prog)s --config my_config.json
        """
    )

    # Preset options
    preset_group = parser.add_argument_group('Preset Options')
    preset_group.add_argument(
        '--preset',
        choices=['classification', 'detection', 'test'],
        help='Use a predefined configuration preset'
    )
    preset_group.add_argument(
        '--difficulty',
        choices=['easy', 'medium', 'hard'],
        default='medium',
        help='Difficulty level for classification preset (default: medium)'
    )

    # Basic options
    basic_group = parser.add_argument_group('Basic Options')
    basic_group.add_argument(
        '--name',
        type=str,
        default='torchsig_rf',
        help='Dataset name (default: torchsig_rf)'
    )
    basic_group.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/rf_datasets'),
        help='Output directory (default: data/rf_datasets)'
    )
    basic_group.add_argument(
        '--samples', '-n',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10000)'
    )

    # Modulation options
    mod_group = parser.add_argument_group('Modulation Options')
    mod_group.add_argument(
        '--modulations', '-m',
        nargs='+',
        help='List of modulation types (e.g., bpsk qpsk 16qam)'
    )
    mod_group.add_argument(
        '--families', '-f',
        nargs='+',
        choices=['ask', 'psk', 'qam', 'fsk', 'ofdm', 'analog'],
        help='Include all modulations from specified families'
    )
    mod_group.add_argument(
        '--list-modulations',
        action='store_true',
        help='List all available modulation types and exit'
    )

    # Signal parameters
    signal_group = parser.add_argument_group('Signal Parameters')
    signal_group.add_argument(
        '--sample-rate',
        type=float,
        default=10e6,
        help='Sample rate in Hz (default: 10e6 = 10 MHz)'
    )
    signal_group.add_argument(
        '--iq-samples',
        type=int,
        default=4096,
        help='Number of IQ samples per signal (default: 4096)'
    )
    signal_group.add_argument(
        '--fft-size',
        type=int,
        default=1024,
        help='FFT size for spectrograms (default: 1024)'
    )

    # Noise and channel
    channel_group = parser.add_argument_group('Channel Parameters')
    channel_group.add_argument(
        '--snr-min',
        type=float,
        default=0.0,
        help='Minimum SNR in dB (default: 0.0)'
    )
    channel_group.add_argument(
        '--snr-max',
        type=float,
        default=30.0,
        help='Maximum SNR in dB (default: 30.0)'
    )
    channel_group.add_argument(
        '--impairment',
        choices=['perfect', 'cabled', 'wireless'],
        default='wireless',
        help='Channel impairment level (default: wireless)'
    )

    # Multi-signal detection
    detect_group = parser.add_argument_group('Detection Parameters')
    detect_group.add_argument(
        '--min-signals',
        type=int,
        default=1,
        help='Minimum signals per sample (default: 1)'
    )
    detect_group.add_argument(
        '--max-signals',
        type=int,
        default=1,
        help='Maximum signals per sample (default: 1, >1 for detection)'
    )
    detect_group.add_argument(
        '--interference-prob',
        type=float,
        default=0.0,
        help='Co-channel interference probability (default: 0.0)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--no-spectrograms',
        action='store_true',
        help='Do not save spectrograms'
    )
    output_group.add_argument(
        '--no-iq',
        action='store_true',
        help='Do not save IQ data'
    )
    output_group.add_argument(
        '--spectrogram-size',
        type=int,
        default=224,
        help='Spectrogram output size (default: 224)'
    )

    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        '--config',
        type=Path,
        help='Load configuration from JSON file'
    )
    other_group.add_argument(
        '--save-config',
        type=Path,
        help='Save configuration to JSON file and exit'
    )
    other_group.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    other_group.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    other_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args()


def list_modulations():
    """Print all available modulation types."""
    print("\nAvailable Modulation Types")
    print("=" * 50)

    for family, mods in MODULATION_TYPES.items():
        print(f"\n{family.upper()} ({len(mods)} types):")
        # Print in columns
        cols = 4
        for i in range(0, len(mods), cols):
            row = mods[i:i+cols]
            print("  " + "  ".join(f"{m:<15}" for m in row))

    all_mods = get_all_modulations()
    print(f"\nTotal: {len(all_mods)} modulation types available")


def build_config(args) -> TorchSigConfig:
    """Build configuration from command line arguments."""

    # Load from file if specified
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        return TorchSigConfig.from_json(args.config)

    # Use preset if specified
    if args.preset:
        logger.info(f"Using preset: {args.preset}")
        if args.preset == 'classification':
            config = TorchSigConfig.classification_preset(
                name=args.name,
                num_samples=args.samples,
                difficulty=args.difficulty,
            )
        elif args.preset == 'detection':
            config = TorchSigConfig.detection_preset(
                name=args.name,
                num_samples=args.samples,
                max_signals=args.max_signals,
            )
        elif args.preset == 'test':
            config = TorchSigConfig.minimal_test_preset(
                name=args.name,
                num_samples=args.samples,
            )

        # Apply overrides
        if args.modulations:
            config.modulations = args.modulations
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.snr_min != 0.0:
            config.snr_db_min = args.snr_min
        if args.snr_max != 30.0:
            config.snr_db_max = args.snr_max

        return config

    # Build from individual arguments
    impairment_map = {
        'perfect': ImpairmentLevel.PERFECT,
        'cabled': ImpairmentLevel.CABLED,
        'wireless': ImpairmentLevel.WIRELESS,
    }

    config = TorchSigConfig(
        name=args.name,
        output_dir=args.output_dir,
        num_samples=args.samples,
        num_iq_samples=args.iq_samples,
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        modulations=args.modulations,
        modulation_families=args.families,
        impairment_level=impairment_map[args.impairment],
        snr_db_min=args.snr_min,
        snr_db_max=args.snr_max,
        num_signals_min=args.min_signals,
        num_signals_max=args.max_signals,
        cochannel_interference_prob=args.interference_prob,
        random_seed=args.seed,
        num_workers=args.workers,
        save_spectrograms=not args.no_spectrograms,
        save_iq_data=not args.no_iq,
        spectrogram_size=args.spectrogram_size,
    )

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list modulations
    if args.list_modulations:
        list_modulations()
        return 0

    # Build configuration
    try:
        config = build_config(args)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Save config and exit if requested
    if args.save_config:
        config.to_json(args.save_config)
        logger.info(f"Configuration saved to: {args.save_config}")
        print(config.summary())
        return 0

    # Print configuration summary
    print(config.summary())
    print()

    # Create generator and run
    generator = TorchSigGenerator(config, verbose=not args.quiet)

    try:
        dataset_path = generator.generate()
        print(f"\nDataset generated successfully!")
        print(f"Location: {dataset_path}")

        # Validate
        validation = generator.validate_dataset(dataset_path)
        if validation['valid']:
            print("Validation: PASSED")
        else:
            print("Validation: FAILED")
            for error in validation['errors']:
                print(f"  - {error}")

        return 0

    except ImportError as e:
        logger.warning(f"TorchSig not available: {e}")
        logger.info("Attempting fallback generation...")

        try:
            dataset_path = generator.generate_without_torchsig()
            print(f"\nDataset generated (fallback mode)!")
            print(f"Location: {dataset_path}")
            return 0
        except Exception as e2:
            logger.error(f"Fallback generation failed: {e2}")
            return 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
