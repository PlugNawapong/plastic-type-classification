"""
Configurable ML Pipeline for Hyperspectral Plastic Classification

This script accepts command-line arguments to override configuration parameters.

Usage Examples:

# Run with default config.py settings
python run_pipeline_config.py --mode full

# Custom normalization percentiles
python run_pipeline_config.py --mode normalize --lower-percentile 1 --upper-percentile 99

# Custom preprocessing
python run_pipeline_config.py --mode train --spectral-binning 4 --denoise

# Custom training parameters
python run_pipeline_config.py --mode train --epochs 100 --lr 0.0001 --batch-size 256

# Wavelength filtering
python run_pipeline_config.py --mode train --wavelength-range 450 700

# Select specific number of bands
python run_pipeline_config.py --mode train --select-bands 50

# Spatial downsampling
python run_pipeline_config.py --mode train --spatial-binning 2

# Custom model
python run_pipeline_config.py --mode train --model-type cnn --dropout 0.5

# Full pipeline with custom settings
python run_pipeline_config.py --mode full --epochs 100 --batch-size 1024 --spectral-binning 2 --denoise
"""

import os
# Fix OpenMP library conflict on macOS (must be set before importing torch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path
import config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Plastic Classification Pipeline with Configurable Parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode
    parser.add_argument(
        '--mode',
        choices=['full', 'normalize', 'train', 'inference'],
        default='full',
        help='Pipeline mode to run (default: full)'
    )

    # Data paths
    parser.add_argument('--training-data', type=str, default=config.TRAINING_DATA_FOLDER,
                        help=f'Training data folder (default: {config.TRAINING_DATA_FOLDER})')
    parser.add_argument('--inference-data', type=str, default=config.INFERENCE_DATA_FOLDER,
                        help=f'Inference data folder (default: {config.INFERENCE_DATA_FOLDER})')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Output directory (default: {config.OUTPUT_DIR})')

    # Normalization
    norm_group = parser.add_argument_group('Normalization Parameters')
    norm_group.add_argument('--skip-normalize', action='store_true',
                            help='Skip normalization (use if data is already normalized)')
    norm_group.add_argument('--lower-percentile', type=float, default=config.NORMALIZATION['lower_percentile'],
                            help=f'Lower percentile for normalization (default: {config.NORMALIZATION["lower_percentile"]})')
    norm_group.add_argument('--upper-percentile', type=float, default=config.NORMALIZATION['upper_percentile'],
                            help=f'Upper percentile for normalization (default: {config.NORMALIZATION["upper_percentile"]})')

    # Preprocessing
    preproc_group = parser.add_argument_group('Preprocessing Parameters')
    preproc_group.add_argument('--wavelength-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                               help='Wavelength range to filter (e.g., --wavelength-range 450 700)')
    preproc_group.add_argument('--select-bands', type=int, metavar='N',
                               help='Select N evenly-spaced bands')
    preproc_group.add_argument('--spectral-binning', type=int, default=config.PREPROCESSING['spectral_binning'],
                               help=f'Spectral binning size (default: {config.PREPROCESSING["spectral_binning"]})')
    preproc_group.add_argument('--spatial-binning', type=int,
                               help='Spatial binning size (e.g., 2 for 2x2 blocks)')
    preproc_group.add_argument('--denoise', action='store_true',
                               help='Enable denoising')
    preproc_group.add_argument('--denoise-method', choices=['gaussian', 'median'],
                               default=config.PREPROCESSING['denoise_method'],
                               help=f'Denoising method (default: {config.PREPROCESSING["denoise_method"]})')
    preproc_group.add_argument('--denoise-strength', type=float, default=config.PREPROCESSING['denoise_strength'],
                               help=f'Denoising strength (default: {config.PREPROCESSING["denoise_strength"]})')

    # Dataset
    data_group = parser.add_argument_group('Dataset Parameters')
    data_group.add_argument('--no-augment', action='store_true',
                            help='Disable data augmentation')
    data_group.add_argument('--val-ratio', type=float, default=config.DATASET['val_ratio'],
                            help=f'Validation set ratio (default: {config.DATASET["val_ratio"]})')
    data_group.add_argument('--batch-size', type=int, default=config.DATASET['batch_size'],
                            help=f'Batch size (default: {config.DATASET["batch_size"]})')
    data_group.add_argument('--num-workers', type=int, default=config.DATASET['num_workers'],
                            help=f'Number of data loading workers (default: {config.DATASET["num_workers"]})')

    # Model
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model-type',
                             choices=['cnn', 'resnet', 'deep', 'inception', 'lstm', 'transformer'],
                             default=config.MODEL['model_type'],
                             help=f'Model architecture (default: {config.MODEL["model_type"]})')
    model_group.add_argument('--dropout', type=float, default=config.MODEL['dropout_rate'],
                             help=f'Dropout rate (default: {config.MODEL["dropout_rate"]})')

    # Training
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=config.TRAINING['num_epochs'],
                             help=f'Number of training epochs (default: {config.TRAINING["num_epochs"]})')
    train_group.add_argument('--lr', type=float, default=config.TRAINING['learning_rate'],
                             help=f'Learning rate (default: {config.TRAINING["learning_rate"]})')
    train_group.add_argument('--weight-decay', type=float, default=config.TRAINING['weight_decay'],
                             help=f'Weight decay (default: {config.TRAINING["weight_decay"]})')
    train_group.add_argument('--no-class-weights', action='store_true',
                             help='Disable class weighting for imbalanced data')

    # Inference
    infer_group = parser.add_argument_group('Inference Parameters')
    infer_group.add_argument('--checkpoint', type=str, default=config.INFERENCE['checkpoint_path'],
                             help=f'Path to model checkpoint (default: {config.INFERENCE["checkpoint_path"]})')
    infer_group.add_argument('--no-prob-maps', action='store_true',
                             help='Disable saving probability maps')

    # Other
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip dependency and data folder checks')

    return parser.parse_args()


def build_config_from_args(args):
    """Build configuration dictionary from command-line arguments."""

    # Preprocessing config
    preprocess_config = {
        'wavelength_range': tuple(args.wavelength_range) if args.wavelength_range else None,
        'select_n_bands': args.select_bands,
        'spectral_binning': args.spectral_binning if not args.select_bands else None,
        'spatial_binning': args.spatial_binning,
        'denoise_enabled': args.denoise,
        'denoise_method': args.denoise_method,
        'denoise_strength': args.denoise_strength,
    }

    # Full config
    full_config = {
        # Paths
        'training_data_folder': args.training_data,
        'inference_data_folder': args.inference_data,
        'normalized_training_folder': config.NORMALIZED_TRAINING_FOLDER,
        'normalized_inference_folder': config.NORMALIZED_INFERENCE_FOLDER,
        'ground_truth_folder': config.GROUND_TRUTH_FOLDER,
        'output_dir': args.output_dir,

        # Normalization
        'normalization': {
            'lower_percentile': args.lower_percentile,
            'upper_percentile': args.upper_percentile,
        },

        # Preprocessing
        'preprocessing': preprocess_config,

        # Dataset
        'dataset': {
            'augment': not args.no_augment,
            'val_ratio': args.val_ratio,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        },

        # Model
        'model': {
            'model_type': args.model_type,
            'dropout_rate': args.dropout,
        },

        # Training
        'training': {
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'use_class_weights': not args.no_class_weights,
        },

        # Inference
        'inference': {
            'checkpoint_path': args.checkpoint,
            'batch_size': args.batch_size,
            'save_probability_maps': not args.no_prob_maps,
        },
    }

    return full_config


def print_config_summary(cfg):
    """Print configuration summary."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)

    print("\n[Data Paths]")
    print(f"  Training data: {cfg['training_data_folder']}")
    print(f"  Inference data: {cfg['inference_data_folder']}")
    print(f"  Output directory: {cfg['output_dir']}")

    print("\n[Normalization]")
    print(f"  Percentile range: {cfg['normalization']['lower_percentile']}% - {cfg['normalization']['upper_percentile']}%")

    print("\n[Preprocessing]")
    prep = cfg['preprocessing']
    print(f"  Wavelength range: {prep['wavelength_range']}")
    print(f"  Select N bands: {prep['select_n_bands']}")
    print(f"  Spectral binning: {prep['spectral_binning']}")
    print(f"  Spatial binning: {prep['spatial_binning']}")
    print(f"  Denoising: {prep['denoise_enabled']} ({prep['denoise_method']}, strength={prep['denoise_strength']})")

    print("\n[Dataset]")
    ds = cfg['dataset']
    print(f"  Augmentation: {ds['augment']}")
    print(f"  Validation ratio: {ds['val_ratio']}")
    print(f"  Batch size: {ds['batch_size']}")

    print("\n[Model]")
    print(f"  Architecture: {cfg['model']['model_type']}")
    print(f"  Dropout rate: {cfg['model']['dropout_rate']}")

    print("\n[Training]")
    tr = cfg['training']
    print(f"  Epochs: {tr['num_epochs']}")
    print(f"  Learning rate: {tr['learning_rate']}")
    print(f"  Weight decay: {tr['weight_decay']}")
    print(f"  Class weights: {tr['use_class_weights']}")

    print("\n" + "="*60 + "\n")


def main():
    args = parse_arguments()
    cfg = build_config_from_args(args)

    print("="*60)
    print("HYPERSPECTRAL PLASTIC CLASSIFICATION PIPELINE")
    print("="*60)

    print_config_summary(cfg)

    # Import pipeline modules
    from pipeline_normalize import normalize_dataset
    from pipeline_train import main as train_main
    from pipeline_inference import main as inference_main
    import json

    # Save configuration
    config_path = Path(cfg['output_dir']) / 'pipeline_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Check dependencies
    if not args.skip_checks:
        print("\nChecking dependencies...")
        required_packages = ['numpy', 'torch', 'PIL', 'scipy', 'tqdm', 'matplotlib']
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            sys.exit(1)
        print("✓ All dependencies installed")

    # Execute pipeline based on mode
    if args.mode in ['full', 'normalize']:
        if args.skip_normalize:
            print("\n" + "="*60)
            print("SKIPPING NORMALIZATION (--skip-normalize flag set)")
            print("="*60)
            print("Using existing normalized data folders")
        else:
            print("\n" + "="*60)
            print("STEP 1: NORMALIZING TRAINING DATA")
            print("="*60)
            normalize_dataset(
                input_folder=cfg['training_data_folder'],
                output_folder=cfg['normalized_training_folder'],
                lower_percentile=cfg['normalization']['lower_percentile'],
                upper_percentile=cfg['normalization']['upper_percentile']
            )

            print("\n" + "="*60)
            print("STEP 2: NORMALIZING INFERENCE DATA")
            print("="*60)
            normalize_dataset(
                input_folder=cfg['inference_data_folder'],
                output_folder=cfg['normalized_inference_folder'],
                lower_percentile=cfg['normalization']['lower_percentile'],
                upper_percentile=cfg['normalization']['upper_percentile']
            )

    if args.mode in ['full', 'train']:
        # Determine which data folder to use
        if args.skip_normalize:
            # Use original data (already normalized)
            training_data_folder = cfg['training_data_folder']
            print(f"\n✓ Using original data (already normalized): {training_data_folder}")
        else:
            # Use normalized data folder
            training_data_folder = cfg['normalized_training_folder']

        # Check if training data exists
        if not Path(training_data_folder).exists():
            print(f"\n❌ Error: Training data not found!")
            print(f"   Expected folder: {training_data_folder}")
            if args.skip_normalize:
                print("   The original data folder doesn't exist.")
            else:
                print("   Run normalization first or use --skip-normalize if data is already normalized.")
            sys.exit(1)

        print("\n" + "="*60)
        print("STEP 3: TRAINING MODEL")
        print("="*60)

        # Create temporary training config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            temp_config_path = f.name

        # Run training with config
        import os
        os.environ['PIPELINE_CONFIG'] = temp_config_path

        # Import and patch training script
        import pipeline_train

        # Override training config
        pipeline_train.CONFIG = {
            'data_folder': training_data_folder,
            'label_path': f"{cfg['ground_truth_folder']}/labels.png",
            'preprocess': cfg['preprocessing'],
            'augment': cfg['dataset']['augment'],
            'val_ratio': cfg['dataset']['val_ratio'],
            'batch_size': cfg['dataset']['batch_size'],
            'num_workers': cfg['dataset']['num_workers'],
            'model_type': cfg['model']['model_type'],
            'dropout_rate': cfg['model']['dropout_rate'],
            'num_epochs': cfg['training']['num_epochs'],
            'learning_rate': cfg['training']['learning_rate'],
            'weight_decay': cfg['training']['weight_decay'],
            'use_class_weights': cfg['training']['use_class_weights'],
            'output_dir': f"{cfg['output_dir']}/training",
        }

        pipeline_train.main()

        # Cleanup
        os.unlink(temp_config_path)

    if args.mode in ['full', 'inference']:
        print("\n" + "="*60)
        print("STEP 4: RUNNING INFERENCE")
        print("="*60)

        # Override inference config
        import pipeline_inference

        pipeline_inference.CONFIG = {
            'checkpoint_path': cfg['inference']['checkpoint_path'],
            'inference_folder': cfg['normalized_inference_folder'],
            'preprocess': cfg['preprocessing'],
            'output_dir': f"{cfg['output_dir']}/inference",
            'save_probability_maps': cfg['inference']['save_probability_maps'],
            'batch_size': cfg['inference']['batch_size'],
        }

        pipeline_inference.main()

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {cfg['output_dir']}/")


if __name__ == '__main__':
    main()
