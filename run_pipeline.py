"""
Complete ML Pipeline for Hyperspectral Plastic Classification

This script orchestrates the entire pipeline:
1. Normalize raw training data
2. Preprocess normalized data (optional)
3. Train 1D CNN model
4. Normalize inference data
5. Perform inference

Usage:
    python run_pipeline.py --mode full              # Run complete pipeline
    python run_pipeline.py --mode normalize         # Only normalize data
    python run_pipeline.py --mode train             # Only train model
    python run_pipeline.py --mode inference         # Only inference
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """
    Run a shell command and handle errors.

    Args:
        command: Command to run (as string or list)
        description: Description of the command
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")

    if isinstance(command, str):
        command = command.split()

    result = subprocess.run(command, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")


def normalize_training_data():
    """Step 1: Normalize training dataset."""
    run_command(
        "python pipeline_normalize.py",
        "Step 1: Normalizing training data"
    )


def normalize_inference_data():
    """Step 1b: Normalize inference dataset."""
    # Create a temporary script or modify pipeline_normalize.py
    # For now, we'll import and run programmatically
    print(f"\n{'='*60}")
    print("Step 1b: Normalizing inference data")
    print(f"{'='*60}\n")

    from pipeline_normalize import normalize_dataset

    normalize_dataset(
        input_folder='Inference_dataset1',
        output_folder='Inference_dataset1_normalized',
        lower_percentile=2,
        upper_percentile=98
    )

    print("\n✓ Step 1b: Normalizing inference data completed successfully")


def train_model():
    """Step 2: Train the model."""
    run_command(
        "python pipeline_train.py",
        "Step 2: Training model"
    )


def run_inference():
    """Step 3: Run inference."""
    run_command(
        "python pipeline_inference.py",
        "Step 3: Running inference"
    )


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        'numpy',
        'torch',
        'PIL',
        'scipy',
        'tqdm',
        'matplotlib'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    print("✓ All dependencies installed\n")


def check_data_folders():
    """Check if required data folders exist."""
    print("Checking data folders...")

    required_folders = [
        'training_dataset',
        'Ground_Truth',
        'Inference_dataset1'
    ]

    missing = []
    for folder in required_folders:
        if not Path(folder).exists():
            missing.append(folder)

    if missing:
        print(f"\n❌ Missing required folders: {', '.join(missing)}")
        sys.exit(1)

    print("✓ All required folders exist\n")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperspectral Plastic Classification Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'normalize', 'train', 'inference'],
        default='full',
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and data checks'
    )

    args = parser.parse_args()

    print("="*60)
    print("HYPERSPECTRAL PLASTIC CLASSIFICATION PIPELINE")
    print("="*60)

    # Checks
    if not args.skip_checks:
        check_dependencies()
        check_data_folders()

    # Run pipeline based on mode
    if args.mode == 'full':
        print("\nRunning FULL pipeline (normalize -> train -> inference)\n")

        # Step 1: Normalize training data
        normalize_training_data()

        # Step 1b: Normalize inference data
        normalize_inference_data()

        # Step 2: Train model
        train_model()

        # Step 3: Run inference
        run_inference()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutputs:")
        print("  - Normalized training data: training_dataset_normalized/")
        print("  - Normalized inference data: Inference_dataset1_normalized/")
        print("  - Trained model: output/training/best_model.pth")
        print("  - Training plots: output/training/training_history.png")
        print("  - Inference results: output/inference/predictions.png")
        print("  - Inference statistics: output/inference/inference_statistics.json")
        print("="*60)

    elif args.mode == 'normalize':
        print("\nRunning NORMALIZE mode\n")
        normalize_training_data()
        normalize_inference_data()

        print("\n" + "="*60)
        print("NORMALIZATION COMPLETED!")
        print("="*60)

    elif args.mode == 'train':
        print("\nRunning TRAIN mode\n")

        # Check if normalized data exists
        if not Path('training_dataset_normalized').exists():
            print("⚠ Warning: Normalized training data not found")
            print("  Running normalization first...")
            normalize_training_data()

        train_model()

        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)

    elif args.mode == 'inference':
        print("\nRunning INFERENCE mode\n")

        # Check if trained model exists
        if not Path('output/training/best_model.pth').exists():
            print("❌ Error: Trained model not found at output/training/best_model.pth")
            print("  Please run training first with: python run_pipeline.py --mode train")
            sys.exit(1)

        # Check if normalized inference data exists
        if not Path('Inference_dataset1_normalized').exists():
            print("⚠ Warning: Normalized inference data not found")
            print("  Running normalization first...")
            normalize_inference_data()

        run_inference()

        print("\n" + "="*60)
        print("INFERENCE COMPLETED!")
        print("="*60)


if __name__ == '__main__':
    main()
