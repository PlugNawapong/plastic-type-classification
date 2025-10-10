#!/usr/bin/env python3
"""
Model Comparison Script
Compare performance of different models on the same dataset
Usage: python compare_models.py --data DATA_FOLDER --models MODEL1.pth MODEL2.pth ...
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube
from pipeline_dataset import CLASS_NAMES, CLASS_MAPPING
from pipeline_model import create_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model checkpoints'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to inference data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/model_comparison',
        help='Output directory'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size'
    )
    
    return parser.parse_args()


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_and_predict(checkpoint_path, cube, device, batch_size):
    """Load model and perform prediction."""
    print(f"\nProcessing: {Path(checkpoint_path).name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model_name = config.get('model_type', 'unknown')
    val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"  Model type: {model_name}")
    print(f"  Training accuracy: {val_acc:.2f}%")
    
    # Create model
    num_bands = cube.shape[2]
    model = create_model(
        num_bands=num_bands,
        num_classes=len(CLASS_NAMES),
        model_type=model_name,
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Predict
    h, w, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    num_pixels = pixels.shape[0]
    
    predictions = np.zeros(num_pixels, dtype=np.int32)
    confidences = np.zeros(num_pixels, dtype=np.float32)
    
    print("  Running inference...")
    with torch.no_grad():
        for i in range(0, num_pixels, batch_size):
            batch_end = min(i + batch_size, num_pixels)
            batch = pixels[i:batch_end]
            
            batch_tensor = torch.FloatTensor(batch).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            conf, pred = probs.max(dim=1)
            predictions[i:batch_end] = pred.cpu().numpy()
            confidences[i:batch_end] = conf.cpu().numpy()
    
    prediction_map = predictions.reshape(h, w)
    confidence_map = confidences.reshape(h, w)
    
    return {
        'name': model_name,
        'checkpoint': Path(checkpoint_path).name,
        'train_acc': val_acc,
        'prediction_map': prediction_map,
        'confidence_map': confidence_map,
    }


def compare_predictions(results):
    """Compare prediction statistics."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = []
    
    for result in results:
        pred_map = result['prediction_map']
        conf_map = result['confidence_map']
        
        # Statistics
        mean_conf = conf_map.mean()
        std_conf = conf_map.std()
        high_conf = (conf_map >= 0.7).sum() / conf_map.size * 100
        
        # Class distribution
        unique, counts = np.unique(pred_map, return_counts=True)
        num_classes_predicted = len(unique)
        
        comparison.append({
            'name': result['name'],
            'checkpoint': result['checkpoint'],
            'train_acc': result['train_acc'],
            'mean_confidence': mean_conf,
            'std_confidence': std_conf,
            'high_confidence_pct': high_conf,
            'classes_predicted': num_classes_predicted,
        })
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Checkpoint':<30} {'Train Acc':<10} {'Mean Conf':<10} {'High Conf %':<12} {'Classes':<8}")
    print("-" * 100)
    
    for comp in comparison:
        print(f"{comp['name']:<20} "
              f"{comp['checkpoint']:<30} "
              f"{comp['train_acc']:>8.2f}% "
              f"{comp['mean_confidence']:>9.4f} "
              f"{comp['high_confidence_pct']:>10.1f}% "
              f"{comp['classes_predicted']:>7}")
    
    return comparison


def plot_comparison(results, output_dir):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_models = len(results)
    
    # 1. Prediction maps
    fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 10))
    if num_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, result in enumerate(results):
        # Prediction map
        ax = axes[0, i]
        im = ax.imshow(result['prediction_map'], cmap='tab10', vmin=0, vmax=10)
        ax.set_title(f"{result['name']}\n(Train: {result['train_acc']:.1f}%)", 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Confidence map
        ax = axes[1, i]
        im = ax.imshow(result['confidence_map'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Confidence\n(Mean: {result['confidence_map'].mean():.3f})", 
                     fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'predictions_comparison.png'}")
    plt.close()
    
    # 2. Confidence distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for result in results:
        conf_flat = result['confidence_map'].flatten()
        ax.hist(conf_flat, bins=50, alpha=0.5, label=result['name'], density=True)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Confidence Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'confidence_distributions.png'}")
    plt.close()
    
    # 3. Class distributions
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    if num_models == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        unique, counts = np.unique(result['prediction_map'], return_counts=True)
        labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"C{c}" for c in unique]
        
        axes[i].bar(range(len(unique)), counts)
        axes[i].set_xticks(range(len(unique)))
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].set_title(result['name'], fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Pixel Count')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'class_distributions.png'}")
    plt.close()
    
    # 4. Summary statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    names = [r['name'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    mean_confs = [r['confidence_map'].mean() for r in results]
    high_conf_pcts = [(r['confidence_map'] >= 0.7).sum() / r['confidence_map'].size * 100 
                      for r in results]
    
    axes[0].bar(names, train_accs, color='skyblue')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training Accuracy', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(names, mean_confs, color='lightgreen')
    axes[1].set_ylabel('Mean Confidence')
    axes[1].set_title('Average Confidence', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(names, high_conf_pcts, color='coral')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title('High Confidence Predictions (≥0.7)', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'summary_statistics.png'}")
    plt.close()


def save_comparison_report(comparison, output_dir):
    """Save comparison report as JSON."""
    output_dir = Path(output_dir)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'models': comparison,
    }
    
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved: {output_dir / 'comparison_report.json'}")


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("MODEL COMPARISON TOOL")
    print("="*80)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\nLoading data: {args.data}")
    cube, wavelengths, header = load_hyperspectral_cube(args.data)
    print(f"  Cube shape: {cube.shape}")
    
    # Process each model
    results = []
    for checkpoint_path in args.models:
        if not Path(checkpoint_path).exists():
            print(f"\n⚠ Skipping {checkpoint_path} (not found)")
            continue
        
        result = load_and_predict(checkpoint_path, cube, device, args.batch_size)
        results.append(result)
    
    if not results:
        print("\n❌ No valid models found!")
        return
    
    # Compare predictions
    comparison = compare_predictions(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_comparison(results, args.output)
    
    # Save report
    save_comparison_report(comparison, args.output)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETED")
    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
