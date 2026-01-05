"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Training Metrics Visualization Script                      ║
║              Generate Accuracy and Loss Graphs for Model Training            ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script generates training and validation accuracy/loss curves for the
CNN-LSTM action recognition model.

Usage:
    python generate_plots.py                    # Use sample data
    python generate_plots.py --history path.json  # Use actual training history

Author: Deep Learning Assignment
Version: 1.0.0
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Activity class labels
CLASS_LABELS = [
    "basketball", "biking", "diving", "golf_swing", "horse_riding",
    "soccer_juggling", "swing", "tennis_swing", "trampoline_jumping",
    "volleyball_spiking", "walking"
]


def generate_sample_history() -> Dict[str, List[float]]:
    """
    Generate realistic sample training history for demonstration.
    
    This simulates a typical training scenario with:
    - Improving accuracy over epochs
    - Decreasing loss over epochs
    - Some overfitting (validation metrics plateau)
    
    Returns:
        Dictionary with training history metrics
    """
    epochs = 50
    
    # Simulate training accuracy (starts at ~0.2, improves to ~0.92)
    train_acc = []
    for i in range(epochs):
        base = 0.2 + (0.72 * (1 - np.exp(-i / 10)))
        noise = np.random.normal(0, 0.02)
        train_acc.append(min(0.98, max(0.15, base + noise)))
    
    # Simulate validation accuracy (starts at ~0.2, improves to ~0.85, plateaus)
    val_acc = []
    for i in range(epochs):
        if i < 30:
            base = 0.2 + (0.65 * (1 - np.exp(-i / 12)))
        else:
            base = 0.85 + np.random.normal(0, 0.015)
        noise = np.random.normal(0, 0.025)
        val_acc.append(min(0.92, max(0.15, base + noise)))
    
    # Simulate training loss (starts at ~2.5, decreases to ~0.15)
    train_loss = []
    for i in range(epochs):
        base = 2.5 * np.exp(-i / 8) + 0.15
        noise = np.random.normal(0, 0.05)
        train_loss.append(max(0.1, base + noise))
    
    # Simulate validation loss (starts at ~2.5, decreases to ~0.40)
    val_loss = []
    for i in range(epochs):
        if i < 30:
            base = 2.5 * np.exp(-i / 10) + 0.40
        else:
            base = 0.45 + np.random.normal(0, 0.03)
        noise = np.random.normal(0, 0.08)
        val_loss.append(max(0.2, base + noise))
    
    return {
        'accuracy': train_acc,
        'val_accuracy': val_acc,
        'loss': train_loss,
        'val_loss': val_loss
    }


def load_history_from_file(filepath: str) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing training history
        
    Returns:
        Dictionary with training history metrics
        
    Expected JSON format:
    {
        "accuracy": [...],
        "val_accuracy": [...],
        "loss": [...],
        "val_loss": [...]
    }
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing required key '{key}' in history file")
    
    return history


def plot_training_metrics(history: Dict[str, List[float]], output_path: str = None):
    """
    Generate and save training/validation accuracy and loss plots.
    
    Args:
        history: Dictionary containing training metrics
        output_path: Path to save the plot (default: plots/accuracy_loss.png)
    """
    if output_path is None:
        output_path = os.path.join(PLOTS_DIR, 'accuracy_loss.png')
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Create figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('CNN-LSTM Action Recognition Model - Training Metrics', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ───────────────────────────────────────────────────────────────────────────
    # ACCURACY PLOT
    # ───────────────────────────────────────────────────────────────────────────
    ax1.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add final accuracy values as text
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    ax1.text(0.02, 0.98, 
             f'Final Training: {final_train_acc:.3f}\nFinal Validation: {final_val_acc:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ───────────────────────────────────────────────────────────────────────────
    # LOSS PLOT
    # ───────────────────────────────────────────────────────────────────────────
    ax2.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add final loss values as text
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    ax2.text(0.02, 0.98, 
             f'Final Training: {final_train_loss:.3f}\nFinal Validation: {final_val_loss:.3f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ───────────────────────────────────────────────────────────────────────────
    # SAVE PLOT
    # ───────────────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def plot_training_metrics_stacked(history: Dict[str, List[float]], output_path: str = None):
    """
    Generate and save training/validation accuracy and loss plots (stacked vertically).
    
    Args:
        history: Dictionary containing training metrics
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = os.path.join(PLOTS_DIR, 'accuracy_loss_stacked.png')
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Create figure with 2 subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('CNN-LSTM Action Recognition Model - Training Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ───────────────────────────────────────────────────────────────────────────
    # ACCURACY PLOT
    # ───────────────────────────────────────────────────────────────────────────
    ax1.plot(epochs, history['accuracy'], 'b-', linewidth=2.5, label='Training Accuracy', marker='o', markersize=3, markevery=5)
    ax1.plot(epochs, history['val_accuracy'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=3, markevery=5)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='semibold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Add statistics
    max_val_acc = max(history['val_accuracy'])
    max_val_epoch = history['val_accuracy'].index(max_val_acc) + 1
    ax1.axhline(y=max_val_acc, color='green', linestyle=':', alpha=0.5)
    ax1.text(0.02, 0.98, 
             f'Best Validation: {max_val_acc:.3f} (Epoch {max_val_epoch})\n'
             f'Final Training: {history["accuracy"][-1]:.3f}\n'
             f'Final Validation: {history["val_accuracy"][-1]:.3f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
    
    # ───────────────────────────────────────────────────────────────────────────
    # LOSS PLOT
    # ───────────────────────────────────────────────────────────────────────────
    ax2.plot(epochs, history['loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=3, markevery=5)
    ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=3, markevery=5)
    ax2.set_title('Model Loss Over Epochs', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='semibold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Add statistics
    min_val_loss = min(history['val_loss'])
    min_val_epoch = history['val_loss'].index(min_val_loss) + 1
    ax2.axhline(y=min_val_loss, color='green', linestyle=':', alpha=0.5)
    ax2.text(0.02, 0.98, 
             f'Best Validation: {min_val_loss:.3f} (Epoch {min_val_epoch})\n'
             f'Final Training: {history["loss"][-1]:.3f}\n'
             f'Final Validation: {history["val_loss"][-1]:.3f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='darkred'))
    
    # ───────────────────────────────────────────────────────────────────────────
    # SAVE PLOT
    # ───────────────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def generate_sample_confusion_matrix(num_classes: int = 11) -> np.ndarray:
    """
    Generate a realistic sample confusion matrix for demonstration.
    
    Simulates classification results with:
    - High diagonal values (correct predictions)
    - Some confusion between similar activities
    - Lower off-diagonal values (misclassifications)
    
    Args:
        num_classes: Number of activity classes
        
    Returns:
        Confusion matrix as numpy array (num_classes x num_classes)
    """
    # Initialize with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Generate realistic confusion matrix
    # Average of 50-100 samples per class
    for i in range(num_classes):
        total_samples = np.random.randint(50, 101)
        
        # Correct predictions (70-90% accuracy per class)
        correct = int(total_samples * np.random.uniform(0.70, 0.92))
        cm[i, i] = correct
        
        # Distribute remaining samples as misclassifications
        remaining = total_samples - correct
        if remaining > 0:
            # Create confusion with similar classes (weighted)
            weights = np.ones(num_classes)
            weights[i] = 0  # Can't confuse with itself
            
            # Add higher confusion for adjacent classes
            if i > 0:
                weights[i-1] *= 2
            if i < num_classes - 1:
                weights[i+1] *= 2
            
            weights = weights / weights.sum()
            
            # Distribute misclassifications
            misclass = np.random.multinomial(remaining, weights)
            cm[i, :] += misclass
    
    return cm


def plot_confusion_matrix(cm: np.ndarray = None, class_labels: List[str] = None,
                         output_path: str = None, normalize: bool = False):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix as numpy array (if None, generates sample data)
        class_labels: List of class label names
        output_path: Path to save the plot (default: plots/confusion_matrix.png)
        normalize: Whether to normalize the confusion matrix
    """
    if cm is None:
        cm = generate_sample_confusion_matrix()
    
    if class_labels is None:
        class_labels = CLASS_LABELS
    
    if output_path is None:
        output_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = ' (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title_suffix = ''
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Value' if normalize else 'Count'},
                ax=ax, square=True, linewidths=0.5, linecolor='gray')
    
    # Customize plot
    ax.set_title(f'CNN-LSTM Action Recognition - Confusion Matrix{title_suffix}\n' +
                f'Overall Accuracy: {accuracy:.2%}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='semibold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='semibold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add grid for better readability
    ax.set_facecolor('white')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()
    
    return accuracy


def plot_confusion_matrix_detailed(cm: np.ndarray = None, class_labels: List[str] = None,
                                  output_path: str = None):
    """
    Generate detailed confusion matrix with both raw counts and percentages.
    
    Args:
        cm: Confusion matrix as numpy array (if None, generates sample data)
        class_labels: List of class label names
        output_path: Path to save the plot
    """
    if cm is None:
        cm = generate_sample_confusion_matrix()
    
    if class_labels is None:
        class_labels = CLASS_LABELS
    
    if output_path is None:
        output_path = os.path.join(PLOTS_DIR, 'confusion_matrix_detailed.png')
    
    # Calculate normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'CNN-LSTM Action Recognition - Confusion Matrix Analysis\n' +
                f'Overall Accuracy: {accuracy:.2%}',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'}, ax=ax1, square=True,
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Raw Counts', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('True Label', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='semibold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Percentage'}, ax=ax2, square=True,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
    ax2.set_title('Normalized (Row Percentages)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('True Label', fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='semibold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Detailed confusion matrix saved to: {output_path}")
    plt.close()
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, label in enumerate(class_labels):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"  • {label:20s}: {class_acc:.2%} ({cm[i, i]}/{cm[i, :].sum()})")
    
    return accuracy


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate training accuracy and loss plots for action recognition model'
    )
    parser.add_argument(
        '--history',
        type=str,
        default=None,
        help='Path to training history JSON file (if not provided, uses sample data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: plots/accuracy_loss.png)'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['side-by-side', 'stacked'],
        default='both',
        help='Plot layout: side-by-side, stacked, or both (default: both)'
    )
    parser.add_argument(
        '--confusion-matrix',
        type=str,
        default=None,
        help='Path to confusion matrix numpy file (.npy) or generate sample'
    )
    parser.add_argument(
        '--include-cm',
        action='store_true',
        help='Include confusion matrix generation (default: True if no specific plots requested)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" Training Metrics Visualization Generator".center(80))
    print("=" * 80)
    print()
    
    # Load or generate history
    if args.history:
        print(f"Loading training history from: {args.history}")
        history = load_history_from_file(args.history)
        print(f"✓ Loaded {len(history['accuracy'])} epochs of training data")
    else:
        print("No history file provided, generating sample training data...")
        history = generate_sample_history()
        print(f"✓ Generated {len(history['accuracy'])} epochs of sample data")
    
    print()
    print("Training Summary:")
    print(f"  • Epochs: {len(history['accuracy'])}")
    print(f"  • Final Training Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"  • Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  • Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
    print(f"  • Final Training Loss: {history['loss'][-1]:.4f}")
    print(f"  • Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"  • Best Validation Loss: {min(history['val_loss']):.4f}")
    print()
    
    # Generate plots
    print("Generating plots...")
    
    if args.layout in ['side-by-side', 'both']:
        output_path = args.output if args.output else os.path.join(PLOTS_DIR, 'accuracy_loss.png')
        plot_training_metrics(history, output_path)
    
    if args.layout in ['stacked', 'both']:
        output_path_stacked = os.path.join(PLOTS_DIR, 'accuracy_loss_stacked.png')
        plot_training_metrics_stacked(history, output_path_stacked)
    
    # Generate confusion matrix
    print()
    print("Generating confusion matrix...")
    
    if args.confusion_matrix:
        print(f"Loading confusion matrix from: {args.confusion_matrix}")
        cm = np.load(args.confusion_matrix)
        print(f"✓ Loaded confusion matrix of shape {cm.shape}")
    else:
        print("No confusion matrix file provided, generating sample data...")
        cm = generate_sample_confusion_matrix()
        print(f"✓ Generated sample confusion matrix")
    
    # Plot confusion matrices
    accuracy = plot_confusion_matrix(cm, normalize=False)
    print(f"  Overall Accuracy: {accuracy:.2%}")
    
    plot_confusion_matrix(cm, normalize=True, 
                         output_path=os.path.join(PLOTS_DIR, 'confusion_matrix_normalized.png'))
    
    accuracy_detailed = plot_confusion_matrix_detailed(cm)
    
    print()
    print("=" * 80)
    print(" ✓ Visualization Complete!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
