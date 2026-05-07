"""
Visualization utilities for training analysis.

Generates plots for:
- Training/validation loss curves
- Metric curves (accuracy, F1, AUROC, etc.)
- Learning rate schedule
- Confusion matrix
- ROC and PR curves (placeholder)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_loss_curves(
    history: Dict,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_curves(
    history: Dict,
    metrics: List[str] = ['agent_accuracy', 'agent_f1', 'failure_accuracy', 'failure_f1'],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot training and validation metric curves.

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (uses dual-task keys: agent_*/failure_*)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if not history.get('train_metrics'):
        return

    # Only keep metrics that are actually present in the history entries
    available = [m for m in metrics if m in history['train_metrics'][0]]
    if not available:
        return

    n_metrics = len(available)
    n_cols = min(n_metrics, 4)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten() if n_metrics > 1 else [axes]

    epochs = range(1, len(history['train_metrics']) + 1)

    for idx, metric in enumerate(available):
        ax = axes_flat[idx]
        train_values = [m[metric] for m in history['train_metrics']]
        val_values   = [m[metric] for m in history['val_metrics']]

        ax.plot(epochs, train_values, 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_values,   'r-', label='Val',   linewidth=2)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metric curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(
    history: Dict,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot learning rate schedule.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['learning_rates']) + 1)
    
    ax.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning rate plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = ['Failure', 'Success'],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        class_names: Names of classes
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize by row (actual class)
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
    
    # Plot
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion', fontsize=11)
    
    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.2%})'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if cm_normalized[i, j] > 0.5 else 'black',
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_metrics_summary(
    history: Dict,
    test_metrics: Dict,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Create a comprehensive summary figure with all key plots.

    Args:
        history: Training history dictionary (uses dual-task keys: agent_*/failure_*)
        test_metrics: Test set metrics
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if not history.get('train_loss'):
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val Loss',   linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('LR Schedule', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Agent F1
    ax3 = fig.add_subplot(gs[1, 0])
    if history.get('train_metrics') and 'agent_f1' in history['train_metrics'][0]:
        train_f1 = [m['agent_f1'] for m in history['train_metrics']]
        val_f1   = [m['agent_f1'] for m in history['val_metrics']]
        ax3.plot(epochs, train_f1, 'b-', label='Train', linewidth=2)
        ax3.plot(epochs, val_f1,   'r-', label='Val',   linewidth=2)
        if 'agent_f1' in test_metrics:
            ax3.axhline(test_metrics['agent_f1'], color='green', linestyle='--', label='Test', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1')
    ax3.set_title('Agent F1', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Failure F1
    ax4 = fig.add_subplot(gs[1, 1])
    if history.get('train_metrics') and 'failure_f1' in history['train_metrics'][0]:
        train_ff1 = [m['failure_f1'] for m in history['train_metrics']]
        val_ff1   = [m['failure_f1'] for m in history['val_metrics']]
        ax4.plot(epochs, train_ff1, 'b-', label='Train', linewidth=2)
        ax4.plot(epochs, val_ff1,   'r-', label='Val',   linewidth=2)
        if 'failure_f1' in test_metrics:
            ax4.axhline(test_metrics['failure_f1'], color='green', linestyle='--', label='Test', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1')
    ax4.set_title('Violation F1', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Agent Accuracy
    ax5 = fig.add_subplot(gs[1, 2])
    if history.get('train_metrics') and 'agent_accuracy' in history['train_metrics'][0]:
        train_acc = [m['agent_accuracy'] for m in history['train_metrics']]
        val_acc   = [m['agent_accuracy'] for m in history['val_metrics']]
        ax5.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
        ax5.plot(epochs, val_acc,   'r-', label='Val',   linewidth=2)
        if 'agent_accuracy' in test_metrics:
            ax5.axhline(test_metrics['agent_accuracy'], color='green', linestyle='--', label='Test', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Agent Accuracy', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Test metrics bar chart
    ax6 = fig.add_subplot(gs[2, :])
    metric_names = ['Agent Acc', 'Agent F1', 'Viol Acc', 'Viol F1']
    metric_values = [
        test_metrics.get('agent_accuracy',   0),
        test_metrics.get('agent_f1',         0),
        test_metrics.get('failure_accuracy', 0),
        test_metrics.get('failure_f1',       0),
    ]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_names)))
    bars = ax6.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax6.set_ylabel('Score')
    ax6.set_title('Final Test Set Metrics', fontweight='bold', fontsize=12)
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Training Summary', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_plots(
    history: Dict,
    test_metrics: Dict,
    confusion_matrix: np.ndarray,
    output_dir: Path,
    format: str = 'png'
):
    """
    Generate all training analysis plots.
    
    Args:
        history: Training history dictionary
        test_metrics: Test set metrics
        confusion_matrix: Confusion matrix
        output_dir: Directory to save plots
        format: File format ('png', 'pdf', 'svg')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating plots...")
    
    # Loss curves
    plot_loss_curves(
        history,
        save_path=output_dir / f'loss_curves.{format}'
    )
    
    # Metric curves
    plot_metric_curves(
        history,
        save_path=output_dir / f'metric_curves.{format}'
    )
    
    # Learning rate
    plot_learning_rate(
        history,
        save_path=output_dir / f'learning_rate.{format}'
    )
    
    # Confusion matrix
    if confusion_matrix is not None:
        plot_confusion_matrix(
            confusion_matrix,
            save_path=output_dir / f'confusion_matrix.{format}'
        )
    
    # Summary
    plot_all_metrics_summary(
        history,
        test_metrics,
        save_path=output_dir / f'training_summary.{format}'
    )
    
    logger.info(f"All plots saved to {output_dir}")


# Placeholder for ROC and PR curves
def plot_roc_curve():
    """Placeholder for ROC curve plotting."""
    # TODO: Implement ROC curve with sklearn.metrics.roc_curve
    pass


def plot_pr_curve():
    """Placeholder for Precision-Recall curve plotting."""
    # TODO: Implement PR curve with sklearn.metrics.precision_recall_curve
    pass
