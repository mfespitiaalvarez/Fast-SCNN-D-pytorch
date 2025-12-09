"""
Create comprehensive plots for paper figures

Generates:
1. Accuracy vs Speed scatter plot
2. Per-class IoU comparison
3. Qualitative comparison grid
4. Ablation study bar chart
5. Alpha trust distribution
6. Model efficiency comparison
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results(json_path):
    """Load evaluation results from JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_speed(results_dict, output_path='figures/accuracy_vs_speed.png'):
    """
    Plot accuracy (mIoU) vs speed (FPS) scatter plot
    results_dict: {model_name: {'mIoU': float, 'fps': float, 'color': str}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, data in results_dict.items():
        ax.scatter(data['fps'], data['mIoU'], 
                  s=200, alpha=0.7, label=model_name, 
                  color=data.get('color', 'blue'),
                  edgecolors='black', linewidth=1.5)
        # Add text annotation
        ax.annotate(model_name, 
                   (data['fps'], data['mIoU']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Inference Speed (FPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_iou(results_fast_scnn, results_fast_scnn_d, 
                       output_path='figures/per_class_iou.png'):
    """
    Plot per-class IoU comparison bar chart
    """
    # Load results
    fast_scnn = load_results(results_fast_scnn)
    fast_scnn_d = load_results(results_fast_scnn_d)
    
    # Get class names and IoU values
    class_names = list(fast_scnn_d['per_class_metrics'].keys())
    iou_fast_scnn = [fast_scnn['per_class_metrics'][c]['iou_percent'] 
                     for c in class_names]
    iou_fast_scnn_d = [fast_scnn_d['per_class_metrics'][c]['iou_percent'] 
                       for c in class_names]
    
    # Calculate improvement
    improvement = np.array(iou_fast_scnn_d) - np.array(iou_fast_scnn)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # Bar chart
    bars1 = ax1.bar(x - width/2, iou_fast_scnn, width, 
                    label='Fast-SCNN', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, iou_fast_scnn_d, width, 
                    label='Fast-SCNN-D', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('IoU (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Per-Class IoU Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Improvement chart
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    bars3 = ax2.bar(x, improvement, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('IoU Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Improvement with Depth', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars3, improvement)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_qualitative_comparison(image_paths, predictions_fast_scnn, predictions_fast_scnn_d,
                               ground_truths, disparities, output_path='figures/qualitative_comparison.png'):
    """
    Create side-by-side qualitative comparison
    image_paths: list of paths to RGB images
    predictions: list of prediction arrays or paths
    """
    num_examples = len(image_paths)
    fig = plt.figure(figsize=(20, 4 * num_examples))
    gs = GridSpec(num_examples, 5, figure=fig, hspace=0.2, wspace=0.1)
    
    for i in range(num_examples):
        # RGB Image
        ax = fig.add_subplot(gs[i, 0])
        img = Image.open(image_paths[i]) if isinstance(image_paths[i], str) else image_paths[i]
        ax.imshow(img)
        ax.set_title('RGB Image', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Disparity
        ax = fig.add_subplot(gs[i, 1])
        disp = disparities[i] if i < len(disparities) else None
        if disp is not None:
            if isinstance(disp, str):
                disp = np.array(Image.open(disp))
            disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
            ax.imshow(disp_vis, cmap='viridis')
        ax.set_title('Disparity', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Fast-SCNN Prediction
        ax = fig.add_subplot(gs[i, 2])
        pred = predictions_fast_scnn[i] if i < len(predictions_fast_scnn) else None
        if pred is not None:
            if isinstance(pred, str):
                pred = Image.open(pred)
            ax.imshow(pred)
        ax.set_title('Fast-SCNN', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Fast-SCNN-D Prediction
        ax = fig.add_subplot(gs[i, 3])
        pred_d = predictions_fast_scnn_d[i] if i < len(predictions_fast_scnn_d) else None
        if pred_d is not None:
            if isinstance(pred_d, str):
                pred_d = Image.open(pred_d)
            ax.imshow(pred_d)
        ax.set_title('Fast-SCNN-D', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Ground Truth
        ax = fig.add_subplot(gs[i, 4])
        gt = ground_truths[i] if i < len(ground_truths) else None
        if gt is not None:
            if isinstance(gt, str):
                gt = Image.open(gt)
            ax.imshow(gt)
        ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('Qualitative Comparison: Fast-SCNN vs Fast-SCNN-D', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_study(ablation_results, output_path='figures/ablation_study.png'):
    """
    Plot ablation study bar chart
    ablation_results: {
        'RGB-only': {'mIoU': float, 'fps': float},
        '+Depth': {'mIoU': float, 'fps': float},
        '+Normals': {'mIoU': float, 'fps': float},
        '+GatedFusion': {'mIoU': float, 'fps': float},
        'Full': {'mIoU': float, 'fps': float}
    }
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(ablation_results.keys())
    mIoU_values = [ablation_results[m]['mIoU'] for m in methods]
    fps_values = [ablation_results[m]['fps'] for m in methods]
    
    x = np.arange(len(methods))
    
    # mIoU plot
    bars1 = ax1.bar(x, mIoU_values, alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean IoU (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Ablation Study: Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, mIoU_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # FPS plot
    bars2 = ax2.bar(x, fps_values, alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Inference Speed (FPS)', fontsize=11, fontweight='bold')
    ax2.set_title('Ablation Study: Speed', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_alpha_distribution(alpha_values, output_path='figures/alpha_distribution.png'):
    """
    Plot histogram of alpha trust values
    alpha_values: numpy array of alpha values [0, 1]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(alpha_values.flatten(), bins=50, alpha=0.7, color='#9b59b6', 
            edgecolor='black', linewidth=1.2)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Balanced (α=0.5)')
    ax1.set_xlabel('Alpha Trust Value (α)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Alpha Trust Values', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative distribution
    sorted_alpha = np.sort(alpha_values.flatten())
    cumulative = np.arange(1, len(sorted_alpha) + 1) / len(sorted_alpha)
    ax2.plot(sorted_alpha, cumulative, linewidth=2, color='#9b59b6')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Balanced (α=0.5)')
    ax2.set_xlabel('Alpha Trust Value (α)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distribution of Alpha', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_efficiency(results_dict, output_path='figures/model_efficiency.png'):
    """
    Plot model size and parameter comparison
    results_dict: {model_name: {'params': float (M), 'size_mb': float, 'fps': float}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(results_dict.keys())
    params = [results_dict[m]['params'] for m in models]
    sizes = [results_dict[m]['size_mb'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Parameters
    bars1 = ax1.bar(x, params, width, alpha=0.8, color='#3498db', 
                   edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Parameters (M)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Model file size
    bars2 = ax2.bar(x, sizes, width, alpha=0.8, color='#e74c3c', 
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
    ax2.set_title('Model File Size', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create paper plots')
    parser.add_argument('--results-fast-scnn', type=str,
                       default='evaluation_results_fast_scnn_citys_val.json',
                       help='Results JSON for Fast-SCNN')
    parser.add_argument('--results-fast-scnn-d', type=str,
                       default='evaluation_results_fast_scnn_d_citys_d_val.json',
                       help='Results JSON for Fast-SCNN-D')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    fast_scnn = load_results(args.results_fast_scnn)
    fast_scnn_d = load_results(args.results_fast_scnn_d)
    
    # 1. Accuracy vs Speed
    accuracy_speed_data = {
        'Fast-SCNN': {
            'mIoU': fast_scnn['metrics']['mean_iou_percent'],
            'fps': fast_scnn['inference_speed']['fps'],
            'color': '#3498db'
        },
        'Fast-SCNN-D': {
            'mIoU': fast_scnn_d['metrics']['mean_iou_percent'],
            'fps': fast_scnn_d['inference_speed']['fps'],
            'color': '#e74c3c'
        }
    }
    plot_accuracy_vs_speed(accuracy_speed_data, 
                          os.path.join(args.output_dir, 'accuracy_vs_speed.png'))
    
    # 2. Per-class IoU
    plot_per_class_iou(args.results_fast_scnn, args.results_fast_scnn_d,
                      os.path.join(args.output_dir, 'per_class_iou.png'))
    
    # 3. Model efficiency
    efficiency_data = {
        'Fast-SCNN': {
            'params': fast_scnn['model_parameters']['total_millions'],
            'size_mb': fast_scnn['model_size_mb']
        },
        'Fast-SCNN-D': {
            'params': fast_scnn_d['model_parameters']['total_millions'],
            'size_mb': fast_scnn_d['model_size_mb']
        }
    }
    plot_model_efficiency(efficiency_data,
                         os.path.join(args.output_dir, 'model_efficiency.png'))
    
    print("\nAll plots generated successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

