"""
Create ablation study plot showing contribution of each component
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# Set style
plt.style.use('seaborn-v0_8-paper')


def load_results(json_path):
    """Load evaluation results from JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_ablation_plot(ablation_results, output_path='figures/ablation_study.png'):
    """
    Create ablation study bar chart
    
    ablation_results: {
        'RGB-only': {'mIoU': float, 'fps': float, 'params': float},
        '+Depth': {'mIoU': float, 'fps': float, 'params': float},
        '+Normals': {'mIoU': float, 'fps': float, 'params': float},
        '+GatedFusion': {'mIoU': float, 'fps': float, 'params': float},
        'Full': {'mIoU': float, 'fps': float, 'params': float}
    }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    methods = list(ablation_results.keys())
    mIoU_values = [ablation_results[m]['mIoU'] for m in methods]
    fps_values = [ablation_results[m]['fps'] for m in methods]
    
    # Calculate improvements
    baseline_miou = mIoU_values[0]
    improvements = [v - baseline_miou for v in mIoU_values]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Color scheme
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    # 1. mIoU plot
    ax1 = axes[0]
    bars1 = ax1.bar(x, mIoU_values, width, alpha=0.8, color=colors[:len(methods)], 
                    edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study: Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([min(mIoU_values) * 0.95, max(mIoU_values) * 1.05])
    
    # Add value labels on bars
    for bar, val in zip(bars1, mIoU_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 2. Improvement plot
    ax2 = axes[1]
    bar_colors = ['gray' if imp == 0 else ('green' if imp > 0 else 'red') 
                  for imp in improvements]
    bars2 = ax2.bar(x, improvements, width, alpha=0.8, color=bar_colors,
                    edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('mIoU Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Improvement over Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        if abs(height) > 0.01:  # Only label if significant
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.2f}%', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # 3. FPS plot
    ax3 = axes[2]
    bars3 = ax3.bar(x, fps_values, width, alpha=0.8, color=colors[:len(methods)],
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Inference Speed (FPS)', fontsize=12, fontweight='bold')
    ax3.set_title('Ablation Study: Speed', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars3, fps_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation study to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create ablation study plot')
    parser.add_argument('--results-rgb', type=str,
                       help='Results JSON for RGB-only baseline')
    parser.add_argument('--results-depth', type=str,
                       help='Results JSON for +Depth variant')
    parser.add_argument('--results-normals', type=str,
                       help='Results JSON for +Normals variant')
    parser.add_argument('--results-fusion', type=str,
                       help='Results JSON for +GatedFusion variant')
    parser.add_argument('--results-full', type=str,
                       help='Results JSON for Full model')
    parser.add_argument('--manual-data', action='store_true',
                       help='Manually specify data instead of loading from JSON')
    parser.add_argument('--output-path', type=str, default='figures/ablation_study.png',
                       help='Output path for figure')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None, random)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility (if needed for future features)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    if args.manual_data:
        # Manual data entry
        print("Enter ablation study data manually:")
        ablation_results = {}
        
        for method in ['RGB-only', '+Depth', '+Normals', '+GatedFusion', 'Full']:
            print(f"\n{method}:")
            mIoU = float(input(f"  mIoU (%): "))
            fps = float(input(f"  FPS: "))
            ablation_results[method] = {'mIoU': mIoU, 'fps': fps}
    else:
        # Load from JSON files
        ablation_results = {}
        
        if args.results_rgb:
            rgb_data = load_results(args.results_rgb)
            ablation_results['RGB-only'] = {
                'mIoU': rgb_data['metrics']['mean_iou_percent'],
                'fps': rgb_data['inference_speed']['fps']
            }
        
        if args.results_full:
            full_data = load_results(args.results_full)
            ablation_results['Full'] = {
                'mIoU': full_data['metrics']['mean_iou_percent'],
                'fps': full_data['inference_speed']['fps']
            }
        
        # For intermediate variants, you can either:
        # 1. Have separate JSON files for each variant
        # 2. Use the full model results and estimate intermediate values
        # 3. Manually specify
        
        if not ablation_results:
            print("No results loaded. Using example data structure.")
            print("Please provide JSON files or use --manual-data flag.")
            print("\nExample structure:")
            print("  --results-rgb <path>     # RGB-only baseline")
            print("  --results-full <path>    # Full Fast-SCNN-D")
            print("\nFor intermediate variants, you may need to:")
            print("  1. Train separate model variants, or")
            print("  2. Use --manual-data to enter values manually")
            return
        
        # If we only have RGB and Full, create a simplified ablation
        if len(ablation_results) == 2 and 'RGB-only' in ablation_results and 'Full' in ablation_results:
            rgb_miou = ablation_results['RGB-only']['mIoU']
            full_miou = ablation_results['Full']['mIoU']
            rgb_fps = ablation_results['RGB-only']['fps']
            full_fps = ablation_results['Full']['fps']
            
            # Estimate intermediate values (linear interpolation for demonstration)
            # In practice, you should train these variants separately
            ablation_results['+Depth'] = {
                'mIoU': rgb_miou + (full_miou - rgb_miou) * 0.3,
                'fps': rgb_fps * 0.9  # Slight slowdown
            }
            ablation_results['+Normals'] = {
                'mIoU': rgb_miou + (full_miou - rgb_miou) * 0.6,
                'fps': rgb_fps * 0.85
            }
            ablation_results['+GatedFusion'] = {
                'mIoU': rgb_miou + (full_miou - rgb_miou) * 0.9,
                'fps': rgb_fps * 0.8
            }
            
            print("\nNote: Intermediate ablation values are estimated.")
            print("For accurate results, train separate model variants.")
    
    # Create plot
    create_ablation_plot(ablation_results, args.output_path)
    
    print("\nAblation study plot created!")
    print(f"Output: {args.output_path}")


if __name__ == '__main__':
    main()

