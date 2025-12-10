"""
Create side-by-side architecture comparison diagram:
Fast-SCNN (original) vs Fast-SCNN-D (ours)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

def create_architecture_comparison(output_path='figures/architecture_comparison.png'):
    """
    Create side-by-side architecture comparison diagram
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('Architecture Comparison: Fast-SCNN vs Fast-SCNN-D', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define colors
    color_rgb = '#4A90E2'  # Blue for RGB
    color_depth = '#E24A4A'  # Red for Depth
    color_fusion = '#50C878'  # Green for Fusion
    color_backbone = '#9B59B6'  # Purple for Backbone
    color_output = '#F39C12'  # Orange for Output
    
    # Box style
    box_style = dict(boxstyle="round,pad=0.5", facecolor='white', 
                     edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # ========== LEFT: Fast-SCNN (Original) ==========
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Fast-SCNN (Original)', fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # Input
    input_box = FancyBboxPatch((3.5, 10.5), 3, 0.8, 
                               boxstyle="round,pad=0.3", 
                               facecolor=color_rgb, edgecolor='black', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(5, 10.9, 'RGB Input\n[3, H, W]', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Learning to Downsample
    lds_box = FancyBboxPatch((3.5, 8.5), 3, 0.8, 
                             boxstyle="round,pad=0.3", 
                             facecolor=color_backbone, edgecolor='black', linewidth=2)
    ax1.add_patch(lds_box)
    ax1.text(5, 8.9, 'Learning to\nDownsample', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Global Feature Extractor
    gfe_box = FancyBboxPatch((3.5, 6.5), 3, 0.8, 
                             boxstyle="round,pad=0.3", 
                             facecolor=color_backbone, edgecolor='black', linewidth=2)
    ax1.add_patch(gfe_box)
    ax1.text(5, 6.9, 'Global Feature\nExtractor', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Feature Fusion
    ff_box = FancyBboxPatch((3.5, 4.5), 3, 0.8, 
                            boxstyle="round,pad=0.3", 
                            facecolor=color_fusion, edgecolor='black', linewidth=2)
    ax1.add_patch(ff_box)
    ax1.text(5, 4.9, 'Feature Fusion\n(Skip Connection)', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Classifier
    cls_box = FancyBboxPatch((3.5, 2.5), 3, 0.8, 
                             boxstyle="round,pad=0.3", 
                             facecolor=color_output, edgecolor='black', linewidth=2)
    ax1.add_patch(cls_box)
    ax1.text(5, 2.9, 'Classifier', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Output
    output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, 
                                boxstyle="round,pad=0.3", 
                                facecolor=color_output, edgecolor='black', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 0.9, 'Segmentation\nOutput', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    ax1.annotate('', xy=(5, 10.5), xytext=(5, 9.3), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 8.5), xytext=(5, 7.3), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 6.5), xytext=(5, 5.3), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 4.5), xytext=(5, 3.3), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 2.5), xytext=(5, 1.3), arrowprops=arrow_props)
    
    # Skip connection (dashed)
    skip_arrow = FancyArrowPatch((4.5, 8.9), (4.5, 5.1), 
                                  arrowstyle='->', lw=2, linestyle='--', 
                                  color='gray', alpha=0.7)
    ax1.add_patch(skip_arrow)
    
    # ========== RIGHT: Fast-SCNN-D (Ours) ==========
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('Fast-SCNN-D (Ours)', fontsize=14, fontweight='bold', pad=15)
    ax2.axis('off')
    
    # Input (RGB + Disparity)
    input_rgb_box = FancyBboxPatch((1.5, 10.5), 2.5, 0.8, 
                                   boxstyle="round,pad=0.3", 
                                   facecolor=color_rgb, edgecolor='black', linewidth=2)
    ax2.add_patch(input_rgb_box)
    ax2.text(2.75, 10.9, 'RGB Input\n[3, H, W]', ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    
    input_depth_box = FancyBboxPatch((6, 10.5), 2.5, 0.8, 
                                     boxstyle="round,pad=0.3", 
                                     facecolor=color_depth, edgecolor='black', linewidth=2)
    ax2.add_patch(input_depth_box)
    ax2.text(7.25, 10.9, 'Disparity\n[1, H, W]', ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    
    # Geometry Feature Generator
    geo_box = FancyBboxPatch((6, 8.5), 2.5, 0.8, 
                              boxstyle="round,pad=0.3", 
                              facecolor=color_depth, edgecolor='black', linewidth=2)
    ax2.add_patch(geo_box)
    ax2.text(7.25, 8.9, 'Geometry Feature\nGenerator\n(Pseudo-HHA)', ha='center', va='center', 
             fontsize=8, fontweight='bold', color='white')
    
    # Dual-Stream LDS
    # RGB Stream
    rgb_stream_box = FancyBboxPatch((1.5, 8.5), 2.5, 0.8, 
                                    boxstyle="round,pad=0.3", 
                                    facecolor=color_rgb, edgecolor='black', linewidth=2)
    ax2.add_patch(rgb_stream_box)
    ax2.text(2.75, 8.9, 'RGB Stream\nLDS', ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    
    # Depth Stream
    depth_stream_box = FancyBboxPatch((6, 6.5), 2.5, 0.8, 
                                      boxstyle="round,pad=0.3", 
                                      facecolor=color_depth, edgecolor='black', linewidth=2)
    ax2.add_patch(depth_stream_box)
    ax2.text(7.25, 6.9, 'Depth Stream\nLDS', ha='center', va='center', 
             fontsize=9, fontweight='bold', color='white')
    
    # Gated Fusion
    fusion_box = FancyBboxPatch((3.5, 6.5), 3, 0.8, 
                                boxstyle="round,pad=0.3", 
                                facecolor=color_fusion, edgecolor='black', linewidth=2)
    ax2.add_patch(fusion_box)
    ax2.text(5, 6.9, 'Adaptive Gated\nFusion', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Global Feature Extractor
    gfe_box = FancyBboxPatch((3.5, 4.5), 3, 0.8, 
                             boxstyle="round,pad=0.3", 
                             facecolor=color_backbone, edgecolor='black', linewidth=2)
    ax2.add_patch(gfe_box)
    ax2.text(5, 4.9, 'Global Feature\nExtractor', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Feature Fusion
    ff_box = FancyBboxPatch((3.5, 2.5), 3, 0.8, 
                            boxstyle="round,pad=0.3", 
                            facecolor=color_fusion, edgecolor='black', linewidth=2)
    ax2.add_patch(ff_box)
    ax2.text(5, 2.9, 'Feature Fusion\n(Skip Connection)', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Classifier
    cls_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, 
                             boxstyle="round,pad=0.3", 
                             facecolor=color_output, edgecolor='black', linewidth=2)
    ax2.add_patch(cls_box)
    ax2.text(5, 0.9, 'Classifier', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')
    
    # Arrows - Input to streams
    ax2.annotate('', xy=(2.75, 10.5), xytext=(2.75, 9.3), arrowprops=arrow_props)
    ax2.annotate('', xy=(7.25, 10.5), xytext=(7.25, 9.3), arrowprops=arrow_props)
    
    # Arrow - Geometry generator to depth stream
    ax2.annotate('', xy=(7.25, 8.5), xytext=(7.25, 7.3), arrowprops=arrow_props)
    
    # Arrows - Streams to fusion
    ax2.annotate('', xy=(4, 8.9), xytext=(4.5, 7.3), arrowprops=arrow_props)
    ax2.annotate('', xy=(6, 6.9), xytext=(5.5, 7.3), arrowprops=arrow_props)
    
    # Arrows - Fusion to rest
    ax2.annotate('', xy=(5, 6.5), xytext=(5, 5.3), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 4.5), xytext=(5, 3.3), arrowprops=arrow_props)
    ax2.annotate('', xy=(5, 2.5), xytext=(5, 1.3), arrowprops=arrow_props)
    
    # Skip connection (dashed)
    skip_arrow = FancyArrowPatch((4.5, 6.9), (4.5, 3.1), 
                                  arrowstyle='->', lw=2, linestyle='--', 
                                  color='gray', alpha=0.7)
    ax2.add_patch(skip_arrow)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=color_rgb, edgecolor='black', label='RGB Processing'),
        mpatches.Patch(facecolor=color_depth, edgecolor='black', label='Depth Processing'),
        mpatches.Patch(facecolor=color_fusion, edgecolor='black', label='Fusion'),
        mpatches.Patch(facecolor=color_backbone, edgecolor='black', label='Backbone'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture comparison saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create architecture comparison diagram')
    parser.add_argument('--output', type=str, default='figures/architecture_comparison.png',
                       help='Output path for the figure')
    args = parser.parse_args()
    
    create_architecture_comparison(args.output)

