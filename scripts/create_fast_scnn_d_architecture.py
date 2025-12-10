"""
Create detailed architecture diagram for Fast-SCNN-D
Matching the original Fast-SCNN horizontal flow style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.patches import FancyBboxPatch as RoundedRect
import numpy as np
import os

def create_fast_scnn_d_architecture(output_path='figures/fast_scnn_d_architecture.png'):
    """
    Create detailed architecture diagram for Fast-SCNN-D
    Horizontal flow from left to right, matching original Fast-SCNN style
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors matching original Fast-SCNN paper
    color_conv2d = '#E74C3C'      # Red
    color_dwconv = '#95A5A6'      # Gray  
    color_dsconv = '#3498DB'      # Blue
    color_bottleneck = '#2ECC71'  # Green
    color_pyramid = '#9B59B6'     # Purple
    color_upsample = '#F39C12'    # Yellow/Orange
    color_softmax = '#E74C3C'     # Red (same as Conv2D)
    color_fusion = '#1ABC9C'      # Teal for fusion
    color_geometry = '#E67E22'    # Orange for geometry
    
    # ========== LEGEND ==========
    legend_y = 7.2
    legend_items = [
        ('Input', 'lightblue'),
        ('Conv2D', color_conv2d),
        ('DWConv', color_dwconv),
        ('DSConv', color_dsconv),
        ('Bottleneck', color_bottleneck),
        ('Pyramid Pooling', color_pyramid),
        ('Upsample', color_upsample),
        ('Fusion', color_fusion),
        ('Geometry', color_geometry),
    ]
    
    x_pos = 0.5
    for label, color in legend_items:
        rect = Rectangle((x_pos, legend_y), 0.25, 0.25, 
                        facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + 0.35, legend_y + 0.125, label, 
                ha='left', va='center', fontsize=8, fontweight='bold')
        x_pos += 1.8 if len(label) < 10 else 2.2
    
    # ========== STAGE 1: Learning to Down-sample (Dual-Stream) ==========
    stage1_x = 1
    stage1_y_center = 4
    
    # Stage label
    ax.text(stage1_x + 2, 6.5, 'Learning to Down-sample', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(stage1_x + 2, 6.2, '(Dual-Stream)', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Input: RGB + Disparity (shown as two inputs)
    input_rgb = FancyBboxPatch((stage1_x, stage1_y_center + 0.8), 0.8, 0.6,
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_rgb)
    ax.text(stage1_x + 0.4, stage1_y_center + 1.1, 'RGB', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    input_disp = FancyBboxPatch((stage1_x, stage1_y_center - 0.4), 0.8, 0.6,
                               boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(input_disp)
    ax.text(stage1_x + 0.4, stage1_y_center - 0.1, 'Disp', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # RGB Stream
    rgb_x = stage1_x + 1.2
    rgb_conv = FancyBboxPatch((rgb_x, stage1_y_center + 0.5), 0.7, 0.5,
                             boxstyle="round,pad=0.05", 
                             facecolor=color_conv2d, edgecolor='black', linewidth=1.5)
    ax.add_patch(rgb_conv)
    ax.text(rgb_x + 0.35, stage1_y_center + 0.75, 'Conv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    rgb_dsconv1 = FancyBboxPatch((rgb_x + 0.9, stage1_y_center + 0.3), 0.8, 0.7,
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(rgb_dsconv1)
    ax.text(rgb_x + 1.3, stage1_y_center + 0.65, 'DSConv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    rgb_dsconv2 = FancyBboxPatch((rgb_x + 1.9, stage1_y_center + 0.4), 0.6, 0.5,
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(rgb_dsconv2)
    ax.text(rgb_x + 2.2, stage1_y_center + 0.65, 'DSConv', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Geometry Feature Generator
    geo_x = stage1_x + 1.2
    geo_box = FancyBboxPatch((geo_x, stage1_y_center - 0.7), 0.7, 0.5,
                             boxstyle="round,pad=0.05", 
                             facecolor=color_geometry, edgecolor='black', linewidth=1.5)
    ax.add_patch(geo_box)
    ax.text(geo_x + 0.35, stage1_y_center - 0.45, 'Geo', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Depth Stream
    depth_conv = FancyBboxPatch((geo_x + 0.9, stage1_y_center - 0.7), 0.7, 0.5,
                                boxstyle="round,pad=0.05", 
                                facecolor=color_conv2d, edgecolor='black', linewidth=1.5)
    ax.add_patch(depth_conv)
    ax.text(geo_x + 1.25, stage1_y_center - 0.45, 'Conv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    depth_dsconv1 = FancyBboxPatch((geo_x + 1.8, stage1_y_center - 0.9), 0.8, 0.7,
                                   boxstyle="round,pad=0.05", 
                                   facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(depth_dsconv1)
    ax.text(geo_x + 2.2, stage1_y_center - 0.55, 'DSConv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    depth_dsconv2 = FancyBboxPatch((geo_x + 2.8, stage1_y_center - 0.8), 0.6, 0.5,
                                   boxstyle="round,pad=0.05", 
                                   facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(depth_dsconv2)
    ax.text(geo_x + 3.1, stage1_y_center - 0.55, 'DSConv', 
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Gated Fusion
    fusion_x = rgb_x + 3.0
    fusion_box = FancyBboxPatch((fusion_x, stage1_y_center - 0.2), 0.8, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor=color_fusion, edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(fusion_x + 0.4, stage1_y_center + 0.2, 'Gated', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax.text(fusion_x + 0.4, stage1_y_center - 0.1, 'Fusion', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Arrows for Stage 1
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to streams
    ax.annotate('', xy=(rgb_x, stage1_y_center + 0.75), 
                xytext=(stage1_x + 0.8, stage1_y_center + 1.1), arrowprops=arrow_props)
    ax.annotate('', xy=(geo_x, stage1_y_center - 0.45), 
                xytext=(stage1_x + 0.8, stage1_y_center - 0.1), arrowprops=arrow_props)
    
    # RGB stream
    ax.annotate('', xy=(rgb_x + 0.9, stage1_y_center + 0.65), 
                xytext=(rgb_x + 0.7, stage1_y_center + 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(rgb_x + 1.9, stage1_y_center + 0.65), 
                xytext=(rgb_x + 1.7, stage1_y_center + 0.65), arrowprops=arrow_props)
    
    # Depth stream
    ax.annotate('', xy=(geo_x + 0.9, stage1_y_center - 0.45), 
                xytext=(geo_x + 0.7, stage1_y_center - 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(geo_x + 1.8, stage1_y_center - 0.55), 
                xytext=(geo_x + 1.6, stage1_y_center - 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(geo_x + 2.8, stage1_y_center - 0.55), 
                xytext=(geo_x + 2.6, stage1_y_center - 0.55), arrowprops=arrow_props)
    
    # Streams to fusion
    ax.annotate('', xy=(fusion_x, stage1_y_center + 0.2), 
                xytext=(rgb_x + 2.5, stage1_y_center + 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(fusion_x, stage1_y_center - 0.1), 
                xytext=(geo_x + 3.4, stage1_y_center - 0.55), arrowprops=arrow_props)
    
    # Split point annotation
    split_x = fusion_x + 0.8
    ax.plot([split_x, split_x], [stage1_y_center - 0.6, stage1_y_center + 0.4], 
            'k--', linewidth=1.5, alpha=0.5)
    
    # ========== STAGE 2: Global Feature Extractor ==========
    stage2_x = split_x + 0.5
    stage2_y_center = 4
    
    ax.text(stage2_x + 3, 6.5, 'Global Feature Extractor', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Bottleneck blocks (9 total: 3+3+3)
    bottleneck_width = 0.5
    bottleneck_height = 0.6
    bottleneck_spacing = 0.6
    
    for i in range(9):
        bx = stage2_x + i * bottleneck_spacing
        bottleneck = FancyBboxPatch((bx, stage2_y_center - 0.3), 
                                    bottleneck_width, bottleneck_height,
                                    boxstyle="round,pad=0.03", 
                                    facecolor=color_bottleneck, 
                                    edgecolor='black', linewidth=1.2)
        ax.add_patch(bottleneck)
        # Make them slightly smaller as they progress
        if i < 3:
            ax.text(bx + 0.25, stage2_y_center, 'B', 
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        elif i < 6:
            ax.text(bx + 0.25, stage2_y_center, 'B', 
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        else:
            ax.text(bx + 0.25, stage2_y_center, 'B', 
                    ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Pyramid Pooling
    pyramid_x = stage2_x + 9 * bottleneck_spacing
    pyramid_box = FancyBboxPatch((pyramid_x, stage2_y_center - 0.3), 1.0, 0.6,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=color_pyramid, edgecolor='black', linewidth=2)
    ax.add_patch(pyramid_box)
    ax.text(pyramid_x + 0.5, stage2_y_center, 'Pyramid\nPooling', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Arrows for Stage 2
    ax.annotate('', xy=(stage2_x, stage2_y_center), 
                xytext=(split_x, stage1_y_center + 0.2), arrowprops=arrow_props)
    
    for i in range(8):
        x1 = stage2_x + (i + 1) * bottleneck_spacing
        x2 = stage2_x + i * bottleneck_spacing + bottleneck_width
        ax.annotate('', xy=(x1, stage2_y_center), 
                    xytext=(x2, stage2_y_center), 
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    ax.annotate('', xy=(pyramid_x, stage2_y_center), 
                xytext=(stage2_x + 9 * bottleneck_spacing, stage2_y_center), 
                arrowprops=arrow_props)
    
    # ========== STAGE 3: Feature Fusion ==========
    stage3_x = pyramid_x + 1.2
    stage3_y_center = 4
    
    ax.text(stage3_x + 1.5, 6.5, 'Feature Fusion', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Top path: from Global Feature Extractor
    top_path_y = stage3_y_center + 0.5
    upsample_top = FancyBboxPatch((stage3_x, top_path_y), 0.7, 0.4,
                                  boxstyle="round,pad=0.05", 
                                  facecolor=color_upsample, edgecolor='black', linewidth=1.5)
    ax.add_patch(upsample_top)
    ax.text(stage3_x + 0.35, top_path_y + 0.2, 'Up', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    dwconv_top = FancyBboxPatch((stage3_x + 0.9, top_path_y), 0.7, 0.4,
                               boxstyle="round,pad=0.05", 
                               facecolor=color_dwconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(dwconv_top)
    ax.text(stage3_x + 1.25, top_path_y + 0.2, 'DW', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    conv_top = FancyBboxPatch((stage3_x + 1.8, top_path_y), 0.7, 0.4,
                             boxstyle="round,pad=0.05", 
                             facecolor=color_conv2d, edgecolor='black', linewidth=1.5)
    ax.add_patch(conv_top)
    ax.text(stage3_x + 2.15, top_path_y + 0.2, 'Conv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Bottom path: skip connection from Stage 1
    bottom_path_y = stage3_y_center - 0.5
    conv_bottom = FancyBboxPatch((stage3_x + 1.8, bottom_path_y), 0.7, 0.4,
                                boxstyle="round,pad=0.05", 
                                facecolor=color_conv2d, edgecolor='black', linewidth=1.5)
    ax.add_patch(conv_bottom)
    ax.text(stage3_x + 2.15, bottom_path_y + 0.2, 'Conv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Fusion (addition)
    fusion_circle_x = stage3_x + 2.7
    fusion_circle = Circle((fusion_circle_x, stage3_y_center), 0.25,
                          facecolor=color_fusion, edgecolor='black', linewidth=2)
    ax.add_patch(fusion_circle)
    ax.text(fusion_circle_x, stage3_y_center, '+', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows for Stage 3
    # From Global Feature Extractor to top path
    ax.annotate('', xy=(stage3_x, top_path_y + 0.2), 
                xytext=(pyramid_x + 1.0, stage2_y_center), arrowprops=arrow_props)
    
    # Top path arrows
    ax.annotate('', xy=(stage3_x + 0.9, top_path_y + 0.2), 
                xytext=(stage3_x + 0.7, top_path_y + 0.2), arrowprops=arrow_props)
    ax.annotate('', xy=(stage3_x + 1.8, top_path_y + 0.2), 
                xytext=(stage3_x + 1.6, top_path_y + 0.2), arrowprops=arrow_props)
    
    # Skip connection (dashed) from Stage 1
    skip_arrow = FancyArrowPatch((fusion_x + 0.8, stage1_y_center + 0.2), 
                                (stage3_x + 1.8, bottom_path_y + 0.2),
                                arrowstyle='->', lw=2, linestyle='--', 
                                color='gray', alpha=0.7, connectionstyle="arc3,rad=0.2")
    ax.add_patch(skip_arrow)
    
    # Paths to fusion
    ax.annotate('', xy=(fusion_circle_x - 0.25, top_path_y + 0.2), 
                xytext=(stage3_x + 2.5, top_path_y + 0.2), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(fusion_circle_x - 0.25, bottom_path_y + 0.2), 
                xytext=(stage3_x + 2.5, bottom_path_y + 0.2), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ========== STAGE 4: Classifier ==========
    stage4_x = fusion_circle_x + 0.6
    stage4_y_center = 4
    
    ax.text(stage4_x + 2, 6.5, 'Classifier', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    dsconv1_cls = FancyBboxPatch((stage4_x, stage4_y_center - 0.2), 0.7, 0.4,
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(dsconv1_cls)
    ax.text(stage4_x + 0.35, stage4_y_center, 'DSConv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    dsconv2_cls = FancyBboxPatch((stage4_x + 0.9, stage4_y_center - 0.2), 0.7, 0.4,
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color_dsconv, edgecolor='black', linewidth=1.5)
    ax.add_patch(dsconv2_cls)
    ax.text(stage4_x + 1.25, stage4_y_center, 'DSConv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    conv_cls = FancyBboxPatch((stage4_x + 1.8, stage4_y_center - 0.2), 0.7, 0.4,
                             boxstyle="round,pad=0.05", 
                             facecolor=color_conv2d, edgecolor='black', linewidth=1.5)
    ax.add_patch(conv_cls)
    ax.text(stage4_x + 2.15, stage4_y_center, 'Conv', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    upsample_cls = FancyBboxPatch((stage4_x + 2.7, stage4_y_center - 0.2), 0.7, 0.4,
                                  boxstyle="round,pad=0.05", 
                                  facecolor=color_upsample, edgecolor='black', linewidth=1.5)
    ax.add_patch(upsample_cls)
    ax.text(stage4_x + 3.05, stage4_y_center, 'Up', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Softmax
    softmax_x = stage4_x + 3.6
    softmax_box = FancyBboxPatch((softmax_x, stage4_y_center - 0.3), 0.8, 0.6,
                                boxstyle="round,pad=0.1", 
                                facecolor=color_softmax, edgecolor='black', linewidth=2)
    ax.add_patch(softmax_box)
    ax.text(softmax_x + 0.4, stage4_y_center, 'Softmax', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Output
    output_x = softmax_x + 1.1
    output_box = FancyBboxPatch((output_x, stage4_y_center - 0.4), 1.0, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(output_x + 0.5, stage4_y_center, 'Output', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for Stage 4
    ax.annotate('', xy=(stage4_x, stage4_y_center), 
                xytext=(fusion_circle_x + 0.25, stage3_y_center), arrowprops=arrow_props)
    
    ax.annotate('', xy=(stage4_x + 0.9, stage4_y_center), 
                xytext=(stage4_x + 0.7, stage4_y_center), arrowprops=arrow_props)
    ax.annotate('', xy=(stage4_x + 1.8, stage4_y_center), 
                xytext=(stage4_x + 1.6, stage4_y_center), arrowprops=arrow_props)
    ax.annotate('', xy=(stage4_x + 2.7, stage4_y_center), 
                xytext=(stage4_x + 2.5, stage4_y_center), arrowprops=arrow_props)
    ax.annotate('', xy=(softmax_x, stage4_y_center), 
                xytext=(stage4_x + 3.4, stage4_y_center), arrowprops=arrow_props)
    ax.annotate('', xy=(output_x, stage4_y_center), 
                xytext=(softmax_x + 0.8, stage4_y_center), arrowprops=arrow_props)
    
    # Title
    ax.text(12, 7.8, 'Fast-SCNN-D Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fast-SCNN-D architecture diagram saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create Fast-SCNN-D architecture diagram')
    parser.add_argument('--output', type=str, default='figures/fast_scnn_d_architecture.png',
                       help='Output path for the figure')
    args = parser.parse_args()
    
    create_fast_scnn_d_architecture(args.output)
