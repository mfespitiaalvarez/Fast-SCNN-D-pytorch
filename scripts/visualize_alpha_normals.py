"""
Visualization script for Alpha Trust and Surface Normals

Creates figures showing:
1. Examples with low/high alpha trust values with feature space ablation
2. Surface normals visualization
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2
from torchvision import transforms
import torch.utils.data as data

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_segmentation_dataset
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from utils.visualize import get_color_pallete
from train import parse_args


class HookManager:
    """Manages forward hooks to extract intermediate features"""
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def register_hook(self, name, module):
        def hook_fn(module, input, output):
            self.features[name] = output.detach()
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}


def visualize_alpha_trust(model, dataloader, device, output_dir='figures', num_examples=4):
    """
    Visualize examples with low and high alpha trust values
    Shows RGB, disparity, alpha map, predictions, and feature space ablation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    hook_manager = HookManager()
    
    # Register hooks to extract alpha and features
    # Hook the GatedFusion module to get alpha
    fusion_module = model.learning_to_downsample.fusion
    
    # Store original forward
    original_forward = fusion_module.forward
    
    # Create wrapper to capture alpha
    alpha_values = []
    def forward_with_alpha(x_rgb, x_depth):
        cat_feats = torch.cat([x_rgb, x_depth], dim=1)
        alpha = fusion_module.gate_conv(cat_feats)
        alpha_values.append(alpha.detach().cpu())
        return (x_rgb * alpha) + (x_depth * (1 - alpha))
    
    fusion_module.forward = forward_with_alpha
    
    # Also hook the dual streams to get RGB and depth features
    hook_manager.register_hook('rgb_stream', model.learning_to_downsample.rgb_dsconv2)
    hook_manager.register_hook('depth_stream', model.learning_to_downsample.depth_dsconv2)
    
    examples_collected = []
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataloader):
            if len(examples_collected) >= num_examples * 2:  # Get more to select best examples
                break
            
            image = image.to(device)
            original_size = image.shape[2:]
            
            # Clear previous alpha
            alpha_values.clear()
            
            # Forward pass
            outputs = model(image)
            pred = torch.argmax(outputs[0], 1)
            
            # Get alpha from captured values
            if len(alpha_values) > 0:
                alpha = alpha_values[0].numpy().squeeze()  # [H, W] (already sigmoid applied)
            else:
                continue
            
            # Get RGB and depth features
            rgb_feat = hook_manager.features.get('rgb_stream', None)
            depth_feat = hook_manager.features.get('depth_stream', None)
            
            # Resize alpha to original image size
            alpha_tensor = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
            alpha_resized = F.interpolate(
                alpha_tensor, size=original_size, mode='bilinear', align_corners=True
            ).squeeze().numpy()
            
            # Get mean alpha value
            mean_alpha = alpha_resized.mean()
            
            # Get original RGB image (denormalize)
            rgb_img = image[0, :3].cpu()
            rgb_img = rgb_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_img = torch.clamp(rgb_img, 0, 1)
            rgb_img = rgb_img.permute(1, 2, 0).numpy()
            
            # Get disparity (4th channel)
            disp = image[0, 3].cpu().numpy()
            
            # Get prediction
            pred_np = pred[0].cpu().numpy()
            pred_colored = np.array(get_color_pallete(pred_np, 'citys'))
            
            # Get label for comparison
            label_np = label[0].numpy() if label is not None else None
            
            examples_collected.append({
                'rgb': rgb_img,
                'disparity': disp,
                'alpha': alpha_resized,
                'mean_alpha': mean_alpha,
                'prediction': pred_colored,
                'label': label_np,
                'rgb_features': rgb_feat,
                'depth_features': depth_feat,
                'idx': idx
            })
    
    # Sort by mean alpha to get low and high examples
    examples_collected.sort(key=lambda x: x['mean_alpha'])
    
    # Select examples: 2 low alpha, 2 high alpha
    low_alpha_examples = examples_collected[:num_examples//2]
    high_alpha_examples = examples_collected[-num_examples//2:]
    selected_examples = low_alpha_examples + high_alpha_examples
    
    # Restore original forward
    fusion_module.forward = original_forward
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(num_examples, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, example in enumerate(selected_examples):
        row = i
        
        # RGB Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(example['rgb'])
        ax.set_title('RGB Image', fontsize=10)
        ax.axis('off')
        
        # Disparity
        ax = fig.add_subplot(gs[row, 1])
        disp_vis = example['disparity']
        # Normalize for visualization
        disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-6)
        ax.imshow(disp_vis, cmap='viridis')
        ax.set_title('Disparity', fontsize=10)
        ax.axis('off')
        
        # Alpha Trust Map
        ax = fig.add_subplot(gs[row, 2])
        alpha_vis = example['alpha']
        im = ax.imshow(alpha_vis, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'Alpha Trust (mean={example["mean_alpha"]:.3f})', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Prediction
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(example['prediction'])
        ax.set_title('Prediction', fontsize=10)
        ax.axis('off')
        
        # Ground Truth (if available)
        ax = fig.add_subplot(gs[row, 4])
        if example['label'] is not None:
            label_colored = np.array(get_color_pallete(example['label'], 'citys'))
            ax.imshow(label_colored)
            ax.set_title('Ground Truth', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No GT', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ground Truth', fontsize=10)
        ax.axis('off')
        
        # Feature Space Visualization (t-SNE or PCA of features)
        ax = fig.add_subplot(gs[row, 5])
        if example['rgb_features'] is not None and example['depth_features'] is not None:
            try:
                # Sample features for visualization
                rgb_feat = example['rgb_features'][0].cpu().numpy()  # [C, H, W]
                depth_feat = example['depth_features'][0].cpu().numpy()
                
                # Flatten spatial dimensions
                rgb_flat = rgb_feat.reshape(rgb_feat.shape[0], -1).T  # [H*W, C]
                depth_flat = depth_feat.reshape(depth_feat.shape[0], -1).T
                
                # Sample points for visualization
                n_samples = min(1000, rgb_flat.shape[0])
                indices = np.random.choice(rgb_flat.shape[0], n_samples, replace=False)
                rgb_sample = rgb_flat[indices]
                depth_sample = depth_flat[indices]
                
                # Use PCA for dimensionality reduction
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    
                    # Combine and reduce
                    combined = np.vstack([rgb_sample, depth_sample])
                    combined_2d = pca.fit_transform(combined)
                    
                    # Plot
                    n_rgb = rgb_sample.shape[0]
                    ax.scatter(combined_2d[:n_rgb, 0], combined_2d[:n_rgb, 1], 
                              c='red', alpha=0.5, s=10, label='RGB')
                    ax.scatter(combined_2d[n_rgb:, 0], combined_2d[n_rgb:, 1], 
                              c='blue', alpha=0.5, s=10, label='Depth')
                    ax.legend(fontsize=8)
                    ax.set_title('Feature Space (PCA)', fontsize=10)
                    ax.set_xlabel('PC1', fontsize=8)
                    ax.set_ylabel('PC2', fontsize=8)
                except ImportError:
                    # Fallback: simple mean/std visualization
                    rgb_mean = rgb_sample.mean(axis=1)
                    depth_mean = depth_sample.mean(axis=1)
                    ax.scatter(rgb_mean, depth_mean, c='purple', alpha=0.5, s=10)
                    ax.set_title('Feature Space\n(Mean Features)', fontsize=10)
                    ax.set_xlabel('RGB Mean', fontsize=8)
                    ax.set_ylabel('Depth Mean', fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                ax.set_title('Feature Space', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Features\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Feature Space', fontsize=10)
        ax.axis('off')
    
    # Add overall title
    fig.suptitle('Alpha Trust Analysis: Low vs High Trust Regions', fontsize=16, y=0.98)
    
    # Add legend for alpha values
    legend_elements = [
        mpatches.Patch(color='red', label='High α (Trust RGB)'),
        mpatches.Patch(color='yellow', label='Medium α (Balanced)'),
        mpatches.Patch(color='green', label='Low α (Trust Depth)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))
    
    plt.savefig(os.path.join(output_dir, 'alpha_trust_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved alpha trust visualization to {output_dir}/alpha_trust_analysis.png")


def visualize_normals(model, dataloader, device, output_dir='figures', num_examples=6):
    """
    Visualize surface normals computed from disparity
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    hook_manager = HookManager()
    
    # Hook the geometry generator to get normals
    geo_generator = model.learning_to_downsample.geo_generator
    hook_manager.register_hook('geometry_features', geo_generator)
    
    examples = []
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataloader):
            if len(examples) >= num_examples:
                break
            
            image = image.to(device)
            original_size = image.shape[2:]
            
            # Forward pass through geometry generator only
            x_raw_disp = image[:, 3:4, :, :]  # [B, 1, H, W]
            geo_features = geo_generator(x_raw_disp)  # [B, 3, H, W]: [depth, norm_x, norm_y]
            
            # Get components
            depth = geo_features[0, 0].cpu().numpy()
            norm_x = geo_features[0, 1].cpu().numpy()
            norm_y = geo_features[0, 2].cpu().numpy()
            
            # Resize to original if needed
            if depth.shape != original_size:
                depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
                norm_x_t = torch.from_numpy(norm_x).unsqueeze(0).unsqueeze(0).float()
                norm_y_t = torch.from_numpy(norm_y).unsqueeze(0).unsqueeze(0).float()
                
                depth_t = F.interpolate(depth_t, size=original_size, mode='bilinear', align_corners=True)
                norm_x_t = F.interpolate(norm_x_t, size=original_size, mode='bilinear', align_corners=True)
                norm_y_t = F.interpolate(norm_y_t, size=original_size, mode='bilinear', align_corners=True)
                
                depth = depth_t.squeeze().numpy()
                norm_x = norm_x_t.squeeze().numpy()
                norm_y = norm_y_t.squeeze().numpy()
            
            # Get RGB image
            rgb_img = image[0, :3].cpu()
            rgb_img = rgb_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_img = torch.clamp(rgb_img, 0, 1)
            rgb_img = rgb_img.permute(1, 2, 0).numpy()
            
            # Get raw disparity
            disp_raw = image[0, 3].cpu().numpy()
            
            # Compute normal visualization (RGB encoding)
            # Normalize normals to [0, 1] for visualization
            norm_x_vis = (norm_x - norm_x.min()) / (norm_x.max() - norm_x.min() + 1e-6)
            norm_y_vis = (norm_y - norm_y.min()) / (norm_y.max() - norm_y.min() + 1e-6)
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            
            # Create RGB normal map: R=norm_x, G=norm_y, B=depth
            normal_map = np.stack([norm_x_vis, norm_y_vis, depth_vis], axis=2)
            
            # Also create a quiver plot for normals (subsampled)
            h, w = norm_x.shape
            step = max(1, min(h, w) // 50)  # Sample every Nth pixel
            y_coords, x_coords = np.meshgrid(
                np.arange(0, h, step),
                np.arange(0, w, step),
                indexing='ij'
            )
            norm_x_sampled = norm_x[::step, ::step]
            norm_y_sampled = norm_y[::step, ::step]
            
            examples.append({
                'rgb': rgb_img,
                'disparity': disp_raw,
                'depth': depth,
                'norm_x': norm_x,
                'norm_y': norm_y,
                'normal_map': normal_map,
                'quiver_data': (x_coords, y_coords, norm_x_sampled, norm_y_sampled)
            })
    
    # Create figure
    fig = plt.figure(figsize=(18, 3 * num_examples))
    gs = GridSpec(num_examples, 5, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, example in enumerate(examples):
        row = i
        
        # RGB Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(example['rgb'])
        ax.set_title('RGB Image', fontsize=10)
        ax.axis('off')
        
        # Disparity
        ax = fig.add_subplot(gs[row, 1])
        disp_vis = (example['disparity'] - example['disparity'].min()) / (example['disparity'].max() - example['disparity'].min() + 1e-6)
        ax.imshow(disp_vis, cmap='viridis')
        ax.set_title('Raw Disparity', fontsize=10)
        ax.axis('off')
        
        # Normal Map (RGB encoded)
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(example['normal_map'])
        ax.set_title('Normal Map\n(R=norm_x, G=norm_y, B=depth)', fontsize=10)
        ax.axis('off')
        
        # Normal X component
        ax = fig.add_subplot(gs[row, 3])
        im = ax.imshow(example['norm_x'], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Normal X Component', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Normal Y component
        ax = fig.add_subplot(gs[row, 4])
        im = ax.imshow(example['norm_y'], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Normal Y Component', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle('Surface Normals Visualization', fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_dir, 'surface_normals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normals visualization to {output_dir}/surface_normals.png")
    
    # Create a detailed quiver plot for one example
    if len(examples) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        example = examples[0]
        
        # Overlay normals on RGB
        axes[0].imshow(example['rgb'])
        x_coords, y_coords, norm_x_sampled, norm_y_sampled = example['quiver_data']
        axes[0].quiver(x_coords, y_coords, norm_x_sampled, norm_y_sampled, 
                      scale=50, color='yellow', width=0.003, alpha=0.7)
        axes[0].set_title('Surface Normals Overlay on RGB', fontsize=12)
        axes[0].axis('off')
        
        # Overlay normals on disparity
        disp_vis = (example['disparity'] - example['disparity'].min()) / (example['disparity'].max() - example['disparity'].min() + 1e-6)
        axes[1].imshow(disp_vis, cmap='viridis')
        axes[1].quiver(x_coords, y_coords, norm_x_sampled, norm_y_sampled,
                      scale=50, color='yellow', width=0.003, alpha=0.7)
        axes[1].set_title('Surface Normals Overlay on Disparity', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'surface_normals_quiver.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved quiver plot to {output_dir}/surface_normals_quiver.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Alpha Trust and Surface Normals')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='citys_d',
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split (val or test)')
    parser.add_argument('--num-examples', type=int, default=4,
                        help='Number of examples to visualize')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--alpha-only', action='store_true',
                        help='Only generate alpha trust visualization')
    parser.add_argument('--normals-only', action='store_true',
                        help='Only generate normals visualization')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='Base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='Crop image size')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Use auxiliary loss')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    data_kwargs = {
        'transform': input_transform,
        'base_size': args.base_size,
        'crop_size': args.crop_size
    }
    
    dataset = get_segmentation_dataset(
        args.dataset, split=args.split, mode='val', **data_kwargs
    )
    
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,  # Shuffle to get diverse examples
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    model = get_fast_scnn_d(
        dataset=args.dataset, aux=args.aux, pretrained=False
    ).to(device)
    
    if os.path.isfile(args.resume):
        print(f'Loading checkpoint from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {}
            for k, v in checkpoint.items():
                new_checkpoint[k.replace('module.', '')] = v
            checkpoint = new_checkpoint
        model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully!')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
    
    # Generate visualizations
    if not args.normals_only:
        print("\nGenerating alpha trust visualization...")
        visualize_alpha_trust(model, dataloader, device, args.output_dir, args.num_examples)
    
    if not args.alpha_only:
        print("\nGenerating normals visualization...")
        visualize_normals(model, dataloader, device, args.output_dir, args.num_examples)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    import argparse
    main()

