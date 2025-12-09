"""
Create pipeline visualization showing intermediate steps:
1. Input RGB image
2. Disparity
3. Generated normals
4. Gated fusion alpha (trust map)
5. Final segmentation output
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import argparse
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_segmentation_dataset
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from utils.visualize import get_color_pallete
from train import parse_args


def denormalize_image(tensor):
    """Denormalize ImageNet normalized tensor"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def create_pipeline_visualization(model, dataloader, device, 
                                 output_path='figures/pipeline_visualization.png',
                                 num_examples=4):
    """
    Create visualization showing pipeline stages
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    
    # Hook to capture alpha values from gated fusion
    fusion_module = model.learning_to_downsample.fusion
    original_forward = fusion_module.forward
    
    alpha_values = []
    def forward_with_alpha(x_rgb, x_depth):
        cat_feats = torch.cat([x_rgb, x_depth], dim=1)
        alpha = fusion_module.gate_conv(cat_feats)
        alpha_values.append(alpha.detach().cpu())
        return (x_rgb * alpha) + (x_depth * (1 - alpha))
    
    fusion_module.forward = forward_with_alpha
    
    examples = []
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataloader):
            if len(examples) >= num_examples:
                break
            
            alpha_values.clear()
            
            image = image.to(device)
            original_size = image.shape[2:]
            
            # Get RGB and disparity
            rgb_tensor = image[0, :3]
            disp_tensor = image[0, 3:4]
            
            # Denormalize RGB for visualization
            rgb_img = denormalize_image(rgb_tensor.cpu())
            rgb_img = torch.clamp(rgb_img, 0, 1)
            rgb_img = rgb_img.permute(1, 2, 0).numpy()
            
            # Get disparity for visualization
            # The disparity tensor is already normalized, so we just visualize it
            # Reverse normalization: (normalized * std) + mean
            disp_normalized = disp_tensor.cpu().numpy().squeeze()
            REAL_MEAN = 35.42
            REAL_STD = 28.04
            disp_real = (disp_normalized * REAL_STD) + REAL_MEAN
            # For visualization, normalize to [0, 1] for better contrast
            disp_vis = (disp_real - disp_real.min()) / (disp_real.max() - disp_real.min() + 1e-6)
            
            # Get geometry features (normals) from the geometry generator
            geo_generator = model.learning_to_downsample.geo_generator
            geo_features = geo_generator(disp_tensor.unsqueeze(0))  # [1, 3, H, W]
            
            depth = geo_features[0, 0].cpu().numpy()
            norm_x = geo_features[0, 1].cpu().numpy()
            norm_y = geo_features[0, 2].cpu().numpy()
            
            # Resize normals to original size if needed
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
            
            # Create normal map visualization
            norm_x_vis = (norm_x - norm_x.min()) / (norm_x.max() - norm_x.min() + 1e-6)
            norm_y_vis = (norm_y - norm_y.min()) / (norm_y.max() - norm_y.min() + 1e-6)
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            normal_map = np.stack([norm_x_vis, norm_y_vis, depth_vis], axis=2)
            
            # Forward pass to get predictions and alpha
            outputs = model(image)
            pred = torch.argmax(outputs[0], 1)[0].cpu().numpy()
            pred_colored = np.array(get_color_pallete(pred, 'citys'))
            
            # Get alpha trust map
            if len(alpha_values) > 0:
                alpha = alpha_values[0].numpy().squeeze()  # [H, W]
                # Resize alpha to original size
                if alpha.shape != original_size:
                    alpha_t = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
                    alpha_t = F.interpolate(alpha_t, size=original_size, mode='bilinear', align_corners=True)
                    alpha = alpha_t.squeeze().numpy()
            else:
                alpha = np.zeros(original_size)
            
            # Get ground truth if available
            label_np = label[0].numpy() if label is not None else None
            if label_np is not None:
                # Convert labelIDs to trainIDs if needed
                if label_np.max() > 18:
                    labelid_to_trainid = {
                        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
                    }
                    lookup = np.zeros(256, dtype=np.uint8)
                    for lid, tid in labelid_to_trainid.items():
                        lookup[lid] = tid
                    label_np = lookup[label_np]
                label_np = np.clip(label_np, 0, 18)
                label_colored = np.array(get_color_pallete(label_np, 'citys'))
            else:
                label_colored = None
            
            examples.append({
                'rgb': rgb_img,
                'disparity': disp_vis,
                'normal_map': normal_map,
                'norm_x': norm_x,
                'norm_y': norm_y,
                'alpha': alpha,
                'prediction': pred_colored,
                'label': label_colored
            })
    
    # Restore original forward
    fusion_module.forward = original_forward
    
    # Create figure
    fig_width = 24
    fig_height_per_row = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height_per_row * num_examples))
    gs = GridSpec(num_examples, 6, figure=fig, hspace=0.25, wspace=0.15)
    
    for i, example in enumerate(examples):
        row = i
        
        # 1. RGB Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(example['rgb'])
        ax.set_title('Input RGB', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # 2. Disparity
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(example['disparity'], cmap='viridis')
        ax.set_title('Disparity', fontsize=11, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        ax.axis('off')
        
        # 3. Generated Normals
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(example['normal_map'])
        ax.set_title('Generated Normals\n(R=norm_x, G=norm_y, B=depth)', 
                    fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # 4. Alpha Trust Map
        ax = fig.add_subplot(gs[row, 3])
        im = ax.imshow(example['alpha'], cmap='RdYlGn', vmin=0, vmax=1)
        mean_alpha = example['alpha'].mean()
        ax.set_title(f'Gated Fusion α\n(mean={mean_alpha:.3f})', 
                    fontsize=11, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        ax.axis('off')
        
        # 5. Final Output
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(example['prediction'])
        ax.set_title('Final Output\n(Fast-SCNN-D)', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # 6. Ground Truth (optional)
        ax = fig.add_subplot(gs[row, 5])
        if example['label'] is not None:
            ax.imshow(example['label'])
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold', pad=8)
        else:
            ax.text(0.5, 0.5, 'No GT', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    
    fig.suptitle('Fast-SCNN-D Pipeline Visualization', 
                fontsize=14, fontweight='bold', y=0.99)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pipeline visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create pipeline visualization')
    parser.add_argument('--resume', type=str, required=True,
                       help='Path to Fast-SCNN-D checkpoint')
    parser.add_argument('--dataset', type=str, default='citys_d',
                       help='Dataset name')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--num-examples', type=int, default=4,
                       help='Number of examples to show')
    parser.add_argument('--output-path', type=str, default='figures/pipeline_visualization.png',
                       help='Output path for figure')
    parser.add_argument('--base-size', type=int, default=1024,
                       help='Base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                       help='Crop image size')
    parser.add_argument('--aux', action='store_true', default=False,
                       help='Use auxiliary loss')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")
    
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
    
    # Create generator for DataLoader if seed is set
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )
    
    # Load model
    model = get_fast_scnn_d(
        dataset=args.dataset, aux=args.aux, pretrained=False
    ).to(device)
    
    if os.path.isfile(args.resume):
        print(f'Loading checkpoint from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            checkpoint = new_checkpoint
        model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully!')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
    
    # Create visualization
    print(f'\nGenerating pipeline visualization with {args.num_examples} examples...')
    create_pipeline_visualization(
        model, dataloader, device, args.output_path, args.num_examples
    )
    
    print('\nDone!')


if __name__ == '__main__':
    main()

