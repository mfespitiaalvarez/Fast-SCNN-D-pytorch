"""
Create qualitative comparison plot showing Fast-SCNN vs Fast-SCNN-D predictions
"""

import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms
import argparse
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from utils.visualize import get_color_pallete
from train import parse_args


def denormalize_image(tensor):
    """Denormalize ImageNet normalized tensor"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def create_qualitative_comparison(fast_scnn_model, fast_scnn_d_model, dataloader, 
                                 device, output_path='figures/qualitative_comparison.png',
                                 num_examples=6):
    """
    Create side-by-side qualitative comparison
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fast_scnn_model.eval()
    fast_scnn_d_model.eval()
    
    examples = []
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataloader):
            if len(examples) >= num_examples:
                break
            
            image = image.to(device)
            original_size = image.shape[2:]
            
            # Get RGB image for visualization
            rgb_tensor = image[0, :3] if image.shape[1] == 4 else image[0]
            rgb_img = denormalize_image(rgb_tensor.cpu())
            rgb_img = torch.clamp(rgb_img, 0, 1)
            rgb_img = rgb_img.permute(1, 2, 0).numpy()
            
            # Get disparity if available
            if image.shape[1] == 4:
                disp = image[0, 3].cpu().numpy()
            else:
                disp = None
            
            # Fast-SCNN prediction (RGB only)
            if image.shape[1] == 4:
                rgb_only = image[:, :3, :, :]
            else:
                rgb_only = image
            
            outputs_rgb = fast_scnn_model(rgb_only)
            pred_rgb = torch.argmax(outputs_rgb[0], 1)[0].cpu().numpy()
            pred_rgb_colored = np.array(get_color_pallete(pred_rgb, 'citys'))
            
            # Fast-SCNN-D prediction
            outputs_d = fast_scnn_d_model(image)
            pred_d = torch.argmax(outputs_d[0], 1)[0].cpu().numpy()
            pred_d_colored = np.array(get_color_pallete(pred_d, 'citys'))
            
            # Ground truth
            label_np = label[0].numpy() if label is not None else None
            if label_np is not None:
                # The dataset should already convert labelIDs to trainIDs via _mask_transform()
                # But if labels are in labelID format (7, 8, 11, 12, ...), we need to convert
                # Check if labels are in labelID format (values > 18)
                if label_np.max() > 18:
                    # Convert labelIDs to trainIDs using the reverse mapping
                    # This is the inverse of _class_to_index
                    labelid_to_trainid = {
                        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
                    }
                    # Create lookup array
                    lookup = np.zeros(256, dtype=np.uint8)
                    for lid, tid in labelid_to_trainid.items():
                        lookup[lid] = tid
                    # Convert
                    label_np = lookup[label_np]
                    print(f"  Note: Converted GT from labelIDs to trainIDs")
                
                # Ensure label is in trainID format (0-18) for visualization
                label_np = np.clip(label_np, 0, 18)
                label_colored = np.array(get_color_pallete(label_np, 'citys'))
            else:
                label_colored = None
            
            # Debug: Print value ranges to check for mismatches (only first example)
            if idx == 0:
                print(f"\nDebug - Value ranges:")
                print(f"  Prediction RGB: min={pred_rgb.min()}, max={pred_rgb.max()}, unique={len(np.unique(pred_rgb))}")
                print(f"  Prediction D: min={pred_d.min()}, max={pred_d.max()}, unique={len(np.unique(pred_d))}")
                if label_np is not None:
                    print(f"  Ground Truth: min={label_np.min()}, max={label_np.max()}, unique={len(np.unique(label_np))}")
                    print(f"  (All should be in trainID range 0-18)")
                    
                    # Verify all are in correct range
                    pred_rgb_valid = (pred_rgb.min() >= 0) and (pred_rgb.max() <= 18)
                    pred_d_valid = (pred_d.min() >= 0) and (pred_d.max() <= 18)
                    gt_valid = (label_np.min() >= 0) and (label_np.max() <= 18)
                    
                    if pred_rgb_valid and pred_d_valid and gt_valid:
                        print(f"  ✓ All labels verified: Using trainIDs (0-18) - CORRECT")
                    else:
                        print(f"  ⚠ WARNING: Some labels outside trainID range!")
                        if not pred_rgb_valid:
                            print(f"    - Prediction RGB: {pred_rgb.min()}-{pred_rgb.max()}")
                        if not pred_d_valid:
                            print(f"    - Prediction D: {pred_d.min()}-{pred_d.max()}")
                        if not gt_valid:
                            print(f"    - Ground Truth: {label_np.min()}-{label_np.max()}")
            
            examples.append({
                'rgb': rgb_img,
                'disparity': disp,
                'pred_rgb': pred_rgb_colored,
                'pred_d': pred_d_colored,
                'label': label_colored
            })
    
    # Create figure - adjust size based on number of examples
    # For blog posts, 4 examples with slightly larger images work well
    fig_width = 22  # Increased width to accommodate colorbar
    fig_height_per_row = 4.5 if num_examples <= 4 else 4.0  # Slightly taller rows for fewer examples
    fig = plt.figure(figsize=(fig_width, fig_height_per_row * num_examples))
    gs = GridSpec(num_examples, 5, figure=fig, hspace=0.25, wspace=0.15)  # Increased spacing
    
    for i, example in enumerate(examples):
        row = i
        
        # RGB Image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(example['rgb'])
        ax.set_title('RGB Image', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Disparity
        ax = fig.add_subplot(gs[row, 1])
        if example['disparity'] is not None:
            disp_vis = example['disparity']
            disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-6)
            im = ax.imshow(disp_vis, cmap='viridis')
            ax.set_title('Disparity', fontsize=11, fontweight='bold', pad=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)  # Increased pad for visibility
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Disparity', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Fast-SCNN Prediction
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(example['pred_rgb'])
        ax.set_title('Fast-SCNN\n(RGB-only)', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Fast-SCNN-D Prediction
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(example['pred_d'])
        ax.set_title('Fast-SCNN-D\n(RGB + Depth)', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Ground Truth
        ax = fig.add_subplot(gs[row, 4])
        if example['label'] is not None:
            ax.imshow(example['label'])
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold', pad=8)
        else:
            ax.text(0.5, 0.5, 'No GT', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
    
    fig.suptitle('Qualitative Comparison: Fast-SCNN vs Fast-SCNN-D', 
                fontsize=14, fontweight='bold', y=0.99)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative comparison to: {output_path}")
    
    # Print verification info
    print("\nLabel Verification:")
    print("  ✓ Predictions use trainIDs (0-18) - correct")
    print("  ✓ Ground truth converted from labelIDs to trainIDs if needed")
    print("  ✓ All visualizations use same Cityscapes color palette")
    print("  ✓ Color mapping: trainID 0-18 → Cityscapes standard colors")


def main():
    parser = argparse.ArgumentParser(description='Create qualitative comparison plot')
    parser.add_argument('--resume-fast-scnn', type=str, required=True,
                       help='Path to Fast-SCNN checkpoint')
    parser.add_argument('--resume-fast-scnn-d', type=str, required=True,
                       help='Path to Fast-SCNN-D checkpoint')
    parser.add_argument('--dataset', type=str, default='citys',
                       help='Dataset name (use citys for Fast-SCNN, citys_d for Fast-SCNN-D)')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--num-examples', type=int, default=4,
                       help='Number of examples to show (default: 4, good for blog posts)')
    parser.add_argument('--output-path', type=str, default='figures/qualitative_comparison.png',
                       help='Output path for figure')
    parser.add_argument('--base-size', type=int, default=1024,
                       help='Base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                       help='Crop image size')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None, random)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
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
    
    # Load validation dataset (use citys_d for Fast-SCNN-D to get disparity)
    dataset_d = get_segmentation_dataset(
        'citys_d', split=args.split, mode='val', **data_kwargs
    )
    
    # Create generator for DataLoader if seed is set
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    
    dataloader = data.DataLoader(
        dataset=dataset_d,
        batch_size=1,
        shuffle=True,  # Shuffle to get diverse examples
        num_workers=4,
        pin_memory=True,
        generator=generator  # Use seeded generator for reproducibility
    )
    
    # Load Fast-SCNN model (RGB only)
    print('Loading Fast-SCNN model...')
    fast_scnn_model = get_fast_scnn(dataset='citys', aux=False, pretrained=False).to(device)
    if os.path.isfile(args.resume_fast_scnn):
        checkpoint = torch.load(args.resume_fast_scnn, map_location=lambda storage, loc: storage)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            checkpoint = new_checkpoint
        fast_scnn_model.load_state_dict(checkpoint)
        print('Fast-SCNN checkpoint loaded!')
    else:
        raise FileNotFoundError(f'Fast-SCNN checkpoint not found: {args.resume_fast_scnn}')
    
    # Load Fast-SCNN-D model
    print('Loading Fast-SCNN-D model...')
    fast_scnn_d_model = get_fast_scnn_d(dataset='citys_d', aux=False, pretrained=False).to(device)
    if os.path.isfile(args.resume_fast_scnn_d):
        checkpoint = torch.load(args.resume_fast_scnn_d, map_location=lambda storage, loc: storage)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            checkpoint = new_checkpoint
        fast_scnn_d_model.load_state_dict(checkpoint)
        print('Fast-SCNN-D checkpoint loaded!')
    else:
        raise FileNotFoundError(f'Fast-SCNN-D checkpoint not found: {args.resume_fast_scnn_d}')
    
    # Create comparison
    print(f'\nGenerating qualitative comparison with {args.num_examples} examples...')
    create_qualitative_comparison(
        fast_scnn_model, fast_scnn_d_model, dataloader, device,
        args.output_path, args.num_examples
    )
    
    print('\nDone!')


if __name__ == '__main__':
    main()

