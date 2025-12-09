"""
Prepare Cityscapes Server Submission

This script generates predictions for the Cityscapes test set and prepares them
for submission to the official evaluation server.

Requirements:
- Predictions must be at original resolution (2048x1024)
- Labels must be encoded as labelIDs (not trainIDs)
- Filenames must match: {city}_{sequence}_{frame}_gtFine_labelIds.png
- Files can be in arbitrary subfolders
- Must create a zip archive (max 100 MB)
"""

import os
import sys
import zipfile
import argparse
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from train import parse_args

# Cityscapes labelID to trainID mapping (reverse of _key)
# trainID -> labelID mapping
# valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
TRAINID_TO_LABELID = {
    0: 7,   # road
    1: 8,   # sidewalk
    2: 11,  # building
    3: 12,  # wall
    4: 13,  # fence
    5: 17,  # pole
    6: 19,  # traffic light
    7: 20,  # traffic sign
    8: 21,  # vegetation
    9: 22,  # terrain
    10: 23, # sky
    11: 24, # person
    12: 25, # rider
    13: 26, # car
    14: 27, # truck
    15: 28, # bus
    16: 31, # train
    17: 32, # motorcycle
    18: 33, # bicycle
}

# Create a lookup array for fast conversion
# For any trainID (0-18), map to labelID. For invalid trainIDs, map to 0 (unlabeled)
TRAINID_TO_LABELID_ARRAY = np.zeros(256, dtype=np.uint8)
for train_id, label_id in TRAINID_TO_LABELID.items():
    TRAINID_TO_LABELID_ARRAY[train_id] = label_id


def trainid_to_labelid(pred_tensor):
    """
    Convert trainIDs (0-18) to labelIDs (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
    
    Args:
        pred_tensor: numpy array of predictions with trainIDs (shape: H, W)
    
    Returns:
        numpy array with labelIDs (shape: H, W)
    """
    # Clip predictions to valid range
    pred_clipped = np.clip(pred_tensor, 0, 18)
    # Map trainIDs to labelIDs
    pred_labelid = TRAINID_TO_LABELID_ARRAY[pred_clipped]
    return pred_labelid


def get_output_filename(input_filename):
    """
    Convert input filename to output filename format.
    
    Input: berlin_000123_000019_leftImg8bit.png
    Output: berlin_000123_000019_gtFine_labelIds.png
    """
    # Remove _leftImg8bit and .png, add _gtFine_labelIds.png
    base = input_filename.replace('_leftImg8bit', '').replace('.png', '')
    return f"{base}_gtFine_labelIds.png"


class SubmissionGenerator(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # Create output directory
        self.output_dir = 'submission_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Image transform (same as training)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # Load test dataset
        # For test mode, we need to handle images at original resolution
        data_kwargs = {
            'transform': input_transform,
            'base_size': args.base_size,
            'crop_size': args.crop_size
        }
        
        # Create dataset in test mode
        # For fast_scnn_d, we need to use custom loader because dataset's test mode
        # doesn't concatenate RGB and disparity
        needs_disparity = (args.model == 'fast_scnn_d' or args.dataset == 'citys_d')
        
        if needs_disparity:
            # Always use custom loader for fast_scnn_d to properly handle disparity
            print(f'Note: Using custom loader for {args.model} to handle disparity...')
            self.image_paths, self.disparity_paths = self._get_test_image_paths(args.dataset)
            self.test_loader = self._create_test_loader(input_transform)
        else:
            # For regular fast_scnn, try to use dataset's test mode
            try:
                test_dataset = get_segmentation_dataset(
                    args.dataset, split='test', mode='test', **data_kwargs
                )
                self.image_paths = test_dataset.images if hasattr(test_dataset, 'images') else []
                self.test_loader = data.DataLoader(
                    dataset=test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
            except (AssertionError, RuntimeError, KeyError) as e:
                # If test split is not supported, load images directly
                print(f'Note: Loading test images directly from filesystem...')
                self.image_paths, self.disparity_paths = self._get_test_image_paths(args.dataset)
                self.test_loader = self._create_test_loader(input_transform)
        
        # Create network
        if args.model == 'fast_scnn_d' or args.dataset == 'citys_d':
            self.model = get_fast_scnn_d(
                dataset=args.dataset, aux=args.aux, pretrained=False
            ).to(self.device)
        else:
            self.model = get_fast_scnn(
                args.dataset, aux=args.aux, pretrained=False
            ).to(self.device)
        
        # Load checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Loading checkpoint from {args.resume}...')
                checkpoint = torch.load(
                    args.resume,
                    map_location=lambda storage, loc: storage
                )
                # Handle DataParallel models
                if any(key.startswith('module.') for key in checkpoint.keys()):
                    new_checkpoint = {}
                    for k, v in checkpoint.items():
                        new_checkpoint[k.replace('module.', '')] = v
                    checkpoint = new_checkpoint
                self.model.load_state_dict(checkpoint)
                print('Checkpoint loaded successfully!')
            else:
                raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
        else:
            raise ValueError('--resume is required for generating submissions')
        
        self.model.eval()
    
    def _get_test_image_paths(self, dataset_name):
        """Get all test image paths from the dataset directory"""
        root = './datasets/citys'
        test_img_folder = os.path.join(root, 'leftImg8bit/test')
        image_paths = []
        disparity_paths = []
        
        needs_disparity = (self.args.model == 'fast_scnn_d' or self.args.dataset == 'citys_d')
        
        if os.path.exists(test_img_folder):
            for root_dir, dirs, files in os.walk(test_img_folder):
                for filename in files:
                    if filename.endswith('.png') and 'leftImg8bit' in filename:
                        img_path = os.path.join(root_dir, filename)
                        image_paths.append(img_path)
                        
                        # Get corresponding disparity path if using fast_scnn_d
                        if needs_disparity:
                            foldername = os.path.basename(root_dir)
                            dispname = filename.replace('leftImg8bit', 'disparity')
                            disp_folder = os.path.join(root, 'disparity/test', foldername)
                            disp_path = os.path.join(disp_folder, dispname)
                            disparity_paths.append(disp_path)
            
            image_paths.sort()
            if disparity_paths:
                disparity_paths.sort()
            print(f'Found {len(image_paths)} test images')
            if needs_disparity and disparity_paths:
                print(f'Found {len(disparity_paths)} disparity images')
                # Check if all disparity files exist
                missing = sum(1 for p in disparity_paths if not os.path.exists(p))
                if missing > 0:
                    print(f'Warning: {missing} disparity files are missing!')
        else:
            raise RuntimeError(f'Test image folder not found: {test_img_folder}')
        
        return image_paths, disparity_paths
    
    def _create_test_loader(self, transform):
        """Create a data loader for test images"""
        from torch.utils.data import Dataset
        
        # Check if we need disparity
        needs_disparity = (self.args.model == 'fast_scnn_d' or self.args.dataset == 'citys_d')
        
        class TestImageDataset(Dataset):
            def __init__(self, image_paths, disparity_paths, transform, needs_disparity):
                self.image_paths = image_paths
                self.disparity_paths = disparity_paths if needs_disparity else None
                self.transform = transform
                self.needs_disparity = needs_disparity
                
                # Disparity processing constants (from cityscapes_d.py)
                self.RAW_MEAN_INT = 9070
                self.REAL_MEAN = 35.42
                self.REAL_STD = 28.04
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                img = Image.open(img_path).convert('RGB')
                
                if self.needs_disparity:
                    # Load and process disparity (matching cityscapes_d.py logic)
                    if self.disparity_paths and idx < len(self.disparity_paths):
                        disp_path = self.disparity_paths[idx]
                        if os.path.exists(disp_path):
                            # Load disparity (16-bit)
                            disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                            
                            # Fill holes with mean
                            mask_valid = disp_raw > 0
                            disp_raw[~mask_valid] = self.RAW_MEAN_INT
                            
                            # Convert to PIL for resizing (matching dataset approach)
                            disp_pil = Image.fromarray(disp_raw.astype(np.uint16))
                            # Resize to match image size (before transform)
                            disp_pil = disp_pil.resize(img.size, Image.NEAREST)
                            
                            # Convert back to numpy and normalize
                            disp_np = np.array(disp_pil).astype(np.float32)
                            # Convert raw to real disparity: d = (raw - 1) / 256.0
                            disp_np = (disp_np - 1.0) / 256.0
                            # Z-score normalization
                            disp_np = (disp_np - self.REAL_MEAN) / self.REAL_STD
                            
                            # Convert to tensor [1, H, W]
                            disp_tensor = torch.from_numpy(disp_np).unsqueeze(0)
                        else:
                            # If disparity file doesn't exist, create zero-filled tensor
                            w, h = img.size
                            disp_tensor = torch.zeros(1, h, w, dtype=torch.float32)
                    else:
                        # No disparity paths provided, create zero-filled tensor
                        w, h = img.size
                        disp_tensor = torch.zeros(1, h, w, dtype=torch.float32)
                
                # Transform RGB (this may resize the image)
                if self.transform:
                    img = self.transform(img)  # [3, H, W]
                
                # Concatenate RGB and disparity if needed
                if self.needs_disparity:
                    # Ensure disparity matches image size after transform
                    # The transform may have resized the image, so resize disparity to match
                    if disp_tensor.shape[1:] != img.shape[1:]:
                        disp_tensor = F.interpolate(
                            disp_tensor.unsqueeze(0), 
                            size=img.shape[1:], 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    input_tensor = torch.cat([img, disp_tensor], dim=0)  # [4, H, W]
                else:
                    input_tensor = img  # [3, H, W]
                
                filename = os.path.basename(img_path)
                return input_tensor, filename
        
        dataset = TestImageDataset(
            self.image_paths, 
            self.disparity_paths if hasattr(self, 'disparity_paths') else None,
            transform, 
            needs_disparity
        )
        return data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def generate_predictions(self):
        """Generate predictions for all test images"""
        print('='*70)
        print('Generating Predictions for Cityscapes Test Set')
        print('='*70)
        print(f'Model: {self.args.model}')
        print(f'Dataset: {self.args.dataset}')
        print(f'Output directory: {self.output_dir}')
        print(f'Number of test images: {len(self.test_loader)}')
        print('='*70)
        
        # Cityscapes test images are always 2048x1024
        TARGET_HEIGHT, TARGET_WIDTH = 1024, 2048
        
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                if len(batch) == 2:
                    image, filename = batch
                    # Handle both tuple/list and string
                    if isinstance(filename, (list, tuple)):
                        filename = filename[0]
                    elif isinstance(filename, torch.Tensor):
                        filename = filename.item() if filename.numel() == 1 else str(filename.item())
                else:
                    image = batch[0]
                    # Get filename from image paths if available
                    if idx < len(self.image_paths):
                        filename = os.path.basename(self.image_paths[idx])
                    else:
                        filename = f'test_{idx:06d}_leftImg8bit.png'
                
                image = image.to(self.device)
                
                # Get input image size
                _, _, input_h, input_w = image.shape
                
                # Forward pass
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy().squeeze(0)  # Remove batch dimension: [H, W]
                
                # Resize prediction to target size (2048x1024) if needed
                pred_h, pred_w = pred_np.shape
                if pred_h != TARGET_HEIGHT or pred_w != TARGET_WIDTH:
                    # Use nearest neighbor interpolation to preserve label IDs
                    pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()
                    pred_resized = F.interpolate(
                        pred_tensor, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='nearest'
                    )
                    pred_np = pred_resized.squeeze().cpu().numpy().astype(np.uint8)
                
                # Verify size
                assert pred_np.shape == (TARGET_HEIGHT, TARGET_WIDTH), \
                    f"Prediction size mismatch: {pred_np.shape} != ({TARGET_HEIGHT}, {TARGET_WIDTH})"
                
                # Convert trainIDs to labelIDs
                pred_labelid = trainid_to_labelid(pred_np)
                
                # Save prediction
                output_filename = get_output_filename(filename)
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save as PNG with labelIDs (uint8)
                pred_image = Image.fromarray(pred_labelid.astype(np.uint8), mode='L')
                pred_image.save(output_path)
                
                if (idx + 1) % 50 == 0:
                    print(f'Processed {idx + 1}/{len(self.test_loader)} images...')
        
        print(f'\nFinished generating {len(self.test_loader)} predictions!')
        print(f'Predictions saved to: {self.output_dir}')
        print(f'All predictions are at resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}')
    
    def create_submission_zip(self, zip_filename='submission.zip'):
        """Create zip archive for submission"""
        print(f'\nCreating submission zip archive: {zip_filename}')
        
        # Remove existing zip if it exists
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        
        # Create zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        # Add file to zip with relative path
                        arcname = os.path.relpath(file_path, self.output_dir)
                        zipf.write(file_path, arcname)
        
        # Check file size
        zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
        print(f'Submission zip created: {zip_filename}')
        print(f'File size: {zip_size_mb:.2f} MB')
        
        if zip_size_mb > 100:
            print(f'WARNING: Zip file size ({zip_size_mb:.2f} MB) exceeds 100 MB limit!')
        else:
            print(f'File size is within the 100 MB limit.')
        
        return zip_filename


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Cityscapes Server Submission',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with fast_scnn model:
  python prepare_submission.py --resume weights/fast_scnn_citys.pth --model fast_scnn --dataset citys

  # With fast_scnn_d model (requires disparity data):
  python prepare_submission.py --resume weights/fast_scnn_d_citys_d.pth --model fast_scnn_d --dataset citys_d

  # Custom zip filename:
  python prepare_submission.py --resume weights/fast_scnn_citys.pth --zip-name my_submission.zip

Note:
  - For fast_scnn_d model, test images must have corresponding disparity files
  - Predictions are automatically resized to 2048x1024 (original Cityscapes resolution)
  - Labels are converted from trainIDs to labelIDs as required by the server
  - The zip file must be under 100 MB for submission
        """
    )
    parser.add_argument('--no-zip', action='store_true',
                        help='Do not create zip archive (default: create zip)')
    parser.add_argument('--zip-name', type=str, default='submission.zip',
                        help='Name of the submission zip file (default: submission.zip)')
    
    # Parse known args first
    args, remaining = parser.parse_known_args()
    
    # Parse the rest using train's parse_args
    import sys
    sys.argv = [sys.argv[0]] + remaining
    train_args = parse_args()
    
    # Merge submission args
    train_args.create_zip = not args.no_zip  # Default to True unless --no-zip is specified
    train_args.zip_name = args.zip_name
    
    # Override some args for submission
    if not train_args.resume:
        print('Error: --resume is required for generating submissions.')
        print('Usage: python prepare_submission.py --resume <checkpoint_path> [other_args]')
        return
    
    # Create generator
    generator = SubmissionGenerator(train_args)
    
    # Generate predictions
    generator.generate_predictions()
    
    # Create zip archive
    if train_args.create_zip:
        zip_path = generator.create_submission_zip(train_args.zip_name)
        print(f'\n{"="*70}')
        print('SUBMISSION READY!')
        print(f'{"="*70}')
        print(f'Zip file: {zip_path}')
        print(f'You can now upload this file to the Cityscapes evaluation server.')
        print(f'Server: https://www.cityscapes-dataset.com/benchmarks/')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()

