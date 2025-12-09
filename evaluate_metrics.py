"""
Comprehensive Evaluation Script for Semantic Segmentation Models
Computes metrics suitable for paper reporting: mIoU, Pixel Accuracy, F1 Score, FPS, etc.
"""
import os
import json
import time
import argparse
import torch
import torch.utils.data as data
import numpy as np
from torchvision import transforms
from collections import OrderedDict

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from models.fast_scnn_d import get_fast_scnn as get_fast_scnn_d
from utils.metric import SegmentationMetric, hist_info, compute_score
from train import parse_args

# Cityscapes class names (19 classes)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]


class ComprehensiveEvaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # Image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # Dataset and dataloader
        # Use test split if specified, otherwise use val
        eval_split = getattr(args, 'eval_split', 'val')
        # For test set evaluation, we need to use 'val' mode (not 'test' mode) 
        # because 'test' mode returns early without labels
        # 'val' mode will load labels if they exist in the test set
        eval_mode = 'val'  # Use 'val' mode to load labels for evaluation
        
        data_kwargs = {
            'transform': input_transform,
            'base_size': args.base_size,
            'crop_size': args.crop_size
        }
        val_dataset = get_segmentation_dataset(
            args.dataset, split=eval_split, mode=eval_mode, **data_kwargs
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.num_classes = val_dataset.num_class
        
        # Check if mask files exist for test set
        if eval_split == 'test' and hasattr(val_dataset, 'mask_paths'):
            mask_exists = any(os.path.isfile(mp) for mp in val_dataset.mask_paths[:10]) if val_dataset.mask_paths else False
            if mask_exists:
                print(f'Note: Found mask files for test set, but they may contain invalid labels.')
        
        # Create network
        if args.model == 'fast_scnn_d' or args.dataset == 'citys_d':
            self.model = get_fast_scnn_d(
                dataset=args.dataset, aux=args.aux, pretrained=False
            ).to(self.device)
        else:
            self.model = get_fast_scnn(
                dataset=args.dataset, aux=args.aux, pretrained=False
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
            print('Warning: No checkpoint specified. Using random initialization.')
        
        self.model.eval()
        
        # Compute model size
        self.model_params = self._count_parameters()
        self.model_size_mb = self._get_model_size()
        
    def _count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_millions': total_params / 1e6,
            'trainable_millions': trainable_params / 1e6
        }
    
    def _get_model_size(self):
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024.0 / 1024.0
        return size_all_mb
    
    def compute_f1_score(self, confusion_matrix):
        """Compute F1 score per class from confusion matrix"""
        n_classes = confusion_matrix.shape[0]
        f1_scores = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp + np.finfo(float).eps)
            recall = tp / (tp + fn + np.finfo(float).eps)
            f1_scores[i] = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)
        
        return f1_scores
    
    def evaluate(self, warmup_iterations=10, timing_iterations=100):
        """Comprehensive evaluation"""
        eval_split = getattr(self.args, 'eval_split', 'val')
        print('='*70)
        print('Starting Comprehensive Evaluation')
        print('='*70)
        print(f'Model: {self.args.model}')
        print(f'Dataset: {self.args.dataset}')
        print(f'Split: {eval_split}')
        print(f'Number of classes: {self.num_classes}')
        print(f'Evaluation samples: {len(self.val_loader)}')
        print('='*70)
        
        # Initialize metrics
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        total_correct = 0
        total_label = 0
        total_inter = np.zeros(self.num_classes)
        total_union = np.zeros(self.num_classes)
        
        # Lists for per-image metrics
        inference_times = []
        input_sizes = []  # Track actual input image sizes
        
        eval_split = getattr(self.args, 'eval_split', 'val')
        split_name = 'test' if eval_split == 'test' else 'validation'
        print(f'\nEvaluating on {split_name} set...')
        with torch.no_grad():
            for idx, (image, label) in enumerate(self.val_loader):
                image = image.to(self.device)
                # Handle label shape - might be [1, H, W] or [H, W]
                label_np = label.numpy().squeeze()  # Remove batch dimension if present
                
                # Track input image size (H, W)
                if idx == 0:  # Get size from first image
                    input_height, input_width = image.shape[-2:]
                    input_sizes.append((input_height, input_width))
                    # Debug: Check shapes and value ranges
                    print(f'\nDebug - First sample:')
                    print(f'  Image shape: {image.shape}')
                    print(f'  Label shape: {label_np.shape}')
                    print(f'  Label dtype: {label_np.dtype}')
                    print(f'  Label min: {label_np.min()}, max: {label_np.max()}')
                    unique_vals = np.unique(label_np)
                    print(f'  Label unique values (first 20): {unique_vals[:20]}')
                    print(f'  Total unique values: {len(unique_vals)}')
                    print(f'  Number of -1 values: {np.sum(label_np == -1)}')
                    print(f'  Number of valid values (>=0 and <{self.num_classes}): {np.sum((label_np >= 0) & (label_np < self.num_classes))}')
                
                # Measure inference time (after warmup)
                if idx >= warmup_iterations:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                
                # Forward pass
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                
                if idx >= warmup_iterations:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                
                pred_np = pred.cpu().data.numpy()
                # Remove batch dimension if present - handle [1, H, W] -> [H, W]
                if pred_np.ndim == 3:
                    if pred_np.shape[0] == 1:
                        pred_np = pred_np[0]  # Remove first dimension: [1, H, W] -> [H, W]
                    else:
                        pred_np = pred_np.squeeze()
                
                if idx == 0:  # Debug first prediction
                    print(f'  Prediction shape: {pred_np.shape}')
                    print(f'  Prediction dtype: {pred_np.dtype}')
                    print(f'  Prediction min: {pred_np.min()}, max: {pred_np.max()}')
                    print(f'  Prediction unique values: {np.unique(pred_np)[:10]}...')
                
                # Ensure shapes match
                if pred_np.shape != label_np.shape:
                    print(f'Warning: Shape mismatch at idx {idx}: pred {pred_np.shape} vs label {label_np.shape}')
                    # Try to reshape if possible
                    if pred_np.size == label_np.size:
                        pred_np = pred_np.reshape(label_np.shape)
                    else:
                        print(f'  Skipping sample {idx} due to shape mismatch')
                        continue
                
                # Update confusion matrix
                hist, labeled, correct = hist_info(pred_np, label_np, self.num_classes)
                confusion_matrix += hist
                total_correct += correct
                total_label += labeled
                
                if idx == 0:  # Debug first sample metrics
                    print(f'  Labeled pixels: {labeled}, Correct: {correct}')
                    print(f'  Confusion matrix sum: {hist.sum()}')
                
                # Update IoU
                pred_flat = pred_np.flatten()
                label_flat = label_np.flatten()
                valid_mask = (label_flat >= 0) & (label_flat < self.num_classes)
                
                for c in range(self.num_classes):
                    inter = np.sum((pred_flat == c) & (label_flat == c) & valid_mask)
                    union = (np.sum((pred_flat == c) & valid_mask) +
                            np.sum((label_flat == c) & valid_mask) - inter)
                    total_inter[c] += inter
                    total_union[c] += union
                
                if (idx + 1) % 50 == 0:
                    print(f'Processed {idx + 1}/{len(self.val_loader)} images...')
        
        # Compute metrics
        print('\nComputing metrics...')
        print(f'Debug - Total labeled pixels: {total_label}')
        print(f'Debug - Total correct pixels: {total_correct}')
        print(f'Debug - Confusion matrix sum: {confusion_matrix.sum()}')
        print(f'Debug - Confusion matrix diagonal: {np.diag(confusion_matrix)}')
        
        if total_label == 0:
            print('\n' + '='*70)
            print('WARNING: No labeled pixels found!')
            print('='*70)
            print('The test set appears to have no valid ground truth labels.')
            print('\nWhy this is expected:')
            print('  The official Cityscapes test set does NOT provide public ground')
            print('  truth labels. This is by design - you must submit predictions to')
            print('  the Cityscapes evaluation server to get test set metrics.')
            print('  See: https://www.cityscapes-dataset.com/benchmarks/')
            print('\nTo get local evaluation metrics, please use the validation set:')
            print('  python evaluate_metrics.py --eval-split val ...')
            print('='*70)
            # Return a minimal results dict to avoid crash
            return {
                'model': self.args.model,
                'dataset': self.args.dataset,
                'split': getattr(self.args, 'eval_split', 'test'),
                'checkpoint': self.args.resume if self.args.resume else 'None',
                'num_classes': self.num_classes,
                'num_samples': len(self.val_loader),
                'error': 'No valid labels found in test set',
                'note': 'Cityscapes test set does not have public ground truth labels'
            }
        
        # Pixel Accuracy
        pixel_acc = total_correct / (total_label + np.finfo(float).eps)
        
        # Per-class IoU
        iou_per_class = total_inter / (total_union + np.finfo(float).eps)
        mIoU = np.nanmean(iou_per_class)
        
        # Per-class F1 Score
        f1_per_class = self.compute_f1_score(confusion_matrix)
        mean_f1 = np.nanmean(f1_per_class)
        
        # Inference Speed (FPS)
        if len(inference_times) > 0:
            avg_inference_time = np.mean(inference_times)
            fps = 1.0 / avg_inference_time
            std_inference_time = np.std(inference_times)
            min_inference_time = np.min(inference_times)
            max_inference_time = np.max(inference_times)
        else:
            avg_inference_time = 0
            fps = 0
            std_inference_time = 0
            min_inference_time = 0
            max_inference_time = 0
        
        # Per-class Precision and Recall
        precision_per_class = np.zeros(self.num_classes)
        recall_per_class = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision_per_class[i] = tp / (tp + fp + np.finfo(float).eps)
            recall_per_class[i] = tp / (tp + fn + np.finfo(float).eps)
        
        mean_precision = np.nanmean(precision_per_class)
        mean_recall = np.nanmean(recall_per_class)
        
        # Get input image size information
        actual_input_size = input_sizes[0] if len(input_sizes) > 0 else (0, 0)
        
        # Compile results
        eval_split = getattr(self.args, 'eval_split', 'val')
        results = {
            'model': self.args.model,
            'dataset': self.args.dataset,
            'split': eval_split,
            'checkpoint': self.args.resume if self.args.resume else 'None',
            'num_classes': self.num_classes,
            'num_samples': len(self.val_loader),
            'input_size': {
                'height': int(actual_input_size[0]),
                'width': int(actual_input_size[1]),
                'format': f'{int(actual_input_size[0])}x{int(actual_input_size[1])}',
                'base_size': int(self.args.base_size),
                'crop_size': int(self.args.crop_size),
            },
            'model_parameters': self.model_params,
            'model_size_mb': self.model_size_mb,
            'metrics': {
                'pixel_accuracy': float(pixel_acc),
                'pixel_accuracy_percent': float(pixel_acc * 100),
                'mean_iou': float(mIoU),
                'mean_iou_percent': float(mIoU * 100),
                'mean_f1_score': float(mean_f1),
                'mean_f1_score_percent': float(mean_f1 * 100),
                'mean_precision': float(mean_precision),
                'mean_precision_percent': float(mean_precision * 100),
                'mean_recall': float(mean_recall),
                'mean_recall_percent': float(mean_recall * 100),
            },
            'inference_speed': {
                'fps': float(fps),
                'avg_inference_time_ms': float(avg_inference_time * 1000),
                'std_inference_time_ms': float(std_inference_time * 1000),
                'min_inference_time_ms': float(min_inference_time * 1000),
                'max_inference_time_ms': float(max_inference_time * 1000),
            },
            'per_class_metrics': {}
        }
        
        # Add per-class metrics
        class_names = CITYSCAPES_CLASSES[:self.num_classes] if self.num_classes == 19 else [
            f'class_{i}' for i in range(self.num_classes)
        ]
        
        for i, class_name in enumerate(class_names):
            results['per_class_metrics'][class_name] = {
                'iou': float(iou_per_class[i]),
                'iou_percent': float(iou_per_class[i] * 100),
                'f1_score': float(f1_per_class[i]),
                'f1_score_percent': float(f1_per_class[i] * 100),
                'precision': float(precision_per_class[i]),
                'precision_percent': float(precision_per_class[i] * 100),
                'recall': float(recall_per_class[i]),
                'recall_percent': float(recall_per_class[i] * 100),
            }
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        print('\n' + '='*70)
        print('EVALUATION RESULTS')
        print('='*70)
        
        # Check if there was an error
        if 'error' in results:
            print(f'\nError: {results["error"]}')
            print(f'Note: {results.get("note", "")}')
            return
        
        print(f'\nModel Information:')
        print(f'  Model: {results["model"]}')
        print(f'  Dataset: {results["dataset"]}')
        print(f'  Split: {results["split"]}')
        print(f'  Checkpoint: {results["checkpoint"]}')
        print(f'  Parameters: {results["model_parameters"]["total_millions"]:.2f}M')
        print(f'  Model Size: {results["model_size_mb"]:.2f} MB')
        print(f'\nInput Image Size:')
        print(f'  Actual Size: {results["input_size"]["format"]} (H x W)')
        print(f'  Base Size: {results["input_size"]["base_size"]}')
        print(f'  Crop Size: {results["input_size"]["crop_size"]}')
        
        print(f'\nOverall Metrics:')
        print(f'  Pixel Accuracy: {results["metrics"]["pixel_accuracy_percent"]:.2f}%')
        print(f'  Mean IoU:       {results["metrics"]["mean_iou_percent"]:.2f}%')
        print(f'  Mean F1 Score:  {results["metrics"]["mean_f1_score_percent"]:.2f}%')
        print(f'  Mean Precision: {results["metrics"]["mean_precision_percent"]:.2f}%')
        print(f'  Mean Recall:    {results["metrics"]["mean_recall_percent"]:.2f}%')
        
        print(f'\nInference Speed:')
        print(f'  FPS:            {results["inference_speed"]["fps"]:.2f}')
        print(f'  Avg Time:       {results["inference_speed"]["avg_inference_time_ms"]:.2f} ms')
        print(f'  Std Time:       {results["inference_speed"]["std_inference_time_ms"]:.2f} ms')
        print(f'  Min Time:       {results["inference_speed"]["min_inference_time_ms"]:.2f} ms')
        print(f'  Max Time:       {results["inference_speed"]["max_inference_time_ms"]:.2f} ms')
        
        print(f'\nPer-Class Metrics:')
        print(f'{"Class":<20} {"IoU":<10} {"F1":<10} {"Precision":<12} {"Recall":<10}')
        print('-'*70)
        
        for class_name, metrics in results['per_class_metrics'].items():
            print(f'{class_name:<20} '
                  f'{metrics["iou_percent"]:>6.2f}%  '
                  f'{metrics["f1_score_percent"]:>6.2f}%  '
                  f'{metrics["precision_percent"]:>8.2f}%  '
                  f'{metrics["recall_percent"]:>6.2f}%')
        
        print('='*70)
    
    def save_results(self, results, output_file='evaluation_results.json'):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--eval-split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on (default: val)')
    # Parse known args first to get eval_split
    args, remaining = parser.parse_known_args()
    
    # Now parse the rest using train's parse_args
    import sys
    sys.argv = [sys.argv[0]] + remaining
    train_args = parse_args()
    
    # Merge eval_split into train_args
    train_args.eval_split = args.eval_split
    
    # Override some args for evaluation
    if not train_args.resume:
        print('Warning: --resume is required for evaluation. Exiting.')
        return
    
    evaluator = ComprehensiveEvaluator(train_args)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    
    # Save results only if evaluation was successful
    if results and 'error' not in results:
        split_suffix = train_args.eval_split
        output_file = f'evaluation_results_{train_args.model}_{train_args.dataset}_{split_suffix}.json'
        evaluator.save_results(results, output_file)
        
        # Also save a summary text file
        summary_file = f'evaluation_summary_{train_args.model}_{train_args.dataset}_{split_suffix}.txt'
        with open(summary_file, 'w') as f:
            f.write('='*70 + '\n')
            f.write('EVALUATION SUMMARY\n')
            f.write('='*70 + '\n\n')
            f.write(f'Model: {results["model"]}\n')
            f.write(f'Dataset: {results["dataset"]}\n')
            f.write(f'Split: {results["split"]}\n')
            f.write(f'Checkpoint: {results["checkpoint"]}\n\n')
            f.write('Key Metrics:\n')
            f.write(f'  Pixel Accuracy: {results["metrics"]["pixel_accuracy_percent"]:.2f}%\n')
            f.write(f'  Mean IoU:       {results["metrics"]["mean_iou_percent"]:.2f}%\n')
            f.write(f'  Mean F1 Score:  {results["metrics"]["mean_f1_score_percent"]:.2f}%\n')
            f.write(f'  FPS:            {results["inference_speed"]["fps"]:.2f}\n')
            f.write(f'  Parameters:     {results["model_parameters"]["total_millions"]:.2f}M\n')
            f.write('='*70 + '\n')
        print(f'Summary saved to: {summary_file}')


if __name__ == '__main__':
    main()

