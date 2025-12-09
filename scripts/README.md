# Scripts Directory

This directory contains utility scripts for the Fast-SCNN-D project.

## Visualization Scripts

### `create_qualitative_plot.py`
Creates side-by-side qualitative comparison of Fast-SCNN vs Fast-SCNN-D predictions.

**Usage:**
```bash
python scripts/create_qualitative_plot.py \
    --resume-fast-scnn weights/fast_scnn_citys.pth \
    --resume-fast-scnn-d weights/fast_scnn_d_citys_d.pth \
    --num-examples 4 \
    --seed 42
```

### `create_pipeline_visualization.py`
Visualizes the Fast-SCNN-D pipeline stages: input RGB, disparity, generated normals, gated fusion alpha, and final output.

**Usage:**
```bash
python scripts/create_pipeline_visualization.py \
    --resume weights/fast_scnn_d_citys_d.pth \
    --num-examples 4 \
    --seed 42
```

### `visualize_alpha_normals.py`
Creates visualizations for alpha trust analysis and surface normals.

**Usage:**
```bash
python scripts/visualize_alpha_normals.py \
    --resume weights/fast_scnn_d_citys_d.pth \
    --alpha-only  # or --normals-only
```

### `create_ablation_plot.py`
Creates ablation study bar charts showing contribution of each component.

**Usage:**
```bash
python scripts/create_ablation_plot.py \
    --results-rgb evaluation_results_fast_scnn_citys_val.json \
    --results-full evaluation_results_fast_scnn_d_citys_d_val.json
```

### `create_paper_plots.py`
Creates comprehensive plots for paper figures (accuracy vs speed, per-class IoU, etc.).

**Usage:**
```bash
python scripts/create_paper_plots.py \
    --results-fast-scnn evaluation_results_fast_scnn_citys_val.json \
    --results-fast-scnn-d evaluation_results_fast_scnn_d_citys_d_val.json
```

## Evaluation Scripts

### `evaluate_metrics.py`
Comprehensive evaluation script computing mIoU, pixel accuracy, F1 score, FPS, etc.

**Usage:**
```bash
python scripts/evaluate_metrics.py \
    --resume weights/fast_scnn_d_citys_d.pth \
    --model fast_scnn_d \
    --dataset citys_d \
    --eval-split val
```

### `eval.py`
Simple evaluation script for quick testing.

**Usage:**
```bash
python scripts/eval.py \
    --resume weights/fast_scnn_d_citys_d.pth \
    --model fast_scnn_d \
    --dataset citys_d
```

## Utility Scripts

### `prepare_submission.py`
Prepares Cityscapes server submission with correct format and labelIDs.

**Usage:**
```bash
python scripts/prepare_submission.py \
    --resume weights/fast_scnn_d_citys_d.pth \
    --model fast_scnn_d \
    --dataset citys_d
```

### `calculate_disparity_stats.py`
Calculates statistics on raw disparity data for normalization.

**Usage:**
```bash
python scripts/calculate_disparity_stats.py
```

### `demo.py`
Demo script for single image prediction.

**Usage:**
```bash
python scripts/demo.py \
    --model fast_scnn_d \
    --input-pic path/to/image.png
```

## Notes

- All scripts automatically add the parent directory to `sys.path` so imports work correctly
- Run scripts from the project root directory
- Use `--seed` argument for reproducibility where available

