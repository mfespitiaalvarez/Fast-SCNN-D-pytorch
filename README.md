# Fast-SCNN-D: Fast Semantic Segmentation Network with Depth

This repository extends the original [Fast-SCNN](https://arxiv.org/pdf/1902.04502) implementation by adding depth/disparity information for improved semantic segmentation performance.

## About

This project is based on the Fast-SCNN repository by [Tramac](https://github.com/Tramac) and extends it with:
- **RGB-D input support**: Incorporates disparity/depth information alongside RGB images
- **Geometry Feature Generator**: Computes surface normals from disparity in real-time
- **Adaptive Gated Fusion**: Learns to dynamically weight RGB vs depth features
- **Dual-stream architecture**: Processes RGB and geometry features separately before fusion

<p align="center"><img width="100%" src="./png/Fast-SCNN.png" /></p>

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-fast-scnn'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#demo'>Demo</a>
- <a href='#results'>Results</a>
- <a href='#todo'>TO DO</a>
- <a href='#references'>Reference</a>

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- See `requirements.txt` for full dependency list

### Setup
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Fast-SCNN-D-pytorch
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Cityscapes dataset (see [Datasets](#datasets) section below)

### Note
This repository extends the original Fast-SCNN implementation. The base Fast-SCNN code is from [Tramac's repository](https://github.com/Tramac/Fast-SCNN-pytorch).

## Datasets

### Cityscapes
For Fast-SCNN-D, you need:
1. **RGB Images**: [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=4) (11GB)
2. **Ground Truth**: [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB)
3. **Disparity Maps**: [disparity_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) (required for Fast-SCNN-D)

Extract all files to `./datasets/citys/` with the following structure:
```
datasets/citys/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
├── gtFine/
│   ├── train/
│   ├── val/
│   └── test/
└── disparity/
    ├── train/
    ├── val/
    └── test/
```

## Training

### Fast-SCNN (RGB-only baseline)
```bash
python scripts/train.py --model fast_scnn --dataset citys --resume <checkpoint_path>
```

### Fast-SCNN-D (RGB + Depth)
```bash
python scripts/train.py --model fast_scnn_d --dataset citys_d --resume <checkpoint_path>
```

### Training Options
- `--base-size`: Base image size (default: 1024)
- `--crop-size`: Crop size for training (default: 768)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 160)
- `--lr`: Learning rate (default: 0.01)
- `--aux`: Use auxiliary loss
- See `scripts/train.py` for all available options

## Evaluation
To evaluate a trained network:
```Shell
python scripts/eval.py
```

For comprehensive evaluation with detailed metrics:
```Shell
python scripts/evaluate_metrics.py --resume weights/fast_scnn_citys.pth --model fast_scnn --dataset citys
```

## Demo
Running a demo:
```Shell
python scripts/demo.py --model fast_scnn --input-pic './png/berlin_000000_000019_leftImg8bit.png'
```

## Scripts
All utility scripts are located in the `scripts/` directory. See [scripts/README.md](scripts/README.md) for detailed usage:
- Visualization scripts (qualitative comparison, pipeline visualization, ablation plots)
- Evaluation scripts
- Submission preparation
- Data analysis tools

## Results

### Official Cityscapes Test Set Results

**Fast-SCNN-D** on Cityscapes test set (submitted to official evaluation server):

| Metric | Value |
|:------:|:-----:|
| **IoU Classes** | **63.36%** |
| **iIoU Classes** | **33.85%** |
| **IoU Categories** | **84.33%** |
| **iIoU Categories** | **65.02%** |

### Validation Set Results

|Method|Dataset|crop_size|mIoU|Pixel Acc|FPS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Fast-SCNN (RGB-only)|Cityscapes|768|49.21%|~92%|331.8|
|Fast-SCNN-D (RGB + Depth)|Cityscapes-D|768|58.96%|~94%|235.1|

**Improvement**: Fast-SCNN-D achieves **+9.75% mIoU** improvement over the RGB-only baseline on validation set while maintaining real-time inference speed (>200 FPS).

### Per-Class Performance (Test Set)

Top performing classes:
- **Road**: 97.32% IoU
- **Sky**: 93.30% IoU  
- **Car**: 91.49% IoU
- **Vegetation**: 90.52% IoU
- **Building**: 89.02% IoU

Note: Test set results from official Cityscapes evaluation server. Validation results based on crop_size=768. See ablation study for component-wise contributions.

<img src="./png/frankfurt_000001_058914_leftImg8bit.png" width="280" /><img src="./png/frankfurt_000001_058914_gtFine_color.png" width="280" /><img src="./png/frankfurt_000001_058914_seg.png" width="280" />
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(a) test image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) ground truth &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(c) predicted result

## Key Features

- **Dual-Stream Architecture**: Separate processing paths for RGB and geometry features
- **Geometry Feature Generator**: Real-time computation of surface normals from disparity
- **Adaptive Gated Fusion**: Spatially-adaptive weighting of RGB vs depth features
- **Real-time Performance**: Maintains >200 FPS inference speed
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## Project Structure

```
Fast-SCNN-D-pytorch/
├── scripts/              # All scripts (training, evaluation, visualization, submission)
├── models/               # Model definitions (Fast-SCNN, Fast-SCNN-D)
├── data_loader/          # Dataset loaders (Cityscapes, Cityscapes-D)
├── utils/                # Utilities (metrics, loss, visualization)
└── datasets/             # Dataset directory (download Cityscapes here)
```

## Citation

If you use this code, please cite:

```bibtex
@article{poudel2019fast,
  title={Fast-SCNN: Fast Semantic Segmentation Network},
  author={Poudel, Rudra PK and Liwicki, Stephan and Cipolla, Roberto},
  journal={arXiv preprint arXiv:1902.04502},
  year={2019}
}
```

And if you use the Fast-SCNN-D extensions:
```bibtex
@misc{fastscnnd2024,
  title={Fast-SCNN-D: Fast Semantic Segmentation with Depth},
  author={Your Name},
  year={2024}
}
```

## Acknowledgments

- **Base Implementation**: This repository extends [Fast-SCNN-pytorch](https://github.com/Tramac/Fast-SCNN-pytorch) by [Tramac](https://github.com/Tramac)
- **Original Paper**: "Fast-SCNN: Fast Semantic Segmentation Network" by Rudra PK Poudel, Stephan Liwicki, Roberto Cipolla

## License

See [LICENSE](LICENSE) file for details.
