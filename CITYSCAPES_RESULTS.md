# Cityscapes Evaluation Results

## Official Test Set Results

**Method**: Fast-SCNN-D  
**Submission Date**: 2025-12-09  
**Challenge**: Pixel-level Semantic Labeling  
**Used Data**: Cityscapes fine annotations 

### Overall Metrics

| Metric | Value |
|:------:|:-----:|
| **IoU Classes** | **63.36%** |
| **iIoU Classes** | **33.85%** |
| **IoU Categories** | **84.33%** |
| **iIoU Categories** | **65.02%** |

### Per-Class Results

| Class | IoU | iIoU |
|:------|:---:|:----:|
| road | 97.32% | - |
| sidewalk | 77.71% | - |
| building | 89.02% | - |
| wall | 47.81% | - |
| fence | 40.39% | - |
| pole | 45.14% | - |
| traffic light | 51.95% | - |
| traffic sign | 60.19% | - |
| vegetation | 90.52% | - |
| terrain | 67.98% | - |
| sky | 93.30% | - |
| person | 72.11% | 46.40% |
| rider | 43.72% | 19.86% |
| car | 91.49% | 84.16% |
| truck | 39.74% | 15.41% |
| bus | 52.91% | 31.20% |
| train | 46.43% | 19.18% |
| motorcycle | 38.65% | 16.79% |
| bicycle | 57.38% | 37.83% |

### Per-Category Results

| Category | IoU | iIoU |
|:---------|:---:|:----:|
| flat | 98.07% | - |
| nature | 90.16% | - |
| object | 53.93% | - |
| sky | 93.30% | - |
| construction | 89.43% | - |
| human | 74.50% | 48.61% |
| vehicle | 90.89% | 81.44% |

## Notes

- Results obtained from official Cityscapes evaluation server
- Test set contains 1525 images
- All predictions submitted in correct format (2048×1024, labelIDs)
- Model maintains real-time inference speed (>200 FPS)

