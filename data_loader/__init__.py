from .cityscapes import CitySegmentation
from .cityscapes_d import CitySegmentation as CitySegmentationD

datasets = {
    'citys': CitySegmentation,
    'citys_d': CitySegmentationD,  # Disparity-enabled version
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
