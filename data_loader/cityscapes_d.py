"""Cityscapes Dataloader with RGB-D Support"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2 

from PIL import Image, ImageOps, ImageFilter

__all__ = ['CitySegmentation']


class CitySegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset (RGB-D)."""
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19
    
    # CALCULATED STATS
    RAW_MEAN_INT = 9070      # For filling holes/padding (Raw 16-bit)
    REAL_MEAN = 35.42        # For Normalization (Real Disparity)
    REAL_STD = 28.04         # For Normalization (Real Disparity)

    def __init__(self, root='./datasets/citys', split='train', mode=None, transform=None,
                 base_size=520, crop_size=480, **kwargs):
        super(CitySegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        
        self.images, self.mask_paths, self.disp_paths = _get_city_pairs(self.root, self.split)
        
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
            
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        # 1. Load RGB
        img = Image.open(self.images[index]).convert('RGB')
        
        # 2. Load Disparity (16-bit)
        disp_path = self.disp_paths[index]
        disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # --- STEP 1: FILL HOLES ---
        # Replace invalid pixels (0) with the DATASET RAW MEAN (9070)
        mask_valid = disp_raw > 0
        disp_raw[~mask_valid] = self.RAW_MEAN_INT
        
        # Convert to PIL for consistent transforms
        disp_img = Image.fromarray(disp_raw)

        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
            
        mask = Image.open(self.mask_paths[index])

        # Synchronized transform (RGB + Disparity + Mask)
        if self.mode == 'train':
            img, mask, disp_img = self._sync_transform(img, mask, disp_img)
        elif self.mode == 'val':
            img, mask, disp_img = self._val_sync_transform(img, mask, disp_img)
        else:
            img, mask = self._img_transform(img), self._mask_transform(mask)
            disp_img = disp_img.resize(img.size, Image.NEAREST)

        # --- STEP 2: NORMALIZE DISPARITY ---
        disp_np = np.array(disp_img).astype(np.float32)
        
        # A. Convert Raw to Real Disparity: d = (raw - 1) / 256.0
        disp_np = (disp_np - 1.0) / 256.0
        
        # B. Z-Score Standardization using calculated stats
        # Mean = 35.42, Std = 28.04
        disp_np = (disp_np - self.REAL_MEAN) / self.REAL_STD
        
        # Convert to Tensor [1, H, W]
        disp_tensor = torch.from_numpy(disp_np).unsqueeze(0)

        # 4. Transform RGB
        if self.transform is not None:
            img = self.transform(img) # Returns Tensor [3, H, W]

        # 5. Concatenate to make [4, H, W]
        input_tensor = torch.cat([img, disp_tensor], dim=0)

        return input_tensor, mask

    def _val_sync_transform(self, img, mask, disp):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        disp = disp.resize((ow, oh), Image.NEAREST)
        
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        disp = disp.crop((x1, y1, x1 + outsize, y1 + outsize))
        
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask, disp

    def _sync_transform(self, img, mask, disp):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            disp = disp.transpose(Image.FLIP_LEFT_RIGHT)
            
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        disp = disp.resize((ow, oh), Image.NEAREST)
        
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            
            # --- STEP 3: PAD DISPARITY WITH MEAN ---
            # Use 9070 so padded regions look like "average depth"
            disp = ImageOps.expand(disp, border=(0, 0, padw, padh), fill=self.RAW_MEAN_INT)
            
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        disp = disp.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # gaussian blur (RGB Only)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask, disp

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder, disp_folder):
        img_paths = []
        mask_paths = []
        disp_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    
                    dispname = filename.replace('leftImg8bit', 'disparity')
                    disppath = os.path.join(disp_folder, foldername, dispname)
                    
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath) and os.path.isfile(disppath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                        disp_paths.append(disppath)
                    else:
                        print('cannot find pairs:', imgpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths, disp_paths

    if split in ('train', 'val', 'test'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        disp_folder = os.path.join(folder, 'disparity/' + split)
        return get_path_pairs(img_folder, mask_folder, disp_folder)
    return [], [], []