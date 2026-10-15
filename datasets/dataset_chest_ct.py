import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

class ChestCT_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split="train", transform=None, input_size=(224, 224)):
        self.base_dir = base_dir # Primary dir, but we will look at others too
        self.list_dir = list_dir 
        self.split = split
        self.transform = transform
        self.input_size = input_size
        
        # Define all source directories
        self.source_dirs = [
            './data/chest_ct',
            './data/chest_ct',
            './data/chest_ct'
        ]
        
        self.sample_list = []
        for d_path in self.source_dirs:
            if os.path.exists(d_path):
                # Get full paths
                patients = sorted([os.path.join(d_path, d) for d in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, d))])
                self.sample_list.extend(patients)
        
        # Simple split: 80% train, 20% val
        random.seed(42)
        # random.shuffle(self.sample_list) 
        
        split_idx = int(len(self.sample_list) * 0.8)
        if split == "train":
            self.sample_list = self.sample_list[:split_idx]
        else:
            self.sample_list = self.sample_list[split_idx:]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # patient_path is now the full path
        patient_dir = self.sample_list[idx]
        patient_name = os.path.basename(patient_dir)
        
        # 1. Find Raw Image
        raw_vol_dir = os.path.join(patient_dir, 'RawVolume')
        image = np.zeros((512, 512), dtype=np.float32)
        
        if os.path.exists(raw_vol_dir):
            raw_subdirs = [d for d in os.listdir(raw_vol_dir) if os.path.isdir(os.path.join(raw_vol_dir, d))]
            
            valid_raw_path = None
            for d in raw_subdirs:
                if "Resampling" not in d:
                    valid_raw_path = os.path.join(raw_vol_dir, d)
                    break
            
            if valid_raw_path is None and len(raw_subdirs) > 0:
                 valid_raw_path = os.path.join(raw_vol_dir, raw_subdirs[0])

            if valid_raw_path:
                nii_files = [f for f in os.listdir(valid_raw_path) if f.endswith('.nii.gz')]
                if nii_files:
                    img_path = os.path.join(valid_raw_path, nii_files[0])
                    img_obj = nib.load(img_path)
                    img_data = img_obj.get_fdata()
                    if img_data.ndim == 3:
                        image = img_data[:, :, 0]
                    else:
                        image = img_data

        # 2. Find Masks
        roi_vol_dir = os.path.join(patient_dir, 'RoiVolume')
        mask = np.zeros_like(image, dtype=np.uint8)
        
        roi_map = {
            'R1.uint16.nii.gz': 1,
            'R2.uint16.nii.gz': 2,
            'R4.uint16.nii.gz': 3,
            'R8.uint16.nii.gz': 4
        }
        
        if os.path.exists(roi_vol_dir):
            for roi_file, label_idx in roi_map.items():
                roi_path = os.path.join(roi_vol_dir, roi_file)
                if os.path.exists(roi_path):
                    roi_obj = nib.load(roi_path)
                    roi_data = roi_obj.get_fdata()
                    if roi_data.ndim == 3:
                        roi_data = roi_data[:, :, 0]
                    
                    if roi_data.shape != image.shape:
                        roi_data = zoom(roi_data, (image.shape[0]/roi_data.shape[0], image.shape[1]/roi_data.shape[1]), order=0)
                        
                    mask[roi_data > 0] = label_idx

        # 3. Preprocessing
        # Normalize to [0, 1]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)

        sample = {'image': image, 'label': mask}
        
        if self.transform:
            sample = self.transform(sample)
        else:
            # Resize
            if image.shape[0] != self.input_size[0] or image.shape[1] != self.input_size[1]:
                image = zoom(image, (self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1]), order=3)
                mask = zoom(mask, (self.input_size[0] / mask.shape[0], self.input_size[1] / mask.shape[1]), order=0)
            
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
            image = image * 2.0 - 1.0
            label = torch.from_numpy(mask.astype(np.float32)).long()
            sample = {'image': image, 'label': label}

        sample['case_name'] = patient_name
        return sample
