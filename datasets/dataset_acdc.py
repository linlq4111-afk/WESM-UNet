import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy import ndimage
import random
import imgaug.augmenters as iaa
import imgaug as ia
import h5py

# Fix for numpy 1.24+ where np.bool is removed
if not hasattr(np, 'bool'):
    np.bool = bool

class ACDCDataset(Dataset):
    def __init__(self, base_dir, list_dir, split="train", transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir
        
        # 使用与msvm-unet-main完全相同的数据增强策略
        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=1.0),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def mask_to_onehot(self, mask):
        """Converts a segmentation mask to one-hot encoding for ACDC (4 classes)"""
        semantic_map = []
        mask = np.expand_dims(mask, -1)
        for colour in range(4):  # ACDC has 4 classes
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
        return semantic_map

    def augment_seg(self, img, seg):
        """Apply augmentation to both image and segmentation mask"""
        seg = self.mask_to_onehot(seg)
        aug_det = self.img_aug.to_deterministic()
        image_aug = aug_det.augment_image(img)

        seg_map = ia.SegmentationMapsOnImage(seg, shape=img.shape)
        seg_map_aug = aug_det.augment_segmentation_maps(seg_map)
        seg_map_aug = seg_map_aug.get_arr()
        seg_map_aug = np.argmax(seg_map_aug, axis=-1).astype(np.float32)
        return image_aug, seg_map_aug

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, "training_npz", slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
            
            # 训练时使用与msvm-unet-main相同的数据增强
            image, label = self.augment_seg(image, label)
            
            # Resize到224x224
            h, w = image.shape
            if h != 224 or w != 224:
                image = zoom(image, (224 / h, 224 / w), order=3)
                label = zoom(label, (224 / h, 224 / w), order=0)

            sample = {'image': image, 'label': label}
            
            # 标准化处理：[0,1] -> [-1,1] 并转换为3通道
            image = (image - image.min()) / (image.max() - image.min())
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image = image.repeat(3, 1, 1)  # 转换为3通道
            image = image * 2.0 - 1.0  # [0,1] -> [-1,1]
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
            
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
            
        else:
            # 验证/测试集
            vol_name = self.sample_list[idx].strip('\n')
            
            # 检查是否是2D切片文件名 (e.g., case_081_sliceED_0.npz)
            # 如果是2D切片名，说明我们在使用valid.txt进行验证
            # 此时应该回退到加载2D切片的方式，或者跳过（因为我们想做3D验证）
            # 但为了兼容性，如果找不到h5文件，我们尝试加载npz
            
            # 移除可能的后缀
            clean_name = vol_name
            if clean_name.endswith('.npz'):
                clean_name = clean_name[:-4]
            if clean_name.endswith('.h5'):
                clean_name = clean_name[:-3]
                
            # 优先尝试加载3D h5文件 (用于测试集 test_volumes.txt)
            h5_path = os.path.join(self.data_dir, "testing_h5", f"{clean_name}.h5")
            
            if os.path.exists(h5_path):
                # 加载3D Volume
                data = h5py.File(h5_path, 'r')
                image, label = data['image'][:], data['label'][:]
                
                # 如果数据是 (H, W, D) 格式，转换为 (D, H, W)
                if image.ndim == 3 and image.shape[-1] < image.shape[0]:
                    image = image.transpose(2, 0, 1)
                    label = label.transpose(2, 0, 1)
                
                sample = {'image': image, 'label': label}
                sample['case_name'] = clean_name
                return sample
            else:
                # 如果找不到h5文件，说明可能是valid.txt中的2D切片名
                # 回退到加载2D npz文件
                if not vol_name.endswith('.npz'):
                    vol_name += '.npz'
                
                data_path = os.path.join(self.data_dir, "training_npz", vol_name)
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Could not find data for {vol_name} in either testing_h5 or training_npz")

                data = np.load(data_path)
                image, label = data['img'], data['label']

                # Resize到224x224
                h, w = image.shape
                if h != 224 or w != 224:
                    image = zoom(image, (224 / h, 224 / w), order=3)
                    label = zoom(label, (224 / h, 224 / w), order=0)

                # 对于2D切片验证，我们返回 (H, W) 格式，不进行重复和归一化到[-1,1]
                # 因为 test_single_volume 会处理这些
                # 但我们需要确保它是 [0, 1] 范围
                image = (image - image.min()) / (image.max() - image.min())
                
                # 转换为Tensor
                image = torch.from_numpy(image.astype(np.float32))
                label = torch.from_numpy(label.astype(np.float32))
                
                sample = {'image': image, 'label': label.long()}
                sample['case_name'] = vol_name.replace('.npz', '')
                return sample


