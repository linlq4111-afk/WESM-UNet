from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from torchvision import transforms


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self).__init__()
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            # 使用验证集作为验证数据
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        if self.transformer is not None:
            img, msk = self.transformer((img, msk))  # 传递元组
        # mask只做resize（NEAREST），不做ToTensor和normalize（如果transformer已做可省略）
        if not isinstance(msk, torch.Tensor):
            msk = msk.resize((256, 256), Image.NEAREST)
            msk = np.array(msk, dtype=np.uint8)
            msk = torch.from_numpy(msk).long()  # shape: (H, W)
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 基础几何增强 (与msvm-unet-main保持一致)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 随机缩放 (类似msvm-unet-main的Affine scale)
        if random.random() < 0.3:
            scale = 0.5 + random.random() * 1.5  # [0.5, 2.0] 与msvm-unet-main一致
            try:
                image = zoom(image, scale, order=3)
                label = zoom(label, scale, order=0)
            except Exception:
                pass

        # 随机平移 (类似msvm-unet-main的translate)
        if random.random() < 0.3:
            max_shift_ratio = 0.2  # 20%，与msvm-unet-main一致
            h, w = image.shape
            shift_y = int((random.random() * 2 - 1) * max_shift_ratio * h)
            shift_x = int((random.random() * 2 - 1) * max_shift_ratio * w)
            try:
                image = ndimage.shift(image, shift=(shift_y, shift_x), order=3, mode='reflect')
                label = ndimage.shift(label, shift=(shift_y, shift_x), order=0, mode='constant', cval=0)
            except Exception:
                pass

        # 强度增强 (与msvm-unet-main保持一致)
        try:
            # 对比度 (LinearContrast 0.5-1.5)
            if random.random() < 0.3:
                contrast = 0.5 + random.random()  # [0.5, 1.5] 与msvm-unet-main一致
                image = np.clip(image * contrast, 0.0, 1.0)

            # 高斯噪声 (AdditiveGaussianNoise scale=0.005*255)
            if random.random() < 0.2:
                noise_std = 0.005  # 与msvm-unet-main一致
                noise = np.random.normal(0.0, noise_std, size=image.shape)
                image = np.clip(image + noise, 0.0, 1.0)

            # 高斯模糊 (GaussianBlur sigma=1.0)
            if random.random() < 0.2:
                sigma = 1.0  # 与msvm-unet-main一致
                image = ndimage.gaussian_filter(image, sigma=sigma)
                image = np.clip(image, 0.0, 1.0)
        except Exception:
            pass

        # 尺寸调整
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 确保图像在[0,1]范围内
        image = np.clip(image, 0.0, 1.0)
        
        # 转换为tensor并重复为3通道
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = image.repeat(3, 1, 1)
        
        # 从头训练：使用简单的标准化，不使用ImageNet预处理
        # 将[0,1]范围映射到[-1,1]范围，类似于msvm-unet-main的处理
        image = image * 2.0 - 1.0
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    