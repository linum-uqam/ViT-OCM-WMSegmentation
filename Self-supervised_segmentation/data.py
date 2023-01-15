from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
from utils import seeding


class AIP_Dataset(Dataset):
    def __init__(self, images_path, transform=None):

        self.images_path = images_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        img = Image.open(self.images_path[index]).convert('RGB')
        img = self.transform(img)
        w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - img.shape[2] % 8
        img = img[:, :w, :h]

        return img, self.images_path[index]

    def __len__(self):
        return self.n_samples

class Croped_Dataset(Dataset):
    def __init__(self, images_path, transform=None,crop = 4,image_size = (800,800)):

        self.images_path = images_path
        self.n_samples = len(images_path)
        self.transform = transform
        self.crop_rate = np.int(np.sqrt(crop))
        self.image_size = image_size

    def __getitem__(self, index):
        """ Reading image """
        img = Image.open(self.images_path[index]).convert('RGB')
        images = []
        to_be_croped_img = img.copy()
        to_be_croped_img = to_be_croped_img.resize(self.image_size)
        img = self.transform(img)
        if self.crop_rate == 2:
            w, h = img.shape[1] - img.shape[1] % 32, img.shape[2] - img.shape[2] % 32
        else:
            w, h = img.shape[1] - img.shape[1] % 64, img.shape[2] - img.shape[2] % 64
        w, h = w // self.crop_rate, h // self.crop_rate
        # Brake the image into 4 equal crops and stack them in a list of images
        w, h = to_be_croped_img.size[0] // self.crop_rate, to_be_croped_img.size[1] // self.crop_rate
        for i in range(self.crop_rate):
            for j in range(self.crop_rate):
                x = to_be_croped_img.crop((j * w, i * h, (j + 1) * w, (i + 1) * h))
                x = self.transform(x)
                images.append(x)

        return img, self.images_path[index],images

    def __len__(self):
        return self.n_samples