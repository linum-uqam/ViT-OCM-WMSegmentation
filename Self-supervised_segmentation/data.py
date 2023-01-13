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
    def __init__(self, images_path):

        self.images_path = images_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        return image

    def __len__(self):
        return self.n_samples