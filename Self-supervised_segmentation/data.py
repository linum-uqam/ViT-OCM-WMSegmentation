from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
from utils import seeding
import torchvision.transforms as T
import os
import cv2
import copy
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from glob import glob


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

class AIP_Masked_Dataset(Dataset):
    def __init__(self, images_path, transform=None):

        self.images_path = images_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        """ Reading image """
        img = Image.open(self.images_path[index]).convert('RGB')
        img, mask = self.transform(img)
        w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - img.shape[2] % 8
        img = img[:, :w, :h]

        return img, mask

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


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:

    def __init__(self, args):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(args.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
        ])
 
        
        model_patch_size=args.MODEL.PATCH_SIZE
        self.image_size = args.DATA.IMG_SIZE[0]
        self.mask_generator = MaskGenerator(
            input_size=args.DATA.IMG_SIZE[0],
            mask_patch_size=args.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=args.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        scaled_mask = copy.deepcopy(mask)
        # 14*14 -> 224*224
        scaled_mask = np.repeat(scaled_mask, 16, axis=0).repeat(16, axis=1)
        # to image
        scaled_mask = scaled_mask * 255
        scaled_mask = scaled_mask.astype(np.uint8)
        scaled_mask = cv2.resize(scaled_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # convert img to numpy
        source = copy.deepcopy(img)
        source = source.numpy()
        source = np.transpose(source, (1, 2, 0))
        source = source * 255
        source = source.astype(np.uint8)
        #save 
        cv2.imwrite(os.path.join("/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output/", "img.png"), source)
        # save in output folder
        cv2.imwrite(os.path.join("/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output/", "mask.png"), scaled_mask)


        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(args):
    if os.path.isfile(args.DATA.IMAGE_PATH):
        train_x = sorted(glob(args.DATA.IMAGE_PATH))
    else:
        train_x = sorted(glob(args.DATA.IMAGE_PATH + "/*.jpg")) 
    transform = SimMIMTransform(args)
    # logger.info(f'Pre-train data transform:\n{transform}')

    # dataset = ImageFolder(args.DATA.IMAGE_PATH, transform)
    dataset = AIP_Masked_Dataset(train_x, transform)
    # logger.info(f'Build dataset: train images = {len(dataset)}')
    
    # sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, args.DATA.BATCH_SIZE, num_workers=args.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader