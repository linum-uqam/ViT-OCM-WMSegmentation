import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import sys
import argparse
import cv2
import requests
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import dino.vision_transformer as vits
from skimage import io
import matplotlib.patches as patches
from torchsummary import summary
import copy
from data import AIP_Dataset, Croped_Dataset
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
from glob import glob
from data import build_loader_simmim
from utils import seeding, create_dir
import time
from logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./checkpoints/model.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="/home/mohamad_h/STEGO/STEGO/src/Aips_Guassian+TopHat/imgs/", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.1, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='', type=str, help='tag of the experiment')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='number of warmup epochs to run')
    parser.add_argument('--base_lr', default=1.25e-3, type=float, help='base learning rate')
    parser.add_argument('--warmup_lr', default=2.5e-7, type=float, help='warmup learning rate')
    parser.add_argument('--min_lr', default=2.5e-7, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--layer_decay', default=0.8, type=float, help='layer decay')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--mask_patch_size', default=16, type=int, help='patch size for the mask')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='ratio of the mask')
    args = parser.parse_args()
    return args

def main(args):
    data_loader_train = build_loader_simmim(args)
    logger.info(f"Creating model:{args.arch}/{args.patch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    logger.info(str(model))
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}  
        msg = model.load_state_dict(state_dict)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our dataset.")
        img_name = "brain_02_z15_roi00.jpg"
        directory_path = "/home/mohamad_h/data/40xmosaics_fullsize_subbg/"
        img_path = directory_path + img_name
    else:
        print(f"Provided image path {args.image_path} will be used.")
        img_path = args.image_path

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # summary(model, (3, 800, 800))

if __name__ == '__main__':
    args = parse_option()
    logger = create_logger(output_dir=args.output_dir, name=f"{args.arch}_{args.patch_size}")
    seeding(0)
    create_dir(args.output_dir)
    main(args)
    
