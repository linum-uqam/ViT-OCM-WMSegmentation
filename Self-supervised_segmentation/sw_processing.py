
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import argparse
import dino.vision_transformer as vits
from utils import compute_attention, create_dir, threshold
import matplotlib.pyplot as plt
import cv2
def concat_crops(crops, stride, window_size):
    crop_number = len(crops)
    crop_iteration = int(np.sqrt(crop_number))
    vertical = []
    # stride = 128
    step = window_size - stride 
    for i in range(crop_iteration):
        horizontal = crops[i * crop_iteration]
        for j in range(1, crop_iteration):
            # Concatenate the current crop with the previous crop horizontally
            left_window = horizontal
            right_window = crops[i * crop_iteration + j]
            crop_left = left_window[:, :-step]
            crop_right = right_window[ :, -stride:]
            overlap = (left_window[:, -step:]// 2 + right_window[ :, :-stride]// 2) 
            horizontal = np.concatenate((crop_left, overlap, crop_right), axis=1)
            
        if i == 0:
            # First row
            vertical = horizontal
        else:
            # Middle rows
            top_overlap = (vertical[-step:, :]// 2 + horizontal[:-stride, :]// 2) 
            vertical = np.concatenate((vertical[:-step, :], top_overlap,horizontal[-stride:, :]), axis=0)
    
    return vertical


# Define the window size and stride


def sliding_window(image, stride=128, window_size=384):
    crops = []
    height, width = image.size

    # Iterate over the windows
    for y in range(0, height - stride*2 , stride):
        for x in range(0, width - stride*2 , stride):
            # Crop the window and apply the transformation
            window = image.crop((x, y, x + window_size, y + window_size))
            window = np.array(window)
            crops.append(window)

    return crops
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output/vit_small/VIT_8_AM_384_16B_0.3R_8MP/ckpt_epoch_5.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_size", default=(1152, 1152), type=int, nargs="+", help="Resize image.")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}  
        msg = model.load_state_dict(state_dict["model"], strict=False)
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

    # Example usage
    to_be_processed_img = Image.open('/home/mohamad_h/data/Data_OCM_ALL/brain_06_z04_roi03.jpg').convert('RGB')
    to_be_processed_img = to_be_processed_img.resize((1152, 1152))
    window_size = 384
    stride = 128
    cropped_images = sliding_window(to_be_processed_img, stride, window_size)
    print(len(cropped_images))                
    output_image = concat_crops(cropped_images, stride, window_size) 
    # save the output image which is a numpy array
    im = Image.fromarray(output_image).convert('RGB')
    im.save('temp/attention_map_sw.jpg')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    average_crops = []
    ouput = "temp/"
    create_dir(ouput)
    for j in range(len(cropped_images)):
        crop = cropped_images[j]
        crop = transform(crop).unsqueeze(0)
        crop = crop.cuda(non_blocking=True)
        feat, attentions, qkv = model.get_intermediate_feat(crop.to(device), n=1)
        query = 0
        w_featmap = crop.shape[-2] // args.patch_size
        h_featmap = crop.shape[-1] // args.patch_size
        attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
        # average attentions over heads
        average_attentions = np.mean(attention_response, axis=0)
        fname = 'temp/attention_map_sw_{}.jpg'.format(j)
        plt.imsave(fname, average_attentions)
        # median filter scipy
        # average_attentions = median_filter(average_attentions, size=args.median_filter)
        # convert torch to numpy array
        average_attentions = np.array(average_attentions)
        # make between 0 and 255
        average_attentions = (average_attentions - average_attentions.min()) / (average_attentions.max() - average_attentions.min())
        average_attentions = average_attentions * 255
        average_attentions = cv2.resize(average_attentions, (average_attentions.shape[1]//8, average_attentions.shape[0]//8))
        # interpolate the attention map to the original size with bicubic interpolation
        average_attentions = cv2.resize(average_attentions, (average_attentions.shape[0]*8, average_attentions.shape[0]*8), interpolation=cv2.INTER_LINEAR)
        average_crops.append(average_attentions)
    average_attentions = concat_crops(average_crops, stride, window_size)
    fname = 'attention_map_sw.jpg'
    plt.imsave(fname, average_attentions)
    output, original_otsu, heatmap_otsu = threshold(im.convert("L") , average_attentions, save=True)
    # fname = 'otsu_sw.jpg'
    # plt.imsave(fname, output)
    fname = 'otsu_sw_heatmap.jpg'
    plt.imsave(fname, heatmap_otsu, cmap='gray')


