
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
from utils import threshold, compute_attention,create_dir, execution_time,concat_crops, yen_threshold, morphology_cleaning
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./checkpoints/model.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(720, 720), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/images/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.1, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    # Boolyan for croping 
    parser.add_argument("--crop", type=int, default=1, help="""Amount of croping (4 or 16)""")
    # Attention query analysis mode boolean
    parser.add_argument("--query_analysis", type=bool, default=False, help="""To analyze the attention query or not""")
    # query rate parameter defaul is 10
    parser.add_argument("--query_rate", type=int, default=10, help="""Rate of the query analysis""")
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
    # summary(model, (3, 800, 800))
    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        #img = Image.open(BytesIO(response.content))
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
    croped_transform = pth_transforms.Compose([
        pth_transforms.Resize((args.image_size[0]//np.int8(np.sqrt(args.crop)), args.image_size[1]//np.int8(np.sqrt(args.crop)))),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def train(model, loader, device):
        for image, img_path in loader:
            with torch.no_grad():
                feat, attentions, qkv = model.get_intermediate_feat(image.to(device), n=1)
                torch.cuda.empty_cache()

                
                original_img = Image.open(img_path[0]).convert('RGB')
                # resize to ars.image_size
                original_img = original_img.resize(args.image_size, Image.Resampling.BICUBIC)
                # to grayscale
                original_img = original_img.convert('L')
                image_name = img_path[0].split("/")[-1].split(".")[0]
                output_directory = args.output_dir + image_name + "/"
                create_dir(output_directory)
                query = 0
                w_featmap = image.shape[-2] // args.patch_size
                h_featmap = image.shape[-1] // args.patch_size
                attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
                # average attentions over heads
                average_attentions = np.mean(attention_response, axis=0)
                # median filter scipy
                average_attentions = median_filter(average_attentions, size=20)
                # save original image
                torchvision.utils.save_image(torchvision.utils.make_grid(image, normalize=True, scale_each=True), os.path.join(output_directory, "img.png"))
                # save attention maps for each head
                for j in range(nh):
                    fname = os.path.join(output_directory, "attn-head" + str(j) + ".png")
                    plt.imsave(fname=fname, arr=attention_response[j], format='png')
                    print(f"{fname} saved.")
                # save average attention
                fname = os.path.join(output_directory, "attn-average.png")
                plt.imsave(fname=fname, arr=average_attentions, format='png')
                print(f"{fname} saved.")
                # save thresholded attention
                if args.threshold is not None:
                    threshold(original_img, average_attentions, output_directory)
                    binary = yen_threshold(original_img, output_directory, save=True)
                    query_points = morphology_cleaning(binary, output_directory, save=True)
                # save query analysis
                if args.query_analysis:
                    path = args.output_dir + image_name + "/analysis/"
                    create_dir(path)
                    print("Saving query analysis")
                    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                    
                    for i in range(0, w_featmap//args.query_rate):
                        for j in range(0, h_featmap//args.query_rate):
                            query = i * w_featmap*args.query_rate + j*args.query_rate
                            print(f"query: {query}")
                            attentions_reshaped, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
                            average_attentions = np.mean(attentions_reshaped, axis=0) 
                            fname = os.path.join(path, f"attn-average-{query}.png")
                            fig, ax = plt.subplots(figsize=(800*px, 800*px))
                            ax.axis('off')   
                            ax.imshow(average_attentions, aspect='auto')
                            if query != 0: 
                                square = patches.Rectangle((j*args.patch_size*args.query_rate,i*args.patch_size*args.query_rate), 8,8, color='RED')
                                ax.add_patch(square)
                            fig.savefig(fname ,bbox_inches='tight', pad_inches=0)
                            plt.close(fig)
                    print("finished saving query analysis")
        return

    def train_croped(model, loader, device):
        for image, img_path,images in loader:
            with torch.no_grad():
                averaged_cropes = []
                thresholded_crops = []
                image_name = img_path[0].split("/")[-1].split(".")[0]
                output_directory = args.output_dir + image_name + f"/croped_{args.crop}/"
                create_dir(output_directory)
                for i in range(len(images)):
                    img = images[i]
                    img = img.to(device)
                    feat, attentions, qkv = model.get_intermediate_feat(img, n=1)
                    torch.cuda.empty_cache()

                    query = 0
                    w_featmap = img.shape[-2] // args.patch_size
                    h_featmap = img.shape[-1] // args.patch_size
                    attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
                    # average attentions over heads
                    average_attentions = np.mean(attention_response, axis=0)
                    # median filter scipy
                    average_attentions = median_filter(average_attentions, size=20)
                    averaged_cropes.append(average_attentions)

                    # call threshold function on all crops in the images list
                    # convert tensor to numpy array
                    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img = img * 255
                    img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                    img = img.convert('L')
                    th1, th2 = threshold(img, average_attentions, args.output_dir, save=False)
                    thresholded_crops.append(th1)

                reconstructed_average = concat_crops(averaged_cropes)

                reconstructed_thresholded_image = concat_crops(thresholded_crops)
                # save thresholded image
                fname = os.path.join(output_directory, "reconstructed_threshold.png")
                plt.imsave(fname=fname, arr=reconstructed_thresholded_image, format='png',cmap='gray')
                print(f"{fname} saved.")

                original_img = Image.open(img_path[0]).convert('RGB')
                # resize to ars.image_size
                original_img = original_img.resize(args.image_size, Image.Resampling.BICUBIC)
                # to grayscale
                original_img = original_img.convert('L')

                original_img.save(os.path.join(output_directory, "img.png"))
                print("img.png saved.")
                # save average attention
                fname = os.path.join(output_directory, "attn-average.png")
                plt.imsave(fname=fname, arr=reconstructed_average, format='png')
                print(f"{fname} saved.")
                # save thresholded attention
                if args.threshold is not None:
                    threshold(original_img, reconstructed_average, output_directory)


                if args.query_analysis:
                    # path = args.output_dir + image_name + "/analysis/"
                    # create_dir(path)
                    # print("Saving query analysis")
                    # px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                    
                    # for i in range(0, w_featmap//args.query_rate):
                    #     for j in range(0, h_featmap//args.query_rate):
                    #         query = i * w_featmap*args.query_rate + j*args.query_rate
                    #         print("path index " + query)
                    #         attentions_reshaped, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
                    #         average_attentions = np.mean(attentions_reshaped, axis=0) 
                    #         fname = os.path.join(path, f"attn-average-{query}.png")
                    #         fig, ax = plt.subplots(figsize=(args.image_size[0]*px, args.image_size[0]*px))
                    #         ax.axis('off')   
                    #         ax.imshow(average_attentions, aspect='auto')
                    #         if query != 0: 
                    #             square = patches.Rectangle((j*args.patch_size*args.query_rate,i*args.patch_size*args.query_rate), 8,8, color='RED')
                    #             ax.add_patch(square)
                    #         fig.savefig(fname ,bbox_inches='tight', pad_inches=0)
                    #         plt.close(fig)
                    # print("finished saving query analysis")
                    print("Query analysis not supported for croped images")
        return




    if os.path.isfile(img_path):
        train_x = sorted(glob(img_path))
    else:
        train_x = sorted(glob(img_path + "/*.jpg"))    
    batch_size = 1
    
    start_time = time.time()
    if(args.crop > 1):
        if args.crop != 4 and args.crop != 16:
            print("crop must be 4 or 16")
        else:
            transformed_dataset = Croped_Dataset(train_x,croped_transform, args.crop,args.image_size)
            dl = DataLoader(transformed_dataset , batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
            train_croped(model, dl, device)
    else:
        transformed_dataset  = AIP_Dataset(train_x,transform)
        dl = DataLoader(transformed_dataset , batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        train(model, dl, device)

    end_time = time.time()
    excution_mins, execution_secs = execution_time(start_time, end_time)
    print(f"Execution time: {excution_mins}m {execution_secs}s")

    