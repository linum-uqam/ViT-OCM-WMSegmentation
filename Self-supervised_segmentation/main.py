import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import argparse
import torch
import numpy as np
import dino.vision_transformer as vits
from scipy.ndimage import median_filter
from utils import threshold, compute_attention,create_dir, seeding
import time
from logger import create_logger
from data import build_eval_loader
import wandb
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import torchvision.transforms as T
import torch.nn as nn 
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/mohamad_h/output/vit_small/AIP+M_224_multistepLR_60B_meanL/ckpt_epoch_25.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="/home/mohamad_h/data/AIP_annotated_data/", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(384, 384), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='/home/mohamad_h/LINUM/Results/AIPs_labeled/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.1, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    # Boolyan for croping 
    parser.add_argument("--crop", type=int, default=1, help="""Amount of croping (4 or 16)""")
    # Attention query analysis mode boolean
    parser.add_argument("--region_query", type=bool, default=False, help="""To analyze the attention query or not""")
    parser.add_argument("--query_analysis", type=bool, default=False, help="""To analyze the attention query or not""")
    # query rate parameter defaul is 10
    parser.add_argument("--query_rate", type=int, default=10, help="""Rate of the query analysis""")
    # boolean to save the queried attention maps with the target points
    parser.add_argument("--save_query", type=bool, default=False, help="""To save the queried attention maps with the target points""")
    # boolean to save the feature maps
    parser.add_argument("--save_feature", type=bool, default=False, help="""To save the feature maps""")
    parser.add_argument("--batch_size", type=int, default=1, help="""Batch size""")
    parser.add_argument('--wandb', default=False, help='whether to use wandb')

    args = parser.parse_args()
    return args

def main(args):

    data_loader = build_eval_loader(args)
    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if args.wandb:
        wandb.watch(model)
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
    
    logger.info(f"Creating model:{args.arch}/{args.patch_size}")
    logger.info(str(model))
    validate(args, data_loader, model, device)

@torch.no_grad()
def validate(args, data_loader, model, device):
    criterion = DiceLoss()
    valid_losses = []
    sum_loss = 0.0
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    f1_meter = AverageMeter()
    transform = T.ToPILImage()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        feat, attentions, qkv = model.get_intermediate_feat(images.to(device), n=1)
        query = 0
        w_featmap = images.shape[-2] // args.patch_size
        h_featmap = images.shape[-1] // args.patch_size
        attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, args.patch_size)
        # average attentions over heads
        average_attentions = np.mean(attention_response, axis=0)
        # median filter scipy
        average_attentions = median_filter(average_attentions, size=10)
        output, original_otsu = threshold(transform(images.squeeze(0)).convert("L") , average_attentions, save=False)
        # output = original_otsu
        output = torch.tensor(output).unsqueeze(0)
        output = output/255
        # target = target.squeeze(0)
        output = output.unsqueeze(0)
        output = output.cuda(non_blocking=True)
        # measure accuracy and record loss
        loss = criterion(output, target)
        valid_losses.append(loss.item())
        sum_loss += loss.item()
        score_jaccard, score_f1, score_recall, score_precision, score_acc = calculate_metrics(target, output)

        loss_meter.update(loss.item(), target.size(0))
        acc_meter.update(score_acc.item(), target.size(0))
        f1_meter.update(score_f1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 1 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'f1 {f1_meter.val:.3f} ({f1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc_average: {acc_meter.avg:.3f} F1_average {f1_meter.avg:.3f}')
    if args.wandb:
                wandb.log({"Loss":loss_meter.val, "Acc":acc_meter.avg, "f1" :f1_meter.avg }, step=idx)
    return acc_meter.avg, f1_meter.avg, loss_meter.avg

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def calculate_metrics(y_true, y_pred):
  
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


if __name__ == "__main__":
    args = parse_args()
    create_dir(args.output_dir)
    logger = create_logger(output_dir=args.output_dir, name=f"{args.arch}_{args.patch_size}")
    seeding(0)
    if args.wandb:
        wandb.login()
        wandb.init(
            project="segmentation_evaluatoin",
            entity="mohamad_hawchar",
            name = args.TAG,
            config=args
            )
    main(args)
    if args.wandb:
        wandb.finish()
