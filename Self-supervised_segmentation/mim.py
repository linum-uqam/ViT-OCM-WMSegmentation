# git add --all -- ':!images/' ':!AIPs_40X/'
# more augmentations 
# full training and validation
# manual segmentations
# learn the purpose of learning schedule
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4 , 5"
import argparse
import torch
from torchvision import transforms as pth_transformsmulti
import dino.vision_transformer as vits
from torchsummary import summary
from glob import glob
from data import build_loader_simmim
from utils import seeding, create_dir, get_grad_norm, save_checkpoint
import time
from config import get_config
from optimizer import build_pretrain_optimizer
from logger import create_logger
from lr_scheduler import build_scheduler
import time
import datetime
from timm.utils import AverageMeter
from model import MIM, build_model

def parse_option():
    parser = argparse.ArgumentParser('MIM Pretraining')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./checkpoints/model_finetune.pth', type=str,
        help="Path to pretrained weights to load.") #
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="/home/mohamad_h/STEGO/STEGO/src/Aips_Guassian+TopHat/imgs/", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(400, 400), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.1, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='number of warmup epochs to run')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--mask_patch_size', default=16, type=int, help='patch size for the mask')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='ratio of the mask')
    args = parser.parse_args()
    args = get_config(args)
    return args

def main(args):
    data_loader_train = build_loader_simmim(args)
    logger.info(f"Creating model:{args.MODEL.NAME}/{args.MODEL.PATCH_SIZE}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    encoder = build_model(args)
    model = MIM(encoder=encoder, encoder_stride=16)
    model.to(device)
    logger.info(str(model))

    optimizer = build_pretrain_optimizer(args, model, logger) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    # summary(model, ((3, 224, 224), (28,28)))

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.TRAIN.START_EPOCH, args.TRAIN.EPOCHS):
        train_one_epoch(args, model.encoder, data_loader_train, optimizer, epoch, lr_scheduler)
        if epoch % args.SAVE_FREQ == 0 or epoch == (args.TRAIN.EPOCHS - 1):
            save_checkpoint(args, epoch, model, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask, _) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss = model(img, mask)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(optimizer, config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(optimizer)
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args = parse_option()
    logger = create_logger(output_dir=args.DATA.OUTPUT_DIR, name=f"{args.MODEL.NAME}_{args.MODEL.PATCH_SIZE}")
    seeding(0)
    create_dir(args.DATA.OUTPUT_DIR)
    main(args)
    
