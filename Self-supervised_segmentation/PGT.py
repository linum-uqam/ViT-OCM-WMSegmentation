

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import numpy as np
from glob import glob  # exctrac the images
import time
from tqdm import tqdm  # for the progress bar
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from operator import add
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import random
import torch.nn.functional as F
import wandb
from model import build_finetune_model, LinearProbing, build_unet
from utils import seeding, create_dir, execution_time, DiceLoss, compute_attention, threshold
from config import get_config
import argparse
import torchvision.transforms as T
rand = False
class Dataset_PGT(Dataset):
    def __init__(self, images_path, image_size=512, encoder = None):

        self.images_path = images_path
        self.image_size = image_size
        self.n_samples = len(images_path)
        self.encoder = encoder

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image/255.0  # (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image

    def __len__(self):
        return self.n_samples


"""## Training"""


def train(model, loader, optimizer, loss_fn, device, encoder):
    sum_loss = 0.0
    model.train()
    for x in loader:
        x = x.to(device, dtype=torch.float32)
        y = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(device, dtype=torch.float32)
        transform = T.ToPILImage()
        with torch.no_grad():
            for i in range(x.shape[0]):
                x_ = x[i].unsqueeze(0)
                feat, attentions, qkv = encoder.get_intermediate_feat(x_, n=1)
                query = 0
                w_featmap = x_.shape[-2] // 8
                h_featmap = x_.shape[-1] // 8
                attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, 8)
                # average attentions over heads

                # rand = True
                if rand:
                    # Generate a random number of rows to select between 1 and 6
                    num_rows = np.random.randint(1, 7)
                    # Generate a random set of row indices
                    selected_rows = np.random.choice(attention_response.shape[0], size=num_rows, replace=False)
                    # Select the corresponding rows of the array
                    selected_attentions = attention_response[selected_rows, :]
                    # Compute the mean of the selected rows
                    average_attentions = np.mean(selected_attentions, axis=0)
                else:
                    average_attentions = np.mean(attention_response, axis=0)
                average_attentions = cv2.resize(average_attentions, (average_attentions.shape[1]//8, average_attentions.shape[0]//8))
                # interpolate the attention map to the original size with bicubic interpolation
                average_attentions = cv2.resize(average_attentions, (x_.shape[-1], x_.shape[-1]), interpolation=cv2.INTER_LINEAR)
                output, original_otsu, heatmap_otsu = threshold(transform(x_.squeeze(0)).convert("L") , average_attentions, save=False)
                # save output
                fname = "mask.png"
                # cv2.imwrite(fname, output)
                y[i] = torch.from_numpy(output/255.0).unsqueeze(0).unsqueeze(0) 
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()  # autograd magic ! Calcule toutes les dérivées partielles
        optimizer.step()  # Effectue un pas dans la direction du gradient

        sum_loss += loss.item()

    sum_loss = sum_loss/len(loader)

    return sum_loss


def evaluate(model, loader, loss_fn, device, encoder):
    sum_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device, dtype=torch.float32)
            y = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(device, dtype=torch.float32)
            transform = T.ToPILImage()
            for i in range(x.shape[0]):
                x_ = x[i].unsqueeze(0)
                feat, attentions, qkv = encoder.get_intermediate_feat(x_, n=1)
                query = 0
                w_featmap = x_.shape[-2] // 8
                h_featmap = x_.shape[-1] // 8
                attention_response, nh = compute_attention(attentions, query, w_featmap, h_featmap, 8)
                # average attentions over heads

                # rand = True
                if rand:
                    # Generate a random number of rows to select between 1 and 6
                    num_rows = np.random.randint(1, 7)
                    # Generate a random set of row indices
                    selected_rows = np.random.choice(attention_response.shape[0], size=num_rows, replace=False)
                    # Select the corresponding rows of the array
                    selected_attentions = attention_response[selected_rows, :]
                    # Compute the mean of the selected rows
                    average_attentions = np.mean(selected_attentions, axis=0)
                else:
                    average_attentions = np.mean(attention_response, axis=0)
                average_attentions = cv2.resize(average_attentions, (average_attentions.shape[1]//8, average_attentions.shape[0]//8))
                # interpolate the attention map to the original size with bicubic interpolation
                average_attentions = cv2.resize(average_attentions, (x_.shape[-1], x_.shape[-1]), interpolation=cv2.INTER_LINEAR)
                output, original_otsu, heatmap_otsu = threshold(transform(x_.squeeze(0)).convert("L") , average_attentions, save=False)
                # save output
                fname = "mask.png"
                # cv2.imwrite(fname, output)
                # convert output to float32
                y[i] = torch.from_numpy(output/255.0).unsqueeze(0).unsqueeze(0)  


            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            sum_loss += loss.item()

        sum_loss = sum_loss/len(loader)
    return sum_loss


def smooth(x, size):
    return np.convolve(x, np.ones(size)/size, mode='valid')


def fully_train(net, model_name, config, encoder):
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    images = sorted(glob(config.DATA.IMAGE_PATH +"/*"))
    print(f"dir: {config.DATA.IMAGE_PATH}")
    train_x = images[:200]
    valid_x = images[200:]
    train_x = train_x[:int(len(train_x)*config.ratio)]

    data_str = f"Dataset Size:\nTrain: {len(train_x)}  - Valid: {len(valid_x)} \n"
    print(data_str)

    """ Hyperparameters """
    size = (config.H, config.W)
    batch_size = config.batch_size
    num_epochs = config.TRAIN.EPOCHS
    lr = config.TRAIN.BASE_LR
    checkpoint_path = f"files/{model_name}.pth"
    device = torch.device('cuda')
    encoder.to(device)
    encoder.eval()
    """ Dataset and loader """
    train_dataset = Dataset_PGT(train_x, image_size=config.H, encoder=encoder)
    valid_dataset = Dataset_PGT(valid_x, image_size=config.H, encoder=encoder)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model = net()
    model = model.to(device)
    if config.WANDB == True:
        wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceLoss()
    # loss_fn = DiceLoss()
    # loss_fn = nn.CrossEntropyLoss()

    """ Training the model """
    best_valid_loss = float("inf")
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, encoder)
        valid_loss = evaluate(model, valid_loader, loss_fn, device, encoder)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = execution_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # if config.WANDB == True:
        #     wandb.log({"Train Loss": train_loss, "Valid Loss": valid_loss})
        print(data_str)
    plt.plot(smooth(train_losses, 1), 'r-', label='training')
    plt.plot(smooth(valid_losses, 1), 'b-', label='validation')
    return train_losses, valid_losses


"""## Testing"""


def calculate_metrics(y_true, y_pred, y_score):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Score """
    y_score = y_score.cpu().numpy()
    y_score = y_score > 0.5
    y_score = y_score.astype(np.uint8)
    y_score = y_score.reshape(-1)
    # #y_score = y_score/255
    # print(y_score)

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
    score_roc = roc_auc_score(y_true, y_score)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_roc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


def fully_test(net, model_name, config):
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")
    create_dir(f"results/{model_name}")

    """ Load dataset """

    #   test_x = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/test/images/*"))
    #   test_y = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/test/labels/*"))

    images = sorted(
        glob(config.eval_dataset_path +"/images/*"))
    labels = sorted(
        glob(config.eval_dataset_path +"/labels/*"))
    test_x = images#[70:]
    test_y = labels#[70:]

    """ Hyperparameters """

    size = (config.W, config.H)
    checkpoint_path = f"files/{model_name}.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = net()
    model = model.to(device)
    # model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    loss_fn = DiceLoss()
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    loss = 0.0
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)
        image = cv2.resize(image, size)
        # image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
        mask = cv2.resize(mask, size)
        # mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)  # (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            loss = loss + loss_fn(pred_y, y)
            y_prob = pred_y
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y, y_prob)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()  # (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)  # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)

        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results/{model_name}/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    roc = metrics_score[5]/len(test_x)
    loss = loss / len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - ROC-AUC : {roc:1.4f}")
    if config.WANDB == True:
        print("Logging to wandb")
        wandb.log({
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "jaccard": jaccard,
            "loss": loss,
            "Dice": 1 - loss,
            #  "input_images": [
            #     wandb.Image(img, caption="Input Image"),
            #     wandb.Image(target, caption="Target"),
            #     wandb.Image(output, caption="Output") ,
            #     wandb.Image(attnetion_image, caption="Attention")
            #     ],
        }, step=i)
    #   print("FPS: ", fps)


def main():
    args = argparse.Namespace()
    args.patch_size = 8
    args.image_size = 384
    args.pretrained_weights = '/home/mohamad_h/LINUM/maitrise-mohamad-hawchar/Self-supervised_segmentation/output/vit_small/VIT_8_AM_384_16B_0.3R_16MP/ckpt_epoch_15.pth'
    args.checkpoint_key = "teacher"
    args.arch = 'vit_small'
    args.opts = None
    args.wandb = False
    args.H = 384
    args.W = 384
    args.ratio = 1.0
    args.finetune = False
    args.image_path = '/home/mohamad_h/data/Aips_Good_Images_Only/imgs/'
    args.eval_dataset_path = '/home/mohamad_h/data/AIP_annotated_data_cleaned/'
    args.epochs = 30
    args.batch_size = 8
    args.base_lr = 1e-3
    config = get_config(args)
    

    if config.WANDB == True:
        wandb.login()
        wandb.init(
            project="todelete",
            entity="mohamad_hawchar",
            name=f"pgt_good_data_only_3epochs_unsupervised",
            config=config,
        )
        config3 = wandb.config

    encoder = build_finetune_model(config)
    if config.finetune == False:
        for param in encoder.parameters():
            param.requires_grad = False
    
    tl_UNet, vl_UNet = fully_train(build_unet, "pgt",config, encoder)
    fully_test(build_unet, "pgt", config)
    if config.WANDB == True:
        wandb.finish()

    # plt.plot(smooth(vl_UNet, 1), 'r-', label='vl_unet')
    # plt.legend()
    # plt.show

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()