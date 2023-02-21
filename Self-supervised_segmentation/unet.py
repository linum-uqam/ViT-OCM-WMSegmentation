
import numpy as np
import os
from glob import glob #exctrac the images
import time
from tqdm import tqdm # for the progress bar
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

log_wandb = True
config = {
    "H" : 256,
    "W" :256,
    "ratio" : 0.6,
    }

if log_wandb == True:
        wandb.login()
        wandb.init(
            project="UNET_OCM",
            entity="mohamad_hawchar",
            name = f"unet_0.6",
            config=config,
            )
        config = wandb.config
H = config.H
W = config.W
ratio = config.ratio    
        
        
class convolution_block(nn.Module):
  def __init__(self,in_c,out_c):
    super().__init__()

    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)

    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)

    self.relu = nn.ReLU()


  def forward(self,inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = convolution_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = convolution_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = convolution_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs) # s1 = 1,64,334,334 p1 = 1,64,167,167
        s2, p2 = self.e2(p1)  # s2 = 1,128,167,167 p2 = 1,128,83,83
        s3, p3 = self.e3(p2) # s3 = 1,256,83,83 p3 = 1,256,41,41
        s4, p4 = self.e4(p3) # s4 = 1,512,41,41 p4 = 1,512,20,20

        """ Bottleneck """
        b = self.b(p4) # b = 1,1024,20,20

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

"""### Attention U-net"""

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class Attention_block(nn.Module):


    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

"""### Nested U-net"""

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

#Nested Unet

class NestedUNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

"""## Utils"""

""" Seeding the random """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class Dataset(Dataset):
    def __init__(self, images_path, masks_path, image_size=512):

        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

"""## Training"""

def train(model, loader, optimizer, loss_fn, device):
    sum_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Initialiser les paramètres des gradients à zéro
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward() # autograd magic ! Calcule toutes les dérivées partielles
        optimizer.step() # Effectue un pas dans la direction du gradient


        sum_loss += loss.item()

    sum_loss = sum_loss/len(loader)

    return sum_loss

def evaluate(model, loader, loss_fn, device):
    sum_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            sum_loss += loss.item()

        sum_loss = sum_loss/len(loader)
    return sum_loss

def smooth(x, size):
    return np.convolve(x, np.ones(size)/size, mode='valid')

def fully_train(net, model_name):
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/train/images/*"))
    # take only percentage of the training data
    train_x = train_x[:int(len(train_x)*ratio)]
    train_y = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/train/labels/*"))
    # take only percentage of the training data
    train_y = train_y[:int(len(train_y)*ratio)]
    valid_x = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/valid/images/*"))
    valid_y = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/valid/labels/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} / {len(train_y)} - Valid: {len(valid_x)} / {len(valid_y)}\n"
    print(data_str)

    """ Hyperparameters """
    size = (H, W)
    batch_size = 8
    num_epochs = 150
    lr = 1e-5
    checkpoint_path = f"files/{model_name}.pth"

    """ Dataset and loader """
    train_dataset = Dataset(train_x, train_y, image_size=H)
    valid_dataset = Dataset(valid_x, valid_y, image_size=H)

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

    device = torch.device('cuda')
    model = net()
    model = model.to(device)
    if log_wandb == True:
        wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceLoss()
    #loss_fn = DiceLoss()
    #loss_fn = nn.CrossEntropyLoss()

    """ Training the model """
    best_valid_loss = float("inf")
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(data_str)
    plt.plot(smooth(train_losses,1), 'r-',label='training')
    plt.plot(smooth(valid_losses, 1), 'b-',label='validation')
    return train_losses,valid_losses

"""## Testing"""

def calculate_metrics(y_true, y_pred , y_score):

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

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc,score_roc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def fully_test(net,model_name):

      """ Seeding """
      seeding(42)

      """ Folders """
      create_dir("results")
      create_dir(f"results/{model_name}")

      """ Load dataset """
      test_x = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/test/images/*"))
      test_y = sorted(glob("/home/mohamad_h/data/AIP_annotated_data_cleaned_splitted/test/labels/*"))

      """ Hyperparameters """

      size = (W, H)
      checkpoint_path = f"files/{model_name}.pth"

      """ Load the checkpoint """
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      model = net()
      model = model.to(device)
      model.load_state_dict(torch.load(checkpoint_path, map_location=device))
      model.eval()
      loss_fn = DiceLoss()
      metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0,0.0]
      loss = 0.0
      time_taken = []

      for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
              """ Extract the name """
              name = x.split("/")[-1].split(".")[0]

              """ Reading image """
              image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
              image = cv2.resize(image, size)
              ## image = cv2.resize(image, size)
              x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
              x = x/255.0
              x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
              x = x.astype(np.float32)
              x = torch.from_numpy(x)
              x = x.to(device)

              """ Reading mask """
              mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
              mask = cv2.resize(mask, size)
              ## mask = cv2.resize(mask, size)
              y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
              y = y/255.0
              y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
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
                  pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
                  pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
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
      if log_wandb == True:
                wandb.log({
                "accuracy": acc,
                 "f1" :f1,
                 "precision":precision,
                 "recall":recall,
                 "jaccard":jaccard,
                 "loss":loss,
                 "Dice":1 - loss,
                #  "input_images": [
                #     wandb.Image(img, caption="Input Image"),
                #     wandb.Image(target, caption="Target"),
                #     wandb.Image(output, caption="Output") ,
                #     wandb.Image(attnetion_image, caption="Attention")
                #     ],
                 }, step=i)
    #   print("FPS: ", fps)

"""## Models Training, Testing & Results

### Unet Results
"""

tl_UNet,vl_UNet =fully_train(build_unet,"unet")

fully_test(build_unet,"unet")
if log_wandb == True:
        wandb.finish()
"""### Attention Unet Resutls"""

# # attention unet
# tl_AttU_Net,vl_AttU_Net = fully_train(AttU_Net,"attention_unet")

# fully_test(AttU_Net,"attention_unet")
# plt.figure(dpi = 200)
# plt.imshow(cv2.imread("/content/results/attention_unet/01_test_0.png"))

# """### Nested Unet Results"""

# # nested unet
# tl_NestedUNet,vl_NestedUNet = fully_train(NestedUNet,"nested_unet")

# fully_test(NestedUNet,"nested_unet")
# plt.figure(dpi = 200)
# plt.imshow(cv2.imread("/content/results/nested_unet/01_test_0.png"))

# """## comparasion"""

# #
plt.plot(smooth(vl_UNet,1), 'r-',label='vl_unet')
# plt.plot(smooth(vl_AttU_Net, 1), 'b-',label='vl_AttU_Net')
# #plt.plot(smooth(vl_R2U_Net, 1), 'g-',label='vl_R2U_Net')
# plt.plot(smooth(vl_NestedUNet, 1), 'g-',label='vl_NestedUNet')
plt.legend()
plt.show

torch.cuda.empty_cache()