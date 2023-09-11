import random
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from torch import nn
# import threshold_yen
from skimage.filters import threshold_yen, threshold_otsu
# import morphological cleaning
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
import numpy as np
import skimage
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.cluster import KMeans

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
def execution_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


""" Thresholding the image in two ways: 
    1- OTSU thresholding on the attention map * original image
    2- OTSU thresholding on the original image
"""
def adaptive_histogram_equalization(image):
    # Convert image to 8-bit single-channel format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def min_max_normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image
    return (image - min_val) / (max_val - min_val)

def threshold(img , attention, output_directory = "",save = True, name = None):
    # multipli img with average attention
    
    # attention = attention / np.max(attention)
    # # attention = adaptive_histogram_equalization(attention)
    # result = img * attention

    attention = min_max_normalize(attention)
    # convert img from PIL to numpy into8
    img = np.array(img)

    # attention = adaptive_histogram_equalization(attention)
    # result = img * attention / np.max(attention)
    
    # result = result.astype(np.uint8)
    alpha = 0.4
    attention = attention *255
    attention = attention.astype(np.uint8)
    result = (img/ 2) * (1-alpha) + (attention/ 2) * alpha
    result = result.astype(np.uint8)
    # save result
    fname = os.path.join(output_directory, "result.png")
    if save:
        plt.imsave(fname=fname, arr=result, format='png') 
    # apply OTSU thresholding to the average result with opencv
    ret , th = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # apply OTSU thresholding to the original image with skimage
    img = np.array(img)
    #thresh = skimage.filters.threshold_otsu(img)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    th2 = img > thresh
    th2 = th2.astype(np.uint8) * 255
    # ret2, th2 = cv2.threshold( np.array(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    ret3, th3 = cv2.threshold( attention, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if name is not None:
        file_name = name + "/"
        create_dir(os.path.join(output_directory,file_name))
    else:
        file_name = ""
    
    if save:
        fname = os.path.join(output_directory,file_name, "OTSU_th" + "_average.png")
        plt.imsave(fname=fname, arr=th, format='png', cmap='gray')
        print(f"{fname} saved.")
        fname = os.path.join(output_directory, "OTSU_th" + "_original.png")
        plt.imsave(fname=fname, arr=th2, format='png', cmap='gray')
        print(f"{fname} saved.")
        fname = os.path.join(output_directory, "weighted_iamge" + "_attention.png")
        plt.imsave(fname=fname, arr=result, format='png', cmap='gray')
        fname = os.path.join(output_directory, "heatmap_otsu" + "_attention.png")
        plt.imsave(fname=fname, arr=th3, format='png', cmap='gray')
        fname = os.path.join(output_directory, "temp.png")
        plt.imsave(fname=fname, arr=attention, format='png')
    return th, th2, th3


def kmeans(img, attention , output_directory = "",save = True, name = None):
    # multipli img with average attention
    # img = np.permute(img, (1,2,0))
    result = img * attention / np.max(attention)
    # save result
    fname = os.path.join(output_directory, "result.png")
    if save:
        plt.imsave(fname=fname, arr=result, format='png')
    #convert resul to CV_8UC1
    result = result.astype(np.uint8)
    # apply OTSU thresholding to the average result with opencv
    Z = result.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label_ours,center_ours=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center_ours = np.uint8(center_ours)
    res_ours = center_ours[label_ours.flatten()]
    res_ours = res_ours.reshape((result.shape))
    ret , res_ours = cv2.threshold(res_ours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # convert pil to numpy
    imag_array = np.array(img)
    imag_array = imag_array.astype(np.uint8)
    Z = imag_array.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.size))
    ret , res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if save:
        if name is not None:
            file_name = name + "/"
            create_dir(os.path.join(output_directory,file_name))
        else:
            file_name = ""
        fname = os.path.join(output_directory,file_name, "Kmeans_th" + "_average.png")
        plt.imsave(fname=fname, arr=res_ours, format='png', cmap='gray')
        print(f"{fname} saved.")
        fname = os.path.join(output_directory, "Kmeans_th" + "_original.png")
        plt.imsave(fname=fname, arr=res, format='png', cmap='gray')
        print(f"{fname} saved.")
    return res_ours, res

def kmeans_feature(img, features , output_directory = "",save = True, name = None):
    # Reshape to (50176X384)
    features = torch.reshape(features, (-1, features.shape[-1]))
    # max_values, _ = torch.max(features, dim=0)
    # img = img[0,0,:,:]
    # img =  img.view(-1)
    # features = torch.mul(features, img[:, None])
    # features = features / max_values
    # Normalize the feature maps
    mean = torch.mean(features, axis=0)
    std = torch.std(features, axis=0)
    features = (features - mean) / std

    # Perform k-means clustering with 2 clusters
    kmeans = KMeans(n_init = 10, n_clusters=2, random_state=0).fit(features)
    labels = kmeans.labels_

    # Reshape the cluster labels to (224X224)
    labels = labels.reshape(features.shape[-1], features.shape[-1])
    # Save the binary segmentation map
    
    if save:
        fname = os.path.join(output_directory, "kmeans_clusterd_segmentation.png")
        plt.imsave(fname=fname, arr=labels, format='png', cmap='gray')
        print(f"{fname} saved.")

    return labels *255

def chan_vese(img, attention , output_directory = "",save = True, name = None):
    # multipli img with average attention
    # img = np.permute(img, (1,2,0))
    result = img * attention / np.max(attention)
    # save result
    fname = os.path.join(output_directory, "result.png")
    if save:
        plt.imsave(fname=fname, arr=result, format='png')
    #convert resul to CV_8UC1
    result = result.astype(np.uint8)
    # apply chan vese thresholding to the average result with skimage 
    res_ours = skimage.segmentation.chan_vese(result, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,  max_num_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=True)
    res = skimage.segmentation.chan_vese(np.array(img), mu=0.25, lambda1=1, lambda2=1, tol=1e-3,  max_num_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=True)
    
    if save:
        if name is not None:
            file_name = name + "/"
            create_dir(os.path.join(output_directory,file_name))
        else:
            file_name = ""
        fname = os.path.join(output_directory,file_name, "ChanVese_th" + "_average.png")
        plt.imsave(fname=fname, arr=res_ours[0], format='png', cmap='gray')
        print(f"{fname} saved.")
        fname = os.path.join(output_directory, "ChanVese_th" + "_original.png")
        plt.imsave(fname=fname, arr=res[0], format='png', cmap='gray')
        print(f"{fname} saved.")
    return res_ours[0]*255, res[0]*255


""" Prepare the attention map for visualization """
def compute_attention(attentions, query,w_featmap, h_featmap, patch_size):
    attentions = attentions[0]
    nh = attentions.shape[1] # number of head, 
    attentions = attentions[0, :, query, 1:].reshape(nh, -1) 
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    return attentions, nh

def yen_threshold(img, output_directory = '', save = True):
    # apply YEN thresholding to the average result with opencv
    # pil image to numpy array
    img = np.array(img)
    th = threshold_yen(img)
    binary = th <= img
    # save as black and white cmap
    
    if save:
        fname = os.path.join(output_directory, "YEN_th" + ".png")
        plt.imsave(fname=fname, arr=binary, format='png', cmap='gray')
    return binary

def get_ROIs(img):
    img = remove_small_objects(img, min_size=20, connectivity=2, in_place=False)
    img = binary_closing(img, disk(2))
    img, num = label(img, return_num = True)
    return img

def morphology_cleaning(img, output_directory = '', save = True):
    # substract objects that are less then 20 px using morphology scikit image
    img = remove_small_objects(img, min_size=20, connectivity=2, in_place=False)
    img = binary_closing(img, disk(2))
    img, num = label(img, return_num = True)
    # get the center of each label in the image
    label_colored = label2rgb(img, bg_label=0)
    points = []

    px = 1/plt.rcParams['figure.dpi'] 
    fig, ax = plt.subplots(figsize=(800*px, 800*px))
    ax.axis('off')
    ax.imshow(label_colored)
    for region in regionprops(img):
        # take regions with large enough areas
        y0, x0 = region.centroid
        if region.area >= 10:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            # circle = mpatches.Circle((x0, y0), radius=2, fill=True, color='white')
            ax.plot(x0, y0, color = 'white', markersize=10,marker = 'x')
            ax.add_patch(rect)
            # ax.add_patch(circle)
            points.append((x0, y0))


    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(output_directory, "morphology_cleaning_boxes" + ".png"),bbox_inches='tight', pad_inches=0)


    print(f"Number of objects after morphology cleaning: {num}")
    fname = os.path.join(output_directory, "morphology_cleaning" + ".png")
    if save:
        plt.imsave(fname=fname, arr=label_colored, format='png', cmap='gray')

    return points
    

def concat_crops(crops):
    crop_number = len(crops)
    crop_iteration = int(np.sqrt(crop_number))
    vertical = []
    for i in range(crop_iteration):
        
        horizontal = crops[i*crop_iteration]
        for j in range(1,crop_iteration):
           horizontal = np.concatenate((horizontal, crops[i*crop_iteration + j ]), axis=1)
        if i == 0:
            vertical = horizontal
        else:
            vertical = np.concatenate((vertical, horizontal), axis=0)
    return vertical

def concat_crops_overlap(crops, stride):
    crop_number = len(crops)
    crop_iteration = int(np.sqrt(crop_number))
    vertical = []
    stride = stride * 2
    for i in range(crop_iteration):
        horizontal = crops[i * crop_iteration]
        for j in range(1, crop_iteration):
            # Concatenate the current crop with the previous crop horizontally
            left_window = horizontal
            right_window = crops[i * crop_iteration + j]
            crop_left = left_window[:, :-stride]
            crop_right = right_window[ :, stride:]
            overlap = (left_window[:, -stride:]// 2 + right_window[ :, :stride]// 2) 
            horizontal = np.concatenate((crop_left, overlap, crop_right), axis=1)
            
        if i == 0:
            # First row
            vertical = horizontal
        elif i == crop_iteration - 1:
            # Last row
            vertical = np.concatenate((vertical, horizontal[stride:, :]), axis=0)
        else:
            # Middle rows
            top_overlap = (vertical[-stride:, :]// 2 + horizontal[:stride, :]// 2) 
            vertical = np.concatenate((vertical[:-stride, :], top_overlap), axis=0)
            vertical = np.concatenate((vertical, horizontal[stride:, :]), axis=0)
    
    return vertical

def sliding_window(image, window_size, stride):
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

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    

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
    
def l1_normalize(feat_map):
    norm_factor = np.sum(np.abs(feat_map)) + 1e-8  # add small constant
    l1_norm_feat_map = feat_map / norm_factor
    return l1_norm_feat_map

def l2_normalize(image):
    norm = np.sqrt(np.sum(np.square(image)))
    if norm == 0:
        return image
    return image / norm

def zscore_normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image
    return (image - mean) / std

def softmax_normalize(feat_map):
    exp_feat_map = np.exp(feat_map)
    if np.any(np.sum(exp_feat_map, axis=1) == 0):
        exp_feat_map = exp_feat_map + 1e-8  # add small constant
    softmax_norm_feat_map = exp_feat_map / np.sum(exp_feat_map, axis=1, keepdims=True)
    return softmax_norm_feat_map

