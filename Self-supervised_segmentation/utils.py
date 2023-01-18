import random
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from torch import nn
# import threshold_yen
from skimage.filters import threshold_yen
# import morphological cleaning
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, binary_dilation, binary_erosion, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches


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
def threshold(img , attention, output_directory,save = True):
    # multipli img with average attention
    result = img * attention / np.max(attention)
    # save result
    fname = os.path.join(output_directory, "result.png")
    if save:
        plt.imsave(fname=fname, arr=result, format='png')
    #convert resul to CV_8UC1
    result = result.astype(np.uint8)
    # apply OTSU thresholding to the average result with opencv
    ret , th = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold( np.array(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    fname = os.path.join(output_directory, "OTSU_th" + "_average.png")
    # save as black and white cmap
    if save:
        plt.imsave(fname=fname, arr=th, format='png', cmap='gray')
        print(f"{fname} saved.")
        fname = os.path.join(output_directory, "OTSU_th" + "_original.png")
        plt.imsave(fname=fname, arr=th2, format='png', cmap='gray')
        print(f"{fname} saved.")
    return th, th2

""" Prepare the attention map for visualization """
def compute_attention(attentions, query,w_featmap, h_featmap, patch_size):
    attentions = attentions[0]
    nh = attentions.shape[1] # number of head, 
    attentions = attentions[0, :, query, 1:].reshape(nh, -1) 
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    return attentions, nh

def yen_threshold(img, output_directory, save = True):
    # apply YEN thresholding to the average result with opencv
    # pil image to numpy array
    img = np.array(img)
    th = threshold_yen(img)
    binary = th <= img
    # save as black and white cmap
    fname = os.path.join(output_directory, "YEN_th" + ".png")
    if save:
        plt.imsave(fname=fname, arr=binary, format='png', cmap='gray')
    return binary

def morphology_cleaning(img, output_directory, save = True):
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
            circle = mpatches.Circle((x0, y0), radius=2, fill=True, color='white')
            ax.add_patch(rect)
            ax.add_patch(circle)
            points.append((x0, y0))


    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_directory, "morphology_cleaning_boxes" + ".png"),bbox_inches='tight', pad_inches=0)


    print(f"Number of objects after morphology cleaning: {num}")
    fname = os.path.join(output_directory, "morphology_cleaning" + ".png")
    if save:
        plt.imsave(fname=fname, arr=label_colored, format='png', cmap='gray')

    return points
    


# function that concatinates 16 crop parts of an image, use numpy
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

