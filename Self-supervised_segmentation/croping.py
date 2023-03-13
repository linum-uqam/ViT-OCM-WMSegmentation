import os
import numpy as np
from PIL import Image

# Set input and output paths
input_path = "/home/mohamad_h/data/Fluo-N2DL-HeLa/Fluo-N2DL-HeLa_v2/images/"
output_path = "/home/mohamad_h/data/Fluo-N2DL-HeLa/Hela_v2_croped/images/"
# Define crop size
crop_size = (384, 384)
print("crop_size", crop_size)
print("input_path", input_path)
print("output_path", output_path)
# Loop over all images in the input folder
for filename in os.listdir(input_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        image = Image.open(os.path.join(input_path, filename))
        # Get image dimensions
        width, height = image.size
        # Calculate number of crops in each dimension
        num_crops_x = int(np.ceil(width / crop_size[0]))
        num_crops_y = int(np.ceil(height / crop_size[1]))
        # Loop over all crops
        for i in range(num_crops_x):
            for j in range(num_crops_y):
                # Calculate crop coordinates
                left = i * crop_size[0]
                upper = j * crop_size[1]
                right = min((i + 1) * crop_size[0], width)
                lower = min((j + 1) * crop_size[1], height)
                # Crop image
                crop = image.crop((left, upper, right, lower))
                # Save crop
                crop_filename = "{}_{}_{}.jpg".format(filename[:-4], i, j)
                fname = os.path.join(output_path, crop_filename)
                crop.save(fname)
                print("Saved {}".format(fname))