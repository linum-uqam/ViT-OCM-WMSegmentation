# Self-supervised Segmentation

This module provides an implementation of a self-supervised segmentation method based on ViT (Vision Transformer) models. The code is adapted from the SimMIM and DINO projects.

## Installation

To install the required dependencies and set up the environment for this module, please follow the installation steps in the [SimMIM repository](https://github.com/microsoft/SimMIM).

## Overview

This module contains multiple scripts that facilitate various tasks related to self-supervised segmentation and analysis of attention maps. Here's a brief overview of the main script:

### `analyse_attention.py`

This script loads a ViT model and can generate numerous results for analysis purposes, including:

- Segmentation of the attention map itself.
- Weighting the original image with multiple approaches before segmentation.
- Applying multiple segmentation techniques (Otsu, K-means, Chan-Vese, Yan, etc.).
- Clustering the ViT features into a segmentation.
- Cropping large images and concatenating them.
- Preprocessing data to extract regions of interest.
- Querying regions of interest for guided attention maps, and more.

#### Script Arguments

- `--arch`: Architecture (supports only ViT at the moment).
- `--patch_size`: Patch resolution of the model.
- `--pretrained_weights`: Path to pretrained weights to load.
- `--image_path`: Path of the image\images to load.
- `--image_size`: Resize image dimensions.
- `--output_dir`: Path where to save visualizations.
- `--threshold`: Threshold value for visualizing masks obtained by thresholding self-attention maps.
- `--crop`: Amount of cropping (4 or 16).
- `--window_stride`: Stride of the sliding window.
- `--region_query`: Analyze the attention query.
- `--query_analysis`: Analyze the attention query.
- `--query_rate`: Rate of the query analysis.
- `--save_query`: Save queried attention maps with target points.
- `--save_feature`: Save the feature maps.

### `mim.py`

The `mim.py` script is responsible for training a Masked Image Modeling (MIM) model on the provided dataset of images. It also supports logging experiments to WandB (Weights and Biases), making it easy to track and analyze the training process.

#### Script Arguments

- `--opts`: Modify config options by adding 'KEY VALUE' pairs.
- `--arch`: Architecture (supports only ViT at the moment).
- `--patch_size`: Patch resolution of the model.
- `--pretrained_weights`: Path to pretrained weights to load.
- `--image_path`: Path of the image dataset to load.
- `--image_size`: Resize image dimensions.
- `--output_dir`: Path where to save visualizations.
- `--output`: Root of the output folder.
- `--epochs`: Number of total epochs to run.
- `--warmup_epochs`: Number of warmup epochs to run.
- `--num_workers`: Number of data loading workers.
- `--batch_size`: Batch size.
- `--mask_patch_size`: Patch size for the mask.
- `--mask_ratio`: Ratio of the mask.
- `--tag`: Tag for the experiment.
- `--wandb`: Whether to use WandB for experiment tracking.
- `--loss_operation`: Choose from mean, sum, or max for loss operation.
- `--eval_dataset_path`: Evaluate the model on the given dataset.
- `--roi_masking`: Whether to use ROI masking.
- `--early_stopping`: Whether to use early stopping during training.

### `eval.py`

The `eval.py` script is designed to evaluate the approach on a labeled dataset. It allows you to assess the performance of your segmentation model and visualize the results. Below are the script's key arguments:

#### Script Arguments

- `--arch`: Architecture (supports only ViT at the moment).
- `--patch_size`: Patch resolution of the model.
- `--pretrained_weights`: Path to pretrained weights to load.
- `--eval_dataset_path`: Path to the labeled dataset for evaluation.
- `--image_size`: Resize image dimensions.
- `--output_dir`: Path where visualizations and evaluation results will be saved.
- `--crop`: Amount of cropping (4 or 16).
- `--region_query`: To analyze the attention query or not.
- `--batch_size`: Batch size.
- `--wandb`: Whether to use WandB for experiment tracking (Boolean).
- `--tag`: Tag for WandB experiment tracking.
- `--method`: Method to implement (e.g., ours, otsu, k-means, k-means_ours, chan-vese, chan-vese_ours, heatmap_threshold).

### `PGT.py`

The `PGT.py` script is designed for training and evaluating the Pseudo-Ground-Truth (PGT) approach. This script allows you to train a model and assess its performance using PGT. Below are the script's key arguments:

#### Script Arguments

- `--patch_size`: Patch resolution of the model.
- `--image_size`: Resize image dimensions.
- `--pretrained_weights`: Path to pretrained weights to load.
- `--checkpoint_key`: Key to use in the checkpoint (e.g., "teacher").
- `--arch`: Architecture (supports only ViT at the moment).
- `--opts`: Modify config options by adding 'KEY VALUE' pairs.
- `--wandb`: Whether to use WandB for experiment tracking (Boolean).
- `--H`: Height of the images.
- `--W`: Width of the images.
- `--ratio`: Ratio for the model.
- `--finetune`: Whether to finetune the model (Boolean).
- `--image_path`: Path to the directory containing the images for training.
- `--eval_dataset_path`: Path to the labeled dataset for evaluation.
- `--epochs`: Number of total epochs to run.
- `--batch_size`: Batch size.
- `--base_lr`: Base learning rate for training.

### `sw_processing.py`

The `sw_processing.py` script is used to experiment with sliding windows and overlap blending to process large images and mosaics using a trained model. This script facilitates the segmentation of large images by breaking them into smaller patches and then combining the results.

#### Script Arguments

- `--arch`: Architecture (supports only ViT at the moment).
- `--patch_size`: Patch resolution of the model.
- `--pretrained_weights`: Path to pretrained weights to load.
- `--image_size`: Resize image dimensions.

### `unet.py`

The `unet.py` script trains and evaluate a supervised unet similarly to `PGT.py`