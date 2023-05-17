import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torchvision
from torchvision import transforms as T
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassJaccardIndex
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import segmentation_models_pytorch as smp
import wandb
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def track_image_examples():
    """
    function to log segmentation examples during training to wandb
    """
    color_mapping = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (130, 76, 0),
    3: (0, 102, 0),
    4: (112, 103, 87),
    5: (28, 42, 168),
    6: (48, 41, 30),
    7: (0, 50, 89),
    8: (107, 142, 35),
    9: (70, 70, 70),
    10: (102, 102, 156),
    11: (254, 228, 12),
    12: (254, 148, 12),
    13: (190, 153, 153),
    14: (153, 153, 153),
    15: (255, 22, 96),
    16: (102, 51, 0),
    17: (9, 143, 150),
    18: (119, 11, 32),
    19: (51, 51, 0),
    20: (190, 250, 190),
    21: (112, 150, 146),
    22: (2, 135, 115),
    23: (255, 0, 0)
    }
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    example_images = []
    example_targets = []
    example_preds = []
    sample_indices = [2,4]
    for idx in sample_indices:
        data, target = val_dataset[idx]
        target = target.float()
        data = data.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        probs = model(data)
        preds = torch.argmax(probs, dim=1)
        
        data = data.squeeze().cpu().permute(1, 2, 0)
        data = data.numpy()
        data = (data * std) + mean
        data = torch.from_numpy(data)
        data = data.permute(2, 0, 1)  # Adjust the dimensions back to (C, H, W)
        data = data.unsqueeze(0)
        
        target = target.squeeze().cpu()
        target = target.numpy()

        # Convert the numpy mask image to RGB
        mask_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                class_number = target[i, j]
                rgb = color_mapping.get(class_number, (0, 0, 0))  # Get RGB value from color_mapping
                mask_rgb[i, j] = rgb
        target = mask_rgb
        target = target / 255
        target = torch.from_numpy(target)
        target = target.float()
        target = target.permute(2,0,1)
        target = target.unsqueeze(0)
        
        preds = preds.squeeze().cpu()
        preds = preds.numpy()
        mask_rgb = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                class_number = preds[i, j]
                rgb = color_mapping.get(class_number, (0, 0, 0))  # Get RGB value from color_mapping
                mask_rgb[i, j] = rgb
        preds = mask_rgb
        preds = preds / 255
        preds = torch.from_numpy(preds)
        preds = preds.float()
        preds = preds.permute(2,0,1)
        preds = preds.unsqueeze(0)
        
        example_images.append(data)
        example_preds.append(preds)
        example_targets.append(target)
        
    grid_images = torch.cat([torch.cat(example_images), torch.cat(example_targets), torch.cat(example_preds)], dim=3)

    grid_images = grid_images.cpu().numpy()


    example_images_grid = torchvision.utils.make_grid(torch.from_numpy(grid_images), nrow=1)

    wandb.log({"Examples/Segmentation_example": [wandb.Image(example_images_grid, caption="Example Images")]})    
