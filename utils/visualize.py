import sys
import os

import cv2
import torch
import os
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import pickle

import enum
import wandb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# TODO finish visualization stuff
"""
    To visualize and confirm datasets are working properly
"""

def draw_masks(image, masks, mask_type='fh', format='binary')->Image:
    """
        Draw masks onto image
    """
    
    if isinstance(image, str):
        image = Image.open(image)

    for mask in masks:
        if type(mask) == str:
            # mask is a path to a mask
            with open(mask, 'rb') as f:
                mask = pickle.load(f)
        
        if format == 'binary':
            mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L').convert('RGBA')
            image.paste(mask_image, (0, 0), mask_image)

    return image

def draw_bboxes(image, bboxes, format="corners") -> Image:
    """ 
        draws bboxes onto an image. Image is either a Image or a url
        format: string of corners (x1,y1,x2,y2) or center (x1,y1,w,h)
        bbox is [x1,y1,x2,y2] tensor
    """

    if type(image) == str:
        image = Image.open(image)
    elif type(image) == torch.Tensor:
        tensor = image.cpu().detach()
        # If tensor is float, scale to [0, 255] and convert to uint8
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            tensor = tensor * 255.0
            tensor = tensor.byte()
        # Convert tensor to NumPy array and reorder dimensions from (C, H, W) to (H, W, C)
        numpy_array = tensor.numpy()
        numpy_array = np.squeeze(numpy_array)
        numpy_array = numpy_array.transpose(1, 2, 0)
        # Convert NumPy array to PIL Image
        image = Image.fromarray(numpy_array)

    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        if format == "corners":
            x1, y1, x2, y2 = bbox
        elif format == "center":
            cx, cy, w, h = bbox
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
        else:
            raise ValueError("Invalid format. Use 'corners' or 'center'.")
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)


    return image

def visualize_custom_coco_dataset(dataset):
    # visualize dataset
    # first, create dataloader
    
    if not dataset is Custom_Coco_Dataset:
        print("NOT A COCO DATASET!")
        return
    
    cv2.namedWindow('COCO Viewer', cv2.WINDOW_NORMAL)
    cur_index = 0
    total_images = len(dataset)

    while cur_index < total_images:
        image, target = dataset[cur_index]
        image = F.to_pil_image(image)

        # draw bboxes on image
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            class_name = f'Class: {label.item()}'
            cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow('COCO Viewer', image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            current_index += 1

    cv2.destroyAllWindows()    

    pass

def wandb_dump_img(imgs,category):
    n_imgs = len(imgs)
    fig, axes = plt.subplots(1,n_imgs,figsize=(5*n_imgs, 5))
    #raw, kmeans on 
    fig.tight_layout()
    for idx,img in enumerate(imgs):
        axes[idx].imshow(img)
    wandb.log({category:wandb.Image(fig)}) 