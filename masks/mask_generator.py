from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
import numpy as np
import torch
 
from .fastsam import FastSAM, FastSAMPrompt
from data import SSLTFDataset

DEFAULT_SAM_PATH = "model/FastSAM-x.pt"

class MaskGenerator:
    def create_mask(self, image):
        """
            image should be a c,h,w tensor
            returns: a (h,w) mask 
        """
        raise NotImplementedError

class FHMaskGenerator(MaskGenerator):
    def __init__(self, scale=1000, min_size=1000):
        self.scale = scale
        self.min_size = min_size

    def create_mask(self, image):
        # image should be c, h, w

        # turn to h,w,c for felzenszwalb
        mask = felzenszwalb(image.permute(1, 2, 0), scale=self.scale, min_size=self.min_size)
        return torch.tensor(mask).int()

class PatchMaskGenerator(MaskGenerator):

    def __init__(self, segments=[3, 3]):
        self.segments = segments

    def create_mask(self, image):
        """
            img should be c,h,w
        """
        dims = list(np.floor_divide(image.shape[1:], self.segments))
        mask = torch.hstack([torch.cat([torch.zeros(dims[0], dims[1]) + i + (j * (self.segments[0]))
                                        for i in range(self.segments[0])]) for j in range(self.segments[1])])
        mods = list(np.mod(image.shape[1:], self.segments))
        if mods[0] != 0:
            mask = torch.cat([mask, torch.stack([mask[-1, :] for _ in range(mods[0])])])
        if mods[1] != 0:
            mask = torch.hstack([mask, torch.stack([mask[:, -1] for _ in range(mods[1])]).T])
        return mask.int()

class SAMMaskGenerator(MaskGenerator):
    def __init__(self, model_path= DEFAULT_SAM_PATH, conf=0.4, iou=0.9, retina_masks=True):
        """
        retina_masks=True determines whether the model uses retina masks for generating segmentation masks.
        imgsz=1024 sets the input image size to 1024x1024 pixels for processing by the model.
        conf=0.4 sets the minimum confidence threshold for object detection.
        iou=0.9 sets the minimum intersection over union threshold for non-maximum suppression to filter out duplicate detections.
        """
        self.retina_masks = retina_masks
        self.conf = conf 
        self.iou = iou
        self.sam_model = FastSAM(model_path)

    def create_mask(self, image_path:str):
        all_results = self.sam_model(image_path, conf=self.conf, iou=self.iou, retina_masks=self.retina_masks)
        prompt_process = FastSAMPrompt(image_path, results=all_results, device='cpu')
        ann = prompt_process.everything_prompt()
        # PROBLEM!!! Ann has too many masks masks...
        ann = self.process_masks(ann)
        return torch.tensor(ann).int()
    
    def process_masks(self, mask):
        """
            takes a mask of [c, h, w], 
            and converts to a h,w with each number being a seperate mask.
            returns a (h,w)tensor int8
        
        """


        return mask