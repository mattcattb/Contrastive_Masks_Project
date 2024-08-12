from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
import numpy as np
import pickle
import torch
import os
import time

import matplotlib.image as mpimg
from tqdm import tqdm
from enum import Enum
 
from .fastsam import FastSAM, FastSAMPrompt
from data import SSLTFDataset
from .mask_generator import MaskGenerator, FHMaskGenerator, PatchMaskGenerator, SAMMaskGenerator
class MaskType(Enum):
    FH = "fh"
    PATCH = "patch"
    SAM = "sam"
    GROUND = "ground"

DEFAULT_SAM_PATH = "model/FastSAM-x.pt"
    
class Preload_Masks():
    def __init__(self,dataset_dir,output_dir,ground_mask_dir='', mask_type=MaskType.FH, experiment_name='',
                 num_threads=os.cpu_count(),scale=1000,min_size=1000,segments=[3,3], sam_path = DEFAULT_SAM_PATH):
        
        self.output_dir=output_dir
        self.mask_type=mask_type
        self.scale = scale
        self.min_size = min_size
        self.segments = segments
        self.experiment_name = experiment_name
        self.num_threads = num_threads
        self.sam_path = sam_path
        self.ground_mask_dir = ground_mask_dir
        self.image_dataset = SSLTFDataset(dataset_dir)
        self.ds_length = len(self.image_dataset)
        self.save_path = os.path.join(self.output_dir,self.experiment_name)
        self.mask_gen = self.create_mask_generator()

    def create_mask_generator(self)->MaskGenerator:

        if self.mask_type == MaskType.FH:
            return FHMaskGenerator(scale=self.scale, min_size=self.min_size)
        elif self.mask_type == MaskType.SAM:
            return SAMMaskGenerator(self.sam_path)
        elif self.mask_type == MaskType.PATCH:
            return PatchMaskGenerator(segments=self.segments)
    
    def make_mask(self,obj):
        image,label,img_path = obj
        suffix = '_'+self.mask_type.value+'.pkl'
        name = os.path.join(self.save_path,os.path.splitext('_'.join(img_path.split('/')[-2:]))[0])
        
        # if SAM, use the image path, not the image
        if self.mask_type == MaskType.SAM:
            input = img_path
        else:
            input = image
        mask = self.mask_gen.create_mask(image=input).to(dtype=torch.int16)
        with open(name+suffix, 'wb') as handle:
            pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [img_path,name+suffix]
    
    def pkl_save(self,file,name):
        with open(name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_dicts(self,img_paths,mask_paths):
        self.pkl_save(mask_paths,os.path.join(self.output_dir,self.experiment_name+'_img_to_'+self.mask_type.value+'.pkl'))
        return
    
    
    def forward(self):
        try:
            os.mkdir(os.path.join(self.output_dir,self.experiment_name))
        except:
            if not os.path.exists(self.output_dir):
                os.makedirs(os.path.join(self.output_dir,self.experiment_name))
                
        print('Dataset Length: %d  '%(self.ds_length))
        start = time.time()
        img_paths,mask_paths = zip(*Parallel(n_jobs=self.num_threads,prefer="threads")
                                 (delayed(self.make_mask)(obj) for obj in tqdm(self.image_dataset)))
        end = time.time()

        self.save_dicts(img_paths,mask_paths)

        print('Time Taken: %f  '%((end - start)/60))
        
        return 