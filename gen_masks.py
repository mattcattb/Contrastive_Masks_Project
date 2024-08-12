import argparse
from masks.mask_preloader import Preload_Masks, MaskType

parser = argparse.ArgumentParser(description='Detcon-Mask Generation')

parser.add_argument("--dataset_dir", metavar="Dataset Directory", type=str, default="/path/to/imagenet/images/train", 
                    help="Dataset directory location, use dataset/train, dataset/val, dataset/test.")

parser.add_argument("--output_dir", metavar="Output Directory", default="/path/to/imagenet/masks/", 
                    help="Output location of masks.")

parser.add_argument("--mask_type", metavar="Mask Type", default="fh", 
                    help="Type of mask, select from: fh, patch, ground.")

parser.add_argument("--experiment_name", metavar="Experiment Name", default="train_tf", 
                    help="Name of experiment, determines output filename.")

parser.add_argument("--ground_mask_dir", metavar="Ground Mask Directory", default="", 
                    help="Directory to load ground truth mask for testing.")
    
import os

valid_masks = ["fh", "patch", "sam"]

def generate_masks_dataset(output_dir, mask_type='fh'):
    """
        Generate mask_type for val, training, and unlabeled images
    
    """
    # mask_types has fh, patch, ground 

    # dataset_dir, output_dir, mask_type, experiment_name, scale, min_size, segments
    imagenet_folder = "/media/mattyb/UBUNTU 22_0/datasets/imagenet_strawberries"
    dir_names = ["train", "unlabeled", "val"]


    for dir_name in dir_names:
        dir_path = os.path.join(imagenet_folder, "images", dir_name)
        experiment_name =  f"exp_{dir_name}_{mask_type}"
        output = os.path.join(output_dir, experiment_name)  
        mask_loader = Preload_Masks(dir_path, output,mask_type=mask_type)
        mask_loader.forward()
    
    # this should create masks for all images
def get_mask_enum(mask_type:str)-> MaskType:
    if mask_type == "fh":
        return MaskType.FH
    elif mask_type == "ground":
        return MaskType.GROUND
    elif mask_type == "sam":
        return MaskType.SAM
    elif mask_type == "patch":
        return MaskType.PATCH

if __name__=="__main__":
    args = vars(parser.parse_args())
    args["mask_type"] = get_mask_enum(args["mask_type"])
    mask_loader = Preload_Masks(**args)
    mask_loader.forward()
