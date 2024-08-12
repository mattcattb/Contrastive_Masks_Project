from data.coco_loader import Custom_Coco_Dataset
from torchvision import transforms
from torch.utils.data import Dataset

from typing import Tuple


def get_transforms(train: bool, size=(256, 256)) -> transforms:
    # todo: make use args to make this...

    transforms_list = []
    
    transforms_list.append(transforms.Resize(size))
    transforms_list.append(transforms.ToTensor())
    if train:
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    

    return transforms.Compose(transforms_list)

def load_datasets(args) -> Tuple[Dataset, Dataset]:

    train_transform= get_transforms(True)

    # load training adn validation dataset 
    training_dataset = Custom_Coco_Dataset(
        args['data']['train']['images_path'], 
        args["data"]['train']['annotations_path'],
        transform=train_transform)
    
    args['num_classes'] = len(training_dataset.coco.cats) + 1 # + 1 is background?

    val_transform =get_transforms(False)

    val_dataset = Custom_Coco_Dataset(
        args['data']['val']['images_path'], 
        args["data"]['val']['annotations_path'],
        transform=val_transform)   

    return training_dataset, val_dataset
