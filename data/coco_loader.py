from torch.utils.data import Dataset
import os
from PIL import Image
from pycocotools.coco import COCO
import torch
import json

"""
    REMEMBER Coco annotations are [xmin, ymin, width, height], full values

    have coco also have area of annotation
    
"""

from typing import List, Dict, Union

class Custom_Coco_Dataset(Dataset):
    def __init__(self, img_dir:str, annotation_file:str, transform=None):
        self.img_dir: str = img_dir
        self.annotation_file: str = annotation_file
        self.transform = transform
        self.coco: COCO = COCO(annotation_file)
        self.ids: List[int] = list(self.coco.imgs.keys())
        # preload image info and annotations
        self.image_info: Dict[int, dict] = {img_id:self.coco.loadImgs(img_id)[0] for img_id in self.ids} # type: ignore # map each ID to the info of that ID
        # map of {img_id : img_objects}

        self.annotations: Dict[int, list] = {img_id : self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)) for img_id in self.ids}
        # dict of {img_id: [anno_1, anno_2 ...]} list of all annotations for that image

    def __getitem__(self, index):

        """
            returns image, results

            results :{
                boxes: tensor of [[xmin,ymin, xmax, ymax]...]
                label : [l1, l2 ...]
                image_id : torch.tensor([img_id])
                area : []
                iscrowd = torch.zeros((len(anns)), dtype=torch.int64)
            }
            all results are pytorch
        
        """
        img_id: int = self.ids[index]
        anns: List[Dict] = self.annotations[img_id] # gets annotation from img_id
        image_info: Dict = self.image_info[img_id]
        image_path: str = os.path.join(self.img_dir, image_info['file_name'])

        try:
            image = Image.open(image_path).convert("RGB")

        except IOError:
            print(f"Error loading image: {image_path}")
            return None
        
        img_width, img_height = image.size

        boxes = []
        labels = []

        for ann in anns:
            # remember, x,y,w,h are already scaled!
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        boxes_tensor: torch.Tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor: torch.Tensor = torch.as_tensor(labels, dtype=torch.int64)
        areas: torch.Tensor = (boxes_tensor[:,3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        is_crowd: torch.Tensor = torch.zeros((len(anns)), dtype=torch.int64)
        
        target: dict = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": is_crowd
        }

        
        if self.transform:
            image = self.transform(image)
            # resize bboxes
            if "Resize" in str(self.transform):
                orig_size = torch.tensor([img_width, img_height], dtype=torch.float32)
                new_size = torch.tensor(image.shape[-2:], dtype=torch.float32)
                scale_factor = new_size / orig_size
                target["boxes"][:, [0, 2]] *= scale_factor[1]
                target["boxes"][:, [1, 3]] *= scale_factor[0]
        
        return image, target

    def __len__(self):
        return len(self.ids)




class Custom_Labeled_Dataset(Dataset):

    def __init__(self, root_dir:str, anno_file:str, transform= None):
        self.root_dir = root_dir
        self.transform = transform

        with open(anno_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.image_to_annotations)

    def __getitem__(self, index:int):
        
        # get image
        img_info = self.images[index]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # get image annotations
        image_id = img_info['id']
        annotations = self.image_to_annotations.get(image_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
        
        # convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes':boxes,
            'labels':labels,
            'image_id': torch.tensor([image_id])
        }
        
        return image, target

class Custom_Unlabeled_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, None
    
def main():
    
    # test out going through dataset
    train_dataset = Custom_Labeled_Dataset("coco_strawberries/images/train", "coco_strawberries/annotations/instances_train.json")

    for i in range(len(train_dataset)):
        
        print(train_dataset[i])

    unlabeled_dataset = Custom_Unlabeled_Dataset("coco_strawberries/images/unlabeled")

    for i in range(len(unlabeled_dataset)):
        print(unlabeled_dataset[i])


    pass

if __name__ == "__main__":
    main()