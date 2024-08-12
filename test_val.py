import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image

def test_validation():
    # Create a model with COCO pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()

    # Create a dummy image and ground truth
    image = torch.rand(3, 300, 400)
    image_path = "/media/mattyb/UBUNTU 22_0/datasets/archive (1)/images/000000000692.jpg"
    image = Image.open(image_path)
    
    target = {
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'image_id': torch.tensor([1])
    }

    # Get model prediction
    with torch.no_grad():
        prediction = model([image])[0]

    # Convert prediction to COCO format
    coco_prediction = {
        'image_id': target['image_id'].item(),
        'category_id': prediction['labels'][0].item(),
        'bbox': prediction['boxes'][0].tolist(),
        'score': prediction['scores'][0].item()
    }

    # Convert ground truth to COCO format
    coco_gt = {
        'annotations': [{
            'id': 1,
            'image_id': target['image_id'].item(),
            'category_id': target['labels'][0].item(),
            'bbox': target['boxes'][0].tolist(),
            'area': (target['boxes'][0][2] - target['boxes'][0][0]) * (target['boxes'][0][3] - target['boxes'][0][1]),
            'iscrowd': 0
        }],
        'images': [{'id': target['image_id'].item()}],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    # Create COCO objects
    coco_gt = COCO()
    coco_gt.dataset = coco_gt
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes([coco_prediction])

    # Create COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = [target['image_id'].item()]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print mAP
    print(f"mAP: {coco_eval.stats[0]}")

if __name__ == "__main__":
    test_validation()