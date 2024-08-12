import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_coco_stats(annotation_file, results) -> dict:
    
    coco_gt:COCO = COCO(annotation_file)
    coco_dt:COCO = coco_gt.loadRes(results)
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats = {
        'Average Precision (AP) @ IoU=0.50:0.95': cocoEval.stats[0],
        'Average Precision (AP) @ IoU=0.50': cocoEval.stats[1],
        'Average Precision (AP) @ IoU=0.75': cocoEval.stats[2],
        'Average Precision (AP) for small objects': cocoEval.stats[3],
        'Average Precision (AP) for medium objects': cocoEval.stats[4],
        'Average Precision (AP) for large objects': cocoEval.stats[5],
        'Average Recall (AR) @ IoU=0.50:0.95': cocoEval.stats[6],
        'Average Recall (AR) for small objects': cocoEval.stats[7],
        'Average Recall (AR) for medium objects': cocoEval.stats[8],
        'Average Recall (AR) for large objects': cocoEval.stats[9]
    }
    
    return stats

from typing import Dict, List

def validate_model(args, model, val_dataset) -> dict:

    batch_size = args['validation']['batch_size']
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model.eval()
    results: List[Dict] = []
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = list(img.to(device) for img in images)
            outputs:list = model(images) # list of dictionaries for each output, has bboxes, labels, scores

            images = [img.cpu() for img in images]

            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().tolist()
                labels = output['labels'].cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    width, height = xmax -xmin, ymax- ymin
                    results.append({
                        'image_id': image_id,
                        'category_id': label,
                        'bbox':[xmin, ymin, width, height],
                        'score':score
                    }) 

    torch.cuda.empty_cache()

    stats = get_coco_stats(val_dataset.annotation_file, results)
    return stats
