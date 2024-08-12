import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(args:dict) -> torchvision.models:
    #! SETUP MODEL

    model_name = args['model']['model_type']

    if model_name == 'faster_rcnn':

        pretrain = args['model']['pretrained']

        if pretrain == "No":
            weights = None
        elif pretrain == "Default":
            weights = "DEFAULT"
        else:
            weights = None

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args['num_classes'])
    
    else:
        model = None
    
    return model
