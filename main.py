import argparse
import yaml

import src.models as models
import src.training as training
import src.validation as validation
import src.logs as logs
import src.dataset_creation as dataset_creation

from typing import List, Dict, Any

"""

    This script will TRAIN A RCNN MODEL on the testing dataset

    todo
    - get annotations from dataset into COCO() from cocotools
    //- transformations to fit input
    //- work out classes in dataset 
    // - graph loss over time and save that
    // - add accumulation gradient
    - allow detection model customization
    - seperate into different filesystems
    - try to contrastive pretrain alexnet and then add detection head

"""

def run_experiment(args):
    train_dataset, val_dataset = dataset_creation.load_datasets(args)
    
    model: Any
    model = models.get_model(args)
    losses: List[float]
    model, losses = training.train(args, train_dataset, model)
    results:Any = validation.validate_model(args, model, val_dataset)
    logs.save_logs(args, results, losses, model)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="path to YAML config file",
                        default="basic_test.yaml")
    parser.add_argument("--output", type=str,
                        help="path to output experiment logs",
                        default='experiments_outputs')    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        yaml_args = yaml.safe_load(f)

    yaml_args['base_path'] = args.output
    
    run_experiment(yaml_args)

if __name__ == "__main__":
    main()