#-*- coding:utf-8 -*-
import os
import yaml
import torch

from trainer.detconb_trainer import DetconBTrainer
from utils import logs
import argparse

parser = argparse.ArgumentParser(description='Detcon-BYOL Training')

parser.add_argument("--cfg", metavar="Config Filename", default="train_imagenet_300", 
                    help="Experiment to run. Default is Imagenet 300 epochs")


def run_task(config):
    logging = logs.get_std_logging()

    config['world_size'] = 1
    config["rank"] = 0
    config['local_rank'] = 0

    trainer = DetconBTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    for epoch in range(start_epoch + 1, trainer.total_epochs + 1):
        trainer.train_epoch(epoch, printer=logging.info)
        trainer.save_checkpoint(epoch)

def main():
    args = parser.parse_args()
    args.local_rank = 0
    
    cfg = args.cfg if args.cfg[-5:] == '.yaml' else args.cfg + '.yaml'
    config_path = os.path.join(os.getcwd(), 'config', cfg)
    assert os.path.exists(config_path), f"Could not find {cfg} in configs directory!"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.local_rank==0:
        print("=> Config Details")
        print(config) #For reference in logs
    
    run_task(config)

if __name__ == "__main__":
    main()
