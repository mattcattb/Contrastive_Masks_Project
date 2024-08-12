import matplotlib.pyplot as plt 
import os
import torch
import json
import yaml

import random
import string
import shutil

import logging
import sys


from datetime import datetime

def plot_losses(path, losses:list, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def generate_random_label():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def generate_timestamp_label():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_experiment_folder(base_path:str) -> str:
    label = generate_timestamp_label()
    experiment_dir = os.path.join(base_path, label) + '/'
    os.makedirs(experiment_dir)

    print(f"Logging to:{experiment_dir}")
    return experiment_dir

def save_logs(args, results:dict, losses:list, model) -> None:

    # save code!

    base_path = args['base_path']
    experiments_folder= make_experiment_folder(base_path)
    num_epochs = args['training']['epochs']

    # create code copy
    code_dir = os.path.join(experiments_folder, "code")
    os.makedirs(code_dir)

    for item in os.listdir('.'):
        if item.endswith('.py'):  # Only copy Python files, adjust as needed
            shutil.copy2(item, os.path.join(code_dir, item))


    # save args as config 
    config_path = os.path.join(experiments_folder, "config.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(experiments_folder, "model.pth"))

    # Plot Losses
    fig_path = os.path.join(experiments_folder, 'loss_graph.png')
    plot_losses(fig_path, losses, num_epochs)
    with open(os.path.join(experiments_folder,'loss_data.json'), 'w') as f:
       json.dump(losses, f)
    
    # dump validation results
    with open(os.path.join(experiments_folder, 'validation_results.json'), 'w') as f:
        json.dump(results, f)


def get_std_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    return logging
