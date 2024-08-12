from torch.utils.data import DataLoader
import torch
from typing import List, Any, Tuple
from torch.utils.data import Dataset

from tqdm import tqdm

def collate_fn(batch)->tuple:
    return tuple(zip(*batch))

def choose_device(args:dict) -> torch.device:

    if (args['training']['device'] == 'gpu') and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def generate_dataloader(args, train_dataset):
    # creates specfic dataloader from config file
    batch_size = args['training']['batch_size']
    num_workers = args['training']['num_workers']
    shuffle = args['training']['shuffle']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader

def generate_optimizer(args, model):
    weight_decay = args['training']['optimizer']['weight_decay']
    learning_rate = args['training']['optimizer']['learning_rate']
    momentum = args['training']['optimizer']['momentum']

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    return optimizer

def train(args: dict, train_dataset: Dataset, model) -> Tuple [Any, List]:

    num_epochs = args['training']['epochs']
    accumulate_grad_batches = args.get('training', {}).get('accumulate', 1)
    
    train_dataloader = generate_dataloader(args, train_dataset)
    optimizer = generate_optimizer(args, model)

    device = choose_device(args)
    
    losses_history = []

    epoch_progress = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    model.to(device)
    print("starting training!")    

    for epoch in epoch_progress:
        
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch{epoch + 1}/{num_epochs}", leave=True)
        for i, (images, targets) in progress_bar:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict:dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            # Normalize the loss to account for accumulation
            losses = losses / accumulate_grad_batches
            losses.backward()

            epoch_loss += loss_value

            if (i + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()        
                progress_bar.set_postfix({'loss_value':f"{loss_value:.4f}"})

        if (len(train_dataloader) % accumulate_grad_batches) != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss/len(train_dataloader)
        losses_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, avg_loss: {avg_loss}")

    torch.cuda.empty_cache()

    # return results
    return model, losses_history