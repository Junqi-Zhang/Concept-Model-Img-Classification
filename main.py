import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import ResNet18
from utils import save, load

##########################
# Basic settings
##########################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 256
n_epoch = 100
batch_size = 64
learning_rate = 1e-3

train_data = 'data/Caltech-256'
eval_data = 'data/Caltech-256'
_time = time.strftime('%Y%m%d%H', time.localtime(time.time()))
checkpoint_dir = 'logs/' + _time

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

##########################
# Dataset and DataLoader
##########################
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset from the directory
train_dataset = ImageFolder(root=train_data, transform=transform)
eval_dataset = ImageFolder(root=eval_data, transform=transform)

# Create DataLoader instances
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
eval_loader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

##########################
# Model, loss and optimizer
##########################
model = ResNet18(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


##########################
# Training pipeline
##########################
def run_epoch(desc, model, dataloader, train=False):
    # train pipeline
    if train:
        model.train()
    else:
        model.eval()
    
    metric_dict = {'loss': 0.0, 'acc': 0.0}
    step = 0
    with tqdm(
        total=len(dataloader),
        desc=desc,
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for data, targets in dataloader:
            # data process
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            if train:
                outputs = model(data)
                loss = criterion(outputs, targets)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(data)
                    loss = criterion(outputs, targets)

            # display the metrics
            with torch.no_grad():
                acc = (torch.argmax(outputs.data, 1) == targets).sum() / targets.size(0)
            metric_dict['loss'] = (metric_dict['loss'] * step + loss.item()) / (step + 1)
            metric_dict['acc'] = (metric_dict['acc'] * step + acc.item()) / (step + 1)
            pbar.set_postfix(
                **{
                    'loss': metric_dict['loss'],
                    'acc': metric_dict['acc'],
                }
            )
            pbar.update(1)

            step += 1
    return metric_dict


for epoch in range(n_epoch):
    # train one epoch
    desc = f'Training epoch {epoch + 1}/{n_epoch}'
    train_dict = run_epoch(desc, model, train_loader, train=True)

    # validation
    desc = '  Validataion'
    eval_dict = run_epoch(desc, model, eval_loader, train=False)

    # model checkpoint
    model_name = 'epoch_%d_%.4f_%.4f_%.4f_%.4f.pt' % (
        epoch,
        train_dict['loss'],
        train_dict['acc'],
        eval_dict['loss'],
        eval_dict['acc'],
    )
    save(model, os.path.join(checkpoint_dir, model_name))
