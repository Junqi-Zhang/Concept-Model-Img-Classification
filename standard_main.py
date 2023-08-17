import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import PROVIDED_MODELS
from utils import save, load

##########################
# Basic settings
##########################

# USE_DATA_FOLDER = "Caltech-256"
USE_DATA_FOLDER = "Sampled_ImageNet"

# USE_MODEL = "ResNet18"
USE_MODEL = "BasicQuantResNet18"

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 250
num_concepts = 64
n_epoch = 20
batch_size = 256
learning_rate = 1e-3

train_data = os.path.join("./data/", USE_DATA_FOLDER, "train")
eval_data = os.path.join("./data/", USE_DATA_FOLDER, "val")
eval_major_data = os.path.join("./data/", USE_DATA_FOLDER, "major_val")
eval_minor_data = os.path.join("./data/", USE_DATA_FOLDER, "minor_val")
_time = time.strftime("%Y%m%d%H", time.localtime(time.time()))

checkpoint_dir = os.path.join("./logs/", USE_DATA_FOLDER, USE_MODEL, _time)
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset from the directory
train_dataset = ImageFolder(root=train_data, transform=transform)
eval_dataset = ImageFolder(root=eval_data, transform=transform)

# major_val 和 minor_val 的类别是 train 和 val 的子集
tmp_major_dataset = ImageFolder(root=eval_major_data, transform=transform)
tmp_minor_dataset = ImageFolder(root=eval_minor_data, transform=transform)


def major_to_train(target):
    idx_transform = dict()
    for key, value in tmp_major_dataset.class_to_idx.items():
        idx_transform[value] = train_dataset.class_to_idx[key]
    return idx_transform[target]


def minor_to_train(target):
    idx_transform = dict()
    for key, value in tmp_minor_dataset.class_to_idx.items():
        idx_transform[value] = train_dataset.class_to_idx[key]
    return idx_transform[target]


eval_major_dataset = ImageFolder(
    root=eval_major_data, transform=transform, target_transform=major_to_train)
eval_minor_dataset = ImageFolder(
    root=eval_minor_data, transform=transform, target_transform=minor_to_train)


# Create DataLoader instances
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
eval_loader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
eval_major_loader = DataLoader(
    eval_major_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
eval_minor_loader = DataLoader(
    eval_minor_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

##########################
# Model, loss and optimizer
##########################

model = PROVIDED_MODELS[USE_MODEL](num_classes, num_concepts).to(device)
criterion = nn.CrossEntropyLoss()


def compute_loss(returned_dict, targets):
    outputs = returned_dict["outputs"]

    # 代码兼容 ResNet18
    attention_weights = returned_dict.get("attention_weights", None)
    concept_similarity = returned_dict.get("concept_similarity", None)

    if (attention_weights is None) and (concept_similarity is None):
        return criterion(outputs, targets)

    # 熵越小, 分布越不均匀
    attention_entropy = torch.mean(-torch.sum(attention_weights *
                                              torch.log2(attention_weights), dim=1))

    # 防止 concept 退化, concept 之间要近似正交
    def concept_diversity_reg():
        # ideal_similarity 可以调整, 单位阵的假设过强
        ideal_similarity = torch.eye(
            num_concepts,
            dtype=torch.float,
            device=device
        )
        return torch.norm(concept_similarity-ideal_similarity)

    # return criterion(outputs, targets) + attention_entropy + concept_diversity_reg()
    # return criterion(outputs, targets) + concept_diversity_reg()
    return criterion(outputs, targets)


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

    metric_dict = {"loss": 0.0, "acc": 0.0}
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
                returned_dict = model(data)
                loss = compute_loss(returned_dict, targets)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    returned_dict = model(data)
                    loss = compute_loss(returned_dict, targets)

            # display the metrics
            with torch.no_grad():
                acc = (torch.argmax(returned_dict["outputs"].data, 
                                    1) == targets).sum() / targets.size(0)
            metric_dict["loss"] = (metric_dict["loss"] *
                                   step + loss.item()) / (step + 1)
            metric_dict["acc"] = (metric_dict["acc"] *
                                  step + acc.item()) / (step + 1)
            pbar.set_postfix(
                **{
                    "loss": metric_dict["loss"],
                    "acc": metric_dict["acc"],
                }
            )
            pbar.update(1)

            step += 1
    return metric_dict


for epoch in range(n_epoch):
    # train one epoch
    desc = f"Training epoch {epoch + 1}/{n_epoch}"
    train_dict = run_epoch(desc, model, train_loader, train=True)

    # validation
    desc = "      Validataion"
    eval_dict = run_epoch(desc, model, eval_loader, train=False)

    desc = "Major Validataion"
    eval_major_dict = run_epoch(desc, model, eval_major_loader, train=False)

    desc = "Minor Validataion"
    eval_minor_dict = run_epoch(desc, model, eval_minor_loader, train=False)

    # model checkpoint
    model_name = "epoch_%d_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f.pt" % (
        epoch + 1,
        train_dict["loss"],
        train_dict["acc"],
        eval_dict["loss"],
        eval_dict["acc"],
        eval_major_dict["loss"],
        eval_major_dict["acc"],
        eval_minor_dict["loss"],
        eval_minor_dict["acc"]
    )
    save(model, os.path.join(checkpoint_dir, model_name))
