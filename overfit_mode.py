import os
import json
import time
import argparse
from tqdm import tqdm
from pprint import pprint

import numpy as np  # 在 import torch 前
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_folders import PROVIDED_DATA_FOLDERS
from models import PROVIDED_MODELS
from utils import save, load


##########################
# Argument parsing
##########################

parser = argparse.ArgumentParser(
    description="Test model's ability to overfit a tiny dataset."
)

parser.add_argument("--data_folder", required=True)

parser.add_argument("--model", required=True)
parser.add_argument("--num_concepts", default=64, type=int)
parser.add_argument("--loss_sparsity_weight", default=0.0, type=float)
parser.add_argument("--loss_diversity_weight", default=0.0, type=float)

parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)

parser.add_argument("--save_interval", default=1, type=int)

# 以下参数以(arg.参数名)的方式进行调用
parser.add_argument("--supplementary_description", default=None)
parser.add_argument("--summary_log_path", required=True)
parser.add_argument("--detailed_log_path", required=True)

args = parser.parse_args()

##########################
# Basic settings
##########################

device = "cuda" if torch.cuda.is_available() else "cpu"

use_data_folder = args.data_folder
use_data_folder_info = PROVIDED_DATA_FOLDERS[use_data_folder]
num_classes = use_data_folder_info["num_classes"]

use_model = args.model
num_concepts = args.num_concepts
loss_sparsity_weight = args.loss_sparsity_weight
loss_diversity_weight = args.loss_diversity_weight

n_epoch = args.num_epochs
batch_size = args.batch_size
learning_rate = 1e-3

save_interval = args.save_interval

_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
checkpoint_dir = os.path.join(
    "./checkpoints/", use_data_folder, use_model, _time)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Confirm basic settings
print("\n"+"#"*100)
print(f"# Desc: {args.supplementary_description}")
print(f"# Use data_folder: {use_data_folder}, {num_classes} classes in total.")
print(f"# Use model: {use_model}, includes {num_concepts} concepts.")
print(f"# Weight for concept  sparsity loss is {loss_sparsity_weight:.4f}.")
print(f"# Weight for concept diversity loss is {loss_diversity_weight:.4f}.")
print(f"# Train up to {n_epoch} epochs, with barch_size={batch_size}.")
print(f"# Save model's checkpoint for every {save_interval} epochs,")
print(f"# checkpoints locate in {checkpoint_dir}.")
print("#"*100)

# 使用相同一个tiny数据集训练并验证
train_data = use_data_folder_info["train_folder_path"]
print(f"# Train on data from {train_data}.")
eval_data = use_data_folder_info["val_folder_path"]
print(f"# Evaluate on data from {eval_data}.")
print("#"*100+"\n")

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

# Create DataLoader instances
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)
eval_loader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)

##########################
# Model, loss and optimizer
##########################

model = PROVIDED_MODELS[use_model](num_classes, num_concepts).to(device)
criterion = nn.CrossEntropyLoss()


def compute_loss(returned_dict, targets):
    outputs = returned_dict["outputs"]

    # 代码兼容 ResNet18 等常规模型
    attention_weights = returned_dict.get("attention_weights", None)
    concept_similarity = returned_dict.get("concept_similarity", None)

    if (attention_weights is None) and (concept_similarity is None):
        loss_classification = criterion(outputs, targets)
        loss_sparsity = torch.tensor(0.0)
        loss_diversity = torch.tensor(0.0)
        loss = loss_classification + loss_sparsity_weight * \
            loss_sparsity + loss_diversity_weight * loss_diversity
        return loss, loss_classification, loss_sparsity, loss_diversity

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

    loss_classification = criterion(outputs, targets)
    loss_sparsity = attention_entropy
    loss_diversity = concept_diversity_reg()

    loss = loss_classification + loss_sparsity_weight * \
        loss_sparsity + loss_diversity_weight * loss_diversity

    return loss, loss_classification, loss_sparsity, loss_diversity


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

    metric_dict = {
        "loss": 0.0,
        "loss_classification": 0.0,
        "loss_sparsity": 0.0,
        "loss_diversity": 0.0,
        "acc": 0.0
    }

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
                loss, loss_classification, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    returned_dict = model(data)
                    loss, loss_classification, loss_sparsity, loss_diversity = compute_loss(
                        returned_dict, targets
                    )

            # display the metrics
            with torch.no_grad():
                acc = (torch.argmax(returned_dict["outputs"].data,
                                    1) == targets).sum() / targets.size(0)
            metric_dict["loss"] = (metric_dict["loss"] *
                                   step + loss.item()) / (step + 1)
            metric_dict["loss_classification"] = (metric_dict["loss_classification"] *
                                                  step + loss_classification.item()) / (step + 1)
            metric_dict["loss_sparsity"] = (metric_dict["loss_sparsity"] *
                                            step + loss_sparsity.item()) / (step + 1)
            metric_dict["loss_diversity"] = (metric_dict["loss_diversity"] *
                                             step + loss_diversity.item()) / (step + 1)
            metric_dict["acc"] = (metric_dict["acc"] *
                                  step + acc.item()) / (step + 1)
            pbar.set_postfix(
                **{
                    "loss": metric_dict["loss"],
                    "loss_cls": metric_dict["loss_classification"],
                    "loss_sps": metric_dict["loss_sparsity"],
                    "loss_dvs": metric_dict["loss_diversity"],
                    "acc": metric_dict["acc"]
                }
            )
            pbar.update(1)

            step += 1
    return metric_dict


early_stopped = False
early_stop_counter = 3

best_val_acc = 0
best_epoch = 0
best_checkpoint_path = ""

for epoch in range(n_epoch):
    # train one epoch
    desc = f"Training epoch {epoch + 1}/{n_epoch}"
    train_dict = run_epoch(desc, model, train_loader, train=True)

    # validation
    desc = "      Validataion"
    eval_dict = run_epoch(desc, model, eval_loader, train=False)

    # model checkpoint
    model_name_elements = [
        "epoch",
        f"{epoch + 1}",
        f"{train_dict['loss']:.4f}",
        f"{train_dict['acc']:.4f}",
        f"{eval_dict['loss']:.4f}",
        f"{eval_dict['acc']:.4f}"
    ]
    model_name = "_".join(model_name_elements) + ".pt"

    if eval_dict['acc'] > best_val_acc:
        best_val_acc = eval_dict['acc']
        best_epoch = epoch + 1
        best_checkpoint_path = os.path.join(checkpoint_dir, model_name)

    if epoch % save_interval == 0:
        save(model, os.path.join(checkpoint_dir, model_name))

    if eval_dict["acc"] > 0.95:
        early_stop_counter -= 1

    if early_stop_counter == 0:
        early_stopped = True
        break

save(model, os.path.join(checkpoint_dir, model_name))

if early_stopped:
    print("Early stopped.")
else:
    print("Normal stopped.")

#####################################
# Save summary and detailed logs
#####################################

print(f"Summary log to be saved in {args.summary_log_path}")
print(f"Detailed log to be saved in {args.detailed_log_path}")

log_elements = {
    "date": time.strftime("%Y%m%d", time.localtime(time.time())),
    "mode": "overfit",
    "data_folder": use_data_folder,
    "model": use_model,
    "num_concepts": num_concepts,
    "loss_sparsity_weight": loss_sparsity_weight,
    "loss_diversity_weight": loss_diversity_weight,
    "supplementary_description": args.supplementary_description,
    "num_epochs": n_epoch,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "save_interval": save_interval,
    "checkpoint_dir": checkpoint_dir,
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "best_checkpoint_path": best_checkpoint_path,
    "detailed_log_path": args.detailed_log_path
}

pprint(log_elements)

with open(args.summary_log_path, "a") as f:
    f.write(json.dumps(log_elements))
    f.write("\n")

print("END. Checkpoints and logs are saved.")
