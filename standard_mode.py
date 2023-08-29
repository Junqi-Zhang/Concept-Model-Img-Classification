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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from data_folders import PROVIDED_DATA_FOLDERS
from models import PROVIDED_MODELS
from utils import save, load


##########################
# Argument parsing
##########################

parser = argparse.ArgumentParser(
    description="Test model's ability of generalization."
)

parser.add_argument("--data_folder", required=True)

parser.add_argument("--model", required=True)
parser.add_argument("--num_concepts", default=64, type=int)
parser.add_argument("--norm_concepts", default="False")
parser.add_argument("--norm_summary", default="False")
parser.add_argument("--loss_sparsity_weight", default=0.0, type=float)
parser.add_argument("--loss_diversity_weight", default=0.0, type=float)

parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)

parser.add_argument("--save_interval", default=1, type=int)

# 以下参数以(arg.参数名)的方式进行调用
parser.add_argument("--supplementary_description", default=None)
parser.add_argument("--summary_log_path", required=True)
parser.add_argument("--detailed_log_path", required=True)
parser.add_argument("--gpu", type=int, required=True)

args = parser.parse_args()

##########################
# Basic settings
##########################

if torch.cuda.is_available():
    device = "cuda"
    # torch.cuda.set_device(args.gpu)
else:
    device = "cpu"

use_data_folder = args.data_folder
use_data_folder_info = PROVIDED_DATA_FOLDERS[use_data_folder]
num_classes = use_data_folder_info["num_classes"]

use_model = args.model
num_concepts = args.num_concepts
norm_concepts = eval(args.norm_concepts)
norm_summary = eval(args.norm_summary)
loss_sparsity_weight = args.loss_sparsity_weight
loss_diversity_weight = args.loss_diversity_weight

n_epoch = args.num_epochs
batch_size = args.batch_size
learning_rate = 1e-3
warmup_epochs = 10

save_interval = args.save_interval

_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
checkpoint_dir = os.path.join(
    "./checkpoints/", use_data_folder, use_model, _time+f"_on_gpu_{args.gpu}")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Confirm basic settings
print("\n"+"#"*100)
print(f"# Desc: {args.supplementary_description}")
print(f"# Use data_folder: {use_data_folder}, {num_classes} classes in total.")
print(f"# Use model: {use_model}, includes {num_concepts} concepts.")
print(f"# Norm Concepts: {norm_concepts}, Norm Summary: {norm_summary}.")
print(f"# Weight for concept  sparsity loss is {loss_sparsity_weight:.4f}.")
print(f"# Weight for concept diversity loss is {loss_diversity_weight:.4f}.")
print(f"# Train up to {n_epoch} epochs, with barch_size={batch_size}.")
print(f"# Save model's checkpoint for every {save_interval} epochs,")
print(f"# checkpoints locate in {checkpoint_dir}.")
print("#"*100)

# 正常训练并验证
train_data = use_data_folder_info["train_folder_path"]
print(f"# Train on data from {train_data}.")
eval_data = use_data_folder_info["val_folder_path"]
print(f"# Evaluate on data from {eval_data}.")
eval_major_data = use_data_folder_info["major_val_folder_path"]
print(f"# Evaluate on major data from {eval_major_data}.")
eval_minor_data = use_data_folder_info["minor_val_folder_path"]
print(f"# Evaluate on minor data from {eval_minor_data}.")
print("#"*100+"\n")

##########################
# Dataset and DataLoader
##########################

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=224,
                                     scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset from the directory
train_dataset = ImageFolder(root=train_data, transform=train_transform)
eval_dataset = ImageFolder(root=eval_data, transform=eval_transform)

# major_val 和 minor_val 的类别是 train 和 val 的子集
tmp_major_dataset = ImageFolder(root=eval_major_data)
tmp_minor_dataset = ImageFolder(root=eval_minor_data)


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
    root=eval_major_data, transform=eval_transform, target_transform=major_to_train)
eval_minor_dataset = ImageFolder(
    root=eval_minor_data, transform=eval_transform, target_transform=minor_to_train)


# Create DataLoader instances

num_workers = 8
pin_memory = True

train_loader = DataLoader(
    train_dataset, shuffle=True,
    batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
)
eval_loader = DataLoader(
    eval_dataset, shuffle=False,
    batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
)
eval_major_loader = DataLoader(
    eval_major_dataset, shuffle=False,
    batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
)
eval_minor_loader = DataLoader(
    eval_minor_dataset, shuffle=False,
    batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
)

##########################
# Model, loss and optimizer
##########################

model = PROVIDED_MODELS[use_model](num_classes,
                                   num_concepts,
                                   norm_concepts,
                                   norm_summary).to(device)
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

    def concept_sparsity_reg():
        # 熵越小, 分布越不均匀
        # clamp attention_weights, 避免log2后出现nan
        return torch.mean(-torch.sum(attention_weights.clamp(min=1e-9) *
                                     torch.log2(attention_weights.clamp(min=1e-9)), dim=1))

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
    loss_sparsity = concept_sparsity_reg()
    loss_diversity = concept_diversity_reg()

    loss = loss_classification + loss_sparsity_weight * \
        loss_sparsity + loss_diversity_weight * loss_diversity

    return loss, loss_classification, loss_sparsity, loss_diversity


optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 创建 warmup 调度器
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return 0.1 * (epoch + 1)
    else:
        return 1


warmup_scheduler = LambdaLR(
    optimizer, lr_lambda=warmup_lambda, verbose=True
)

plateau_scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10, verbose=True
)

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

    attention = []

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
            if returned_dict.get("attention_weights", None) is not None:
                attention.append(
                    returned_dict["attention_weights"].detach().cpu().numpy()
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

    if len(attention) > 0:
        analyze_sparsity(attention)
    return metric_dict


def analyze_sparsity(attn):
    baseline = np.mean(attn)
    percentile_25 = np.percentile(attn, 25) / baseline
    percentile_50 = np.percentile(attn, 50) / baseline
    percentile_75 = np.percentile(attn, 75) / baseline
    percentile_90 = np.percentile(attn, 90) / baseline
    percentile_99 = np.percentile(attn, 99) / baseline
    print('25: %.5f, 50: %.5f, 75: %.5f, 90: %.5f, 99: %.5f' % (
        percentile_25, percentile_50, percentile_75, percentile_90, percentile_99))


early_stopped = False
early_stop_counter = 0
patience = 20

best_val_acc = 0
best_val_acc_major = 0
best_val_acc_minor = 0
best_epoch = 0
best_checkpoint_path = ""
last_val_acc = 0
last_val_acc_major = 0
last_val_acc_minor = 0
last_epoch = 0
last_checkpoint_path = ""

for epoch in range(n_epoch):

    # train one epoch
    desc = f"Training epoch {epoch + 1}/{n_epoch}"
    train_dict = run_epoch(desc, model, train_loader, train=True)

    # validation
    desc = f"Evaluate epoch {epoch + 1}/{n_epoch}"
    eval_dict = run_epoch(desc, model, eval_loader, train=False)

    desc = f"MajorVal epoch {epoch + 1}/{n_epoch}"
    eval_major_dict = run_epoch(desc, model, eval_major_loader, train=False)

    desc = f"MinorVal epoch {epoch + 1}/{n_epoch}"
    eval_minor_dict = run_epoch(desc, model, eval_minor_loader, train=False)

    # 调整学习率
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    plateau_scheduler.step(eval_dict["acc"])

    # model checkpoint
    model_name_elements = [
        "epoch",
        f"{epoch + 1}",
        f"{train_dict['loss']:.4f}",
        f"{train_dict['acc']:.4f}",
        f"{eval_dict['loss']:.4f}",
        f"{eval_dict['acc']:.4f}",
        f"{eval_major_dict['loss']:.4f}",
        f"{eval_major_dict['acc']:.4f}",
        f"{eval_minor_dict['loss']:.4f}",
        f"{eval_minor_dict['acc']:.4f}"
    ]
    model_name = "_".join(model_name_elements) + ".pt"

    if eval_dict['acc'] > best_val_acc:
        early_stop_counter = 0
        best_val_acc = eval_dict['acc']
        best_val_acc_major = eval_major_dict['acc']
        best_val_acc_minor = eval_minor_dict['acc']
        best_epoch = epoch + 1
        if best_checkpoint_path != "":
            os.remove(best_checkpoint_path)
        best_checkpoint_path = os.path.join(
            checkpoint_dir, "best_" + model_name
        )
        save(model, best_checkpoint_path)
    else:
        early_stop_counter += 1
        print(f"early_stop_counter: {early_stop_counter}\n")

    last_val_acc = eval_dict['acc']
    last_val_acc_major = eval_major_dict['acc']
    last_val_acc_minor = eval_minor_dict['acc']
    last_epoch = epoch + 1
    last_checkpoint_path = os.path.join(checkpoint_dir, model_name)

    if epoch % save_interval == 0:
        save(model, last_checkpoint_path)

    if early_stop_counter >= patience:
        early_stopped = True
        break

if not os.path.exists(last_checkpoint_path):
    save(model, last_checkpoint_path)

if early_stopped:
    print("Early stopped.")
else:
    print("Normal stopped.")

#####################################
# Save summary and detailed logs
#####################################

log_elements = {
    "date": time.strftime("%Y%m%d", time.localtime(time.time())),
    "mode": "standard",
    "data_folder": use_data_folder,
    "model": use_model,
    "num_concepts": num_concepts,
    "norm_concepts": norm_concepts,
    "norm_summary": norm_summary,
    "loss_sparsity_weight": loss_sparsity_weight,
    "loss_diversity_weight": loss_diversity_weight,
    "supplementary_description": args.supplementary_description,
    "num_epochs": n_epoch,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "save_interval": save_interval,
    "checkpoint_dir": checkpoint_dir,
    "early_stopped": early_stopped,
    "best_val_acc": best_val_acc,
    "best_val_acc_major": best_val_acc_major,
    "best_val_acc_minor": best_val_acc_minor,
    "best_epoch": best_epoch,
    "best_checkpoint_path": best_checkpoint_path,
    "last_val_acc": last_val_acc,
    "last_val_acc_major": last_val_acc_major,
    "last_val_acc_minor": last_val_acc_minor,
    "last_epoch": last_epoch,
    "last_checkpoint_path": last_checkpoint_path,
    "detailed_log_path": args.detailed_log_path
}

pprint(log_elements)

print(f"Summary log to be saved in {args.summary_log_path}")
print(f"Detailed log to be saved in {args.detailed_log_path}")

with open(args.summary_log_path, "a") as f:
    f.write(json.dumps(log_elements))
    f.write("\n")

print("END. Checkpoints and logs are saved.")
