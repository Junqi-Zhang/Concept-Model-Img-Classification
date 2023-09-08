import os
import sys
import json
import time
import argparse
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

import numpy as np  # 在 import torch 前
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from data_folders import PROVIDED_DATA_FOLDERS
from models import MODELS
from models_exp import MODELS_EXP
from utils import save, load
from utils import capped_lp_norm, orthogonality_l2_norm, PIController


##########################
# Argument parsing
##########################

parser = argparse.ArgumentParser(
    description="Test model's ability of generalization."
)

parser.add_argument("--data_folder", required=True)

parser.add_argument("--model", required=True)
parser.add_argument("--num_concepts", default=64, type=int)
parser.add_argument("--num_attended_concepts", default=5, type=int)
parser.add_argument("--norm_concepts", default="False")
parser.add_argument("--norm_summary", default="False")
parser.add_argument("--grad_factor", default=1.0, type=float)
parser.add_argument("--loss_sparsity_weight", default=0.0, type=float)
parser.add_argument("--loss_sparsity_adaptive", default="False")
parser.add_argument("--loss_diversity_weight", default=0.0, type=float)

parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)

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

PROVIDED_MODELS = OrderedDict(**MODELS, **MODELS_EXP)
use_model = args.model
num_concepts = args.num_concepts
num_attended_concepts = args.num_attended_concepts
norm_concepts = eval(args.norm_concepts)
norm_summary = eval(args.norm_summary)
grad_factor = args.grad_factor
loss_sparsity_weight = args.loss_sparsity_weight
loss_sparsity_adaptive = eval(args.loss_sparsity_adaptive)
loss_diversity_weight = args.loss_diversity_weight

n_epoch = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = 0.01
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
print(f"# Gradient Factor on Softmax: {grad_factor}.")
print(
    f"# Weight for concept  sparsity loss is {loss_sparsity_weight:.4f}, "
    f"adaptive: {loss_sparsity_adaptive}, target: {num_attended_concepts}."
)
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
train_classes_idx = [idx for idx in train_dataset.class_to_idx.values()]
eval_dataset = ImageFolder(root=eval_data, transform=eval_transform)
eval_classes_idx = [idx for idx in eval_dataset.class_to_idx.values()]

# major_val 和 minor_val 的类别是 train 和 val 的子集
tmp_major_dataset = ImageFolder(root=eval_major_data)
tmp_minor_dataset = ImageFolder(root=eval_minor_data)

major_to_train_idx_transform = dict()
for key, value in tmp_major_dataset.class_to_idx.items():
    major_to_train_idx_transform[value] = train_dataset.class_to_idx[key]


def major_to_train(target):
    return major_to_train_idx_transform[target]


minor_to_train_idx_transform = dict()
for key, value in tmp_minor_dataset.class_to_idx.items():
    minor_to_train_idx_transform[value] = train_dataset.class_to_idx[key]


def minor_to_train(target):
    return minor_to_train_idx_transform[target]


eval_major_dataset = ImageFolder(
    root=eval_major_data, transform=eval_transform, target_transform=major_to_train
)
eval_major_classes_idx = [
    major_to_train(idx) for idx in eval_major_dataset.class_to_idx.values()
]
eval_minor_dataset = ImageFolder(
    root=eval_minor_data, transform=eval_transform, target_transform=minor_to_train
)
eval_minor_classes_idx = [
    minor_to_train(idx) for idx in eval_minor_dataset.class_to_idx.values()
]


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
                                   norm_summary,
                                   grad_factor).to(device)
criterion = nn.CrossEntropyLoss()
sparsity_controller = PIController(
    kp=0.001, ki=0.00001,
    target_metric=num_attended_concepts,
    initial_weight=loss_sparsity_weight
)


def compute_loss(returned_dict, targets, train=False):

    global loss_sparsity_weight

    outputs = returned_dict["outputs"]

    # 代码兼容 ResNet18 等常规模型
    attention_weights = returned_dict.get("attention_weights", None)
    concept_similarity = returned_dict.get("concept_similarity", None)

    def normalize_rows(input_tensor, epsilon=1e-10):
        input_tensor = input_tensor.to(torch.float)
        row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
        row_sums += epsilon  # 添加一个小的正数以避免除以0
        normalized_tensor = input_tensor / row_sums
        return normalized_tensor

    loss_cls_per_img = criterion(outputs, targets)  # B * K
    loss_img_per_cls = criterion(
        outputs.t(), normalize_rows(F.one_hot(targets, num_classes).t())
    )  # K * B
    loss_classification = (loss_cls_per_img + loss_img_per_cls) / 2.0

    if (attention_weights is None) and (concept_similarity is None):
        loss_sparsity = torch.tensor(0.0)
        loss_diversity = torch.tensor(0.0)
        loss = loss_classification + loss_sparsity_weight * \
            loss_sparsity + loss_diversity_weight * loss_diversity
        return loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity

    loss_sparsity = capped_lp_norm(attention_weights, reduction="sum")
    loss_diversity = orthogonality_l2_norm(concept_similarity)

    def compute_s50(attention_weights):
        with torch.no_grad():
            s50 = torch.sum(attention_weights > 0, dim=1).median().item()
        return s50

    if loss_sparsity_adaptive and train:  # 只有训练的时候才能PI控制
        s50 = compute_s50(attention_weights)
        loss_sparsity_weight = sparsity_controller.update(s50)

    loss = loss_classification + loss_sparsity_weight * \
        loss_sparsity + loss_diversity_weight * loss_diversity

    return loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity


optimizer = optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)


# 创建 warmup 调度器
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return 1 / warmup_epochs * (epoch + 1)
    else:
        return 1


warmup_scheduler = LambdaLR(
    optimizer, lr_lambda=warmup_lambda, verbose=True
)

plateau_scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=True
)

##########################
# Training pipeline
##########################


def run_epoch(desc, model, dataloader, classes_idx, train=False):
    # train pipeline
    if train:
        model.train()
    else:
        model.eval()

    metric_dict = {
        "loss": 0.0,
        "loss_cls_per_img": 0.0,
        "loss_img_per_cls": 0.0,
        "loss_sparsity": 0.0,
        "loss_sparsity_weight": loss_sparsity_weight,
        "loss_diversity": 0.0,
        "acc": 0.0,
        "acc_subset": 0.0,
        "s50": -1.0,
        "s90": -1.0
    }

    step = 0
    with tqdm(
        total=len(dataloader),
        desc=desc,
        postfix=dict,
        mininterval=1,
        file=sys.stdout,
        dynamic_ncols=True
    ) as pbar:
        for data, targets in dataloader:
            # data process
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            if train:
                returned_dict = model(data)
                loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets, train=True
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    returned_dict = model(data)
                    loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                        returned_dict, targets, train=False
                    )

            # display the metrics
            with torch.no_grad():

                acc = (torch.argmax(returned_dict["outputs"].data,
                                    1) == targets).sum() / targets.size(0)

                mask = torch.zeros_like(returned_dict["outputs"].data)
                mask[:, classes_idx] = 1
                acc_subset = (torch.argmax(returned_dict["outputs"].data * mask,
                                           1) == targets).sum() / targets.size(0)

                if returned_dict.get("attention_weights", None) is not None:
                    c = torch.sum(
                        (returned_dict.get("attention_weights").data - 1e-7) > 0,
                        dim=1
                    ).cpu().numpy()
                    n_selected_50 = np.percentile(c, 50)
                    n_selected_90 = np.percentile(c, 90)
                else:
                    n_selected_50 = -1
                    n_selected_90 = -1

            metric_dict["loss"] = (metric_dict["loss"] *
                                   step + loss.item()) / (step + 1)
            metric_dict["loss_cls_per_img"] = (metric_dict["loss_cls_per_img"] *
                                               step + loss_cls_per_img.item()) / (step + 1)
            metric_dict["loss_img_per_cls"] = (metric_dict["loss_img_per_cls"] *
                                               step + loss_img_per_cls.item()) / (step + 1)
            metric_dict["loss_sparsity"] = (metric_dict["loss_sparsity"] *
                                            step + loss_sparsity.item()) / (step + 1)
            metric_dict["loss_sparsity_weight"] = loss_sparsity_weight
            metric_dict["loss_diversity"] = (metric_dict["loss_diversity"] *
                                             step + loss_diversity.item()) / (step + 1)
            metric_dict["acc"] = (metric_dict["acc"] *
                                  step + acc.item()) / (step + 1)
            metric_dict["acc_subset"] = (metric_dict["acc_subset"] *
                                         step + acc_subset.item()) / (step + 1)
            metric_dict["s50"] = (metric_dict["s50"] *
                                  step + n_selected_50) / (step + 1)
            metric_dict["s90"] = (metric_dict["s90"] *
                                  step + n_selected_90) / (step + 1)

            pbar.set_postfix(
                **{
                    "loss": metric_dict["loss"],
                    "loss_cpi": metric_dict["loss_cls_per_img"],
                    "loss_ipc": metric_dict["loss_img_per_cls"],
                    "loss_sps": metric_dict["loss_sparsity"],
                    "loss_sps_w": metric_dict["loss_sparsity_weight"],
                    "loss_dvs": metric_dict["loss_diversity"],
                    "acc": metric_dict["acc"],
                    "acc_sub": metric_dict["acc_subset"],
                    "s50": metric_dict["s50"],
                    "s90": metric_dict["s90"]
                }
            )
            pbar.update(1)

            step += 1

    return metric_dict


early_stopped = False
early_stop_counter = 0
patience = 20

best_train_acc = 0
best_val_acc = 0
best_val_acc_major = 0
best_val_acc_subset_major = 0
best_val_acc_minor = 0
best_val_acc_subset_minor = 0
best_val_s50 = -1
best_val_s90 = -1
best_val_loss_dvs = 0
best_epoch = 0
best_checkpoint_path = ""

last_train_acc = 0
last_val_acc = 0
last_val_acc_major = 0
last_val_acc_subset_major = 0
last_val_acc_minor = 0
last_val_acc_subset_minor = 0
last_val_s50 = -1
last_val_s90 = -1
last_val_loss_dvs = 0
last_epoch = 0
last_checkpoint_path = ""

for epoch in range(n_epoch):

    # train one epoch
    desc = f"Training epoch {epoch + 1}/{n_epoch}"
    train_dict = run_epoch(desc, model, train_loader,
                           train_classes_idx, train=True)

    # validation
    desc = f"Evaluate epoch {epoch + 1}/{n_epoch}"
    eval_dict = run_epoch(desc, model, eval_loader,
                          eval_classes_idx, train=False)

    desc = f"MajorVal epoch {epoch + 1}/{n_epoch}"
    eval_major_dict = run_epoch(desc, model, eval_major_loader,
                                eval_major_classes_idx, train=False)

    desc = f"MinorVal epoch {epoch + 1}/{n_epoch}"
    eval_minor_dict = run_epoch(desc, model, eval_minor_loader,
                                eval_minor_classes_idx, train=False)

    # 调整学习率
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    plateau_scheduler.step(eval_minor_dict["acc_subset"])  # 监控 minor 内部 acc

    # model checkpoint
    model_name_elements = [
        "epoch",
        f"{epoch + 1}",
        f"{train_dict['acc']:.4f}",
        f"{eval_dict['acc']:.4f}",
        f"{eval_major_dict['acc']:.4f}",
        f"{eval_major_dict['acc_subset']:.4f}",
        f"{eval_minor_dict['acc']:.4f}",
        f"{eval_minor_dict['acc_subset']:.4f}",
        f"{train_dict['s50']:.1f}",
        f"{train_dict['s90']:.1f}",
        f"{eval_dict['s50']:.1f}",
        f"{eval_dict['s90']:.1f}",
        f"{train_dict['loss_diversity']:.1f}"
    ]
    model_name = "_".join(model_name_elements) + ".pt"

    # 早停监控 minor acc_subset
    if eval_minor_dict["acc_subset"] >= best_val_acc_subset_minor:
        early_stop_counter = 0
        best_train_acc = train_dict["acc"]
        best_val_acc = eval_dict["acc"]
        best_val_acc_major = eval_major_dict["acc"]
        best_val_acc_subset_major = eval_major_dict["acc_subset"]
        best_val_acc_minor = eval_minor_dict["acc"]
        best_val_acc_subset_minor = eval_minor_dict["acc_subset"]
        best_val_s50 = eval_dict["s50"]
        best_val_s90 = eval_dict["s90"]
        best_val_loss_dvs = train_dict["loss_diversity"]
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

    last_train_acc = train_dict["acc"]
    last_val_acc = eval_dict["acc"]
    last_val_acc_major = eval_major_dict["acc"]
    last_val_acc_subset_major = eval_major_dict["acc_subset"]
    last_val_acc_minor = eval_minor_dict["acc"]
    last_val_acc_subset_minor = eval_minor_dict["acc_subset"]
    last_val_s50 = eval_dict["s50"]
    last_val_s90 = eval_dict["s90"]
    last_val_loss_dvs = train_dict["loss_diversity"]
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
    "num_attended_concepts": num_attended_concepts,
    "norm_concepts": norm_concepts,
    "norm_summary": norm_summary,
    "grad_factor": grad_factor,
    "loss_sparsity_weight": loss_sparsity_weight,
    "loss_sparsity_adaptive": loss_sparsity_adaptive,
    "loss_diversity_weight": loss_diversity_weight,
    "supplementary_description": args.supplementary_description,
    "num_epochs": n_epoch,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "save_interval": save_interval,
    "checkpoint_dir": checkpoint_dir,
    "early_stopped": early_stopped,
    "best_train_acc": best_train_acc,
    "best_val_acc": best_val_acc,
    "best_val_acc_major": best_val_acc_major,
    "best_val_acc_subset_major": best_val_acc_subset_major,
    "best_val_acc_minor": best_val_acc_minor,
    "best_val_acc_subset_minor": best_val_acc_subset_minor,
    "best_val_s50": best_val_s50,
    "best_val_s90": best_val_s90,
    "best_val_loss_dvs": best_val_loss_dvs,
    "best_epoch": best_epoch,
    "best_checkpoint_path": best_checkpoint_path,
    "last_train_acc": last_train_acc,
    "last_val_acc": last_val_acc,
    "last_val_acc_major": last_val_acc_major,
    "last_val_acc_subset_major": last_val_acc_subset_major,
    "last_val_acc_minor": last_val_acc_minor,
    "last_val_acc_subset_minor": last_val_acc_subset_minor,
    "last_val_s50": last_val_s50,
    "last_val_s90": last_val_s90,
    "last_val_loss_dvs": last_val_loss_dvs,
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
