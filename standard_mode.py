import os
import sys
import json
import math
import time
import argparse
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from preprocess.imagenet_classes import IMAGENET2012_CLASSES_IDX
from modules.data_folders import PROVIDED_DATASETS
from modules.models import MODELS
from modules.models_exp import MODELS_EXP
from modules.utils import save_model, load_model, Recorder
from modules.losses import capped_lp_norm_hinge, orthogonality_l2_norm, PIController


##########################
# Argument parsing
##########################

parser = argparse.ArgumentParser(
    description="Test model's ability of generalization."
)
# dataset
parser.add_argument("--dataset_name", required=True, type=str)
# model
parser.add_argument("--warmup_model", default="", type=str)
parser.add_argument("--warmup_checkpoint_path", default="", type=str)
parser.add_argument("--text_embeds_path", default="", type=str)
parser.add_argument("--use_model", required=True, type=str)
parser.add_argument("--att_smoothing", default=0.0, type=float)
parser.add_argument("--num_concepts", default=512, type=int)
parser.add_argument("--num_attended_concepts", default=50, type=int)
parser.add_argument("--norm_concepts", default="False")
parser.add_argument("--concept_dim", default=512, type=int)
parser.add_argument("--norm_summary", default="False")
parser.add_argument("--grad_factor", default=1.0, type=float)
# loss
parser.add_argument("--loss_sparsity_weight", default=0.0, type=float)
parser.add_argument("--loss_sparsity_adaptive", default="False")
parser.add_argument("--loss_diversity_weight", default=0.0, type=float)
# train
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--warmup_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--monitor_metric", default="minor_acc_subset", type=str)
parser.add_argument("--plateau_patience", default=3, type=int)
parser.add_argument("--early_stop_patience", default=12, type=int)
# log
parser.add_argument("--save_interval", default=1, type=int)
parser.add_argument("--supplementary_description", default="", type=str)
parser.add_argument("--summary_log_path", required=True, type=str)
parser.add_argument("--detailed_log_path", required=True, type=str)
# device
parser.add_argument("--gpu", required=True, type=int)
parser.add_argument("--dataloader_workers", default=8, type=int)
parser.add_argument("--dataloader_pin_memory", default="True")

args = parser.parse_args()

# eval boolean args from string
args.norm_concepts = eval(args.norm_concepts)
args.norm_summary = eval(args.norm_summary)
args.loss_sparsity_adaptive = eval(args.loss_sparsity_adaptive)
args.dataloader_pin_memory = eval(args.dataloader_pin_memory)

##########################
# Basic settings
##########################

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = Recorder(**vars(args))

config.update(PROVIDED_DATASETS[config.dataset_name])

if config.warmup_checkpoint_path == "":
    config.warmup_checkpoint_epoch = None
else:
    config.warmup_checkpoint_epoch = eval(
        os.path.basename(config.warmup_checkpoint_path).split("_")[1]
    )

config.time_stamp = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
config.checkpoint_dir = os.path.join(
    "./checkpoints/",
    config.dataset_name,
    config.warmup_model + config.use_model,
    config.time_stamp + f"_on_gpu_{config.gpu}"
)
if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)

# Confirm basic settings
print("\n"+"#"*100)
pprint(config.to_dict())
print("#"*100)

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
train_dataset = ImageFolder(
    root=config.train_folder_path,
    transform=train_transform
)
train_idx_transform = dict()
for key, value in train_dataset.class_to_idx.items():
    train_idx_transform[value] = IMAGENET2012_CLASSES_IDX[key]
train_classes_idx = [
    train_idx_transform[idx] for idx in train_dataset.class_to_idx.values()
]
train_acc_mask_idx = [
    idx for idx in train_dataset.class_to_idx.values()
]

eval_dataset = ImageFolder(
    root=config.val_folder_path,
    transform=eval_transform
)
eval_idx_transform = dict()
for key, value in eval_dataset.class_to_idx.items():
    eval_idx_transform[value] = IMAGENET2012_CLASSES_IDX[key]
eval_classes_idx = [
    eval_idx_transform[idx] for idx in eval_dataset.class_to_idx.values()
]
eval_acc_mask_idx = [
    idx for idx in eval_dataset.class_to_idx.values()
]

tmp_major_dataset = ImageFolder(root=config.major_val_folder_path)
major_to_eval_idx_transform = dict()
for key, value in tmp_major_dataset.class_to_idx.items():
    major_to_eval_idx_transform[value] = eval_dataset.class_to_idx[key]


def major_to_eval_transform(target):
    return major_to_eval_idx_transform[target]


eval_major_dataset = ImageFolder(
    root=config.major_val_folder_path,
    transform=eval_transform,
    target_transform=major_to_eval_transform
)
eval_major_acc_mask_idx = [
    major_to_eval_transform(idx) for idx in eval_major_dataset.class_to_idx.values()
]

tmp_minor_dataset = ImageFolder(root=config.minor_val_folder_path)
minor_to_eval_idx_transform = dict()
for key, value in tmp_minor_dataset.class_to_idx.items():
    minor_to_eval_idx_transform[value] = eval_dataset.class_to_idx[key]


def minor_to_eval_transform(target):
    return minor_to_eval_idx_transform[target]


eval_minor_dataset = ImageFolder(
    root=config.minor_val_folder_path,
    transform=eval_transform,
    target_transform=minor_to_eval_transform
)
eval_minor_acc_mask_idx = [
    minor_to_eval_transform(idx) for idx in eval_minor_dataset.class_to_idx.values()
]


# Create DataLoader instances

train_loader = DataLoader(
    train_dataset, shuffle=True,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_loader = DataLoader(
    eval_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_major_loader = DataLoader(
    eval_major_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_minor_loader = DataLoader(
    eval_minor_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)

##########################
# Model, loss and optimizer
##########################

PROVIDED_MODELS = OrderedDict(**MODELS, **MODELS_EXP)

model_parameters = dict(
    {
        "num_classes": config.num_classes,
        "num_concepts": config.num_concepts,
        "norm_concepts": config.norm_concepts,
        "concept_dim": config.concept_dim,
        "norm_summary": config.norm_summary,
        "grad_factor": config.grad_factor,
        "smoothing": config.att_smoothing,
        "text_embeds": torch.load(config.text_embeds_path).t()
    }
)

if config.warmup_model == "":
    model = PROVIDED_MODELS[config.use_model](**model_parameters).to(device)
elif config.warmup_model == config.use_model:
    model = PROVIDED_MODELS[config.use_model](**model_parameters).to(device)
    load_model(model, config.warmup_checkpoint_path)
else:
    pre_model = PROVIDED_MODELS[config.warmup_model](
        **model_parameters).to(device)
    load_model(pre_model, config.warmup_checkpoint_path)
    model = PROVIDED_MODELS[config.use_model](**model_parameters).to(device)
    model.load_state_dict(
        {
            name: param
            for name, param in pre_model.state_dict().items()
            if name in model.state_dict()
        },
        strict=False
    )

criterion = nn.CrossEntropyLoss()
sparsity_controller = PIController(
    kp=0.001, ki=0.00001,
    target_metric=config.num_attended_concepts,
    initial_weight=config.loss_sparsity_weight
)


def compute_loss(returned_dict, targets, train=False):
    outputs = returned_dict["outputs"]
    attention_weights = returned_dict["attention_weights"]
    concept_similarity = returned_dict["concept_similarity"]

    def normalize_rows(input_tensor, epsilon=1e-10):
        input_tensor = input_tensor.to(torch.float)
        row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
        row_sums += epsilon
        normalized_tensor = input_tensor / row_sums
        return normalized_tensor

    loss_cls_per_img = criterion(outputs, targets)  # B * K
    if train:
        num_classes = len(train_classes_idx)
    else:
        num_classes = len(eval_classes_idx)
    loss_img_per_cls = criterion(
        outputs.t(), normalize_rows(F.one_hot(targets, num_classes).t())
    )  # K * B
    loss_classification = (loss_cls_per_img + loss_img_per_cls) / 2.0

    loss_sparsity = capped_lp_norm_hinge(
        attention_weights,
        target=config.num_attended_concepts,
        gamma=1.0/config.num_attended_concepts,
        reduction="sum")
    loss_diversity = orthogonality_l2_norm(concept_similarity)

    def num_attended_concepts_s99(attention_weights):
        with torch.no_grad():
            s99 = torch.quantile(
                torch.sum(
                    (attention_weights - 1e-7) > 0,
                    dim=1
                ).type(torch.float),
                0.99
            ).item()
        return s99

    if config.loss_sparsity_adaptive and train:
        s99 = num_attended_concepts_s99(attention_weights)
        config.loss_sparsity_weight = sparsity_controller.update(s99)

    loss = loss_classification + config.loss_sparsity_weight * \
        loss_sparsity + config.loss_diversity_weight * loss_diversity

    return loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity


optimizer = optim.AdamW(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)


def warmup_lambda(epoch):
    half_epochs = math.ceil(config.warmup_epochs / 2)
    if epoch < half_epochs:
        return 1 / half_epochs
    elif epoch < config.warmup_epochs:
        return 1 / half_epochs * (epoch - half_epochs + 1)
    else:
        return 1


warmup_scheduler = LambdaLR(
    optimizer, lr_lambda=warmup_lambda, verbose=True
)

plateau_scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=config.plateau_patience, verbose=True
)

##########################
# Training pipeline
##########################


def run_epoch(desc, model, dataloader, acc_mask_idx, train=False, metric_prefix=""):
    # train pipeline
    if train:
        model.train()
    else:
        model.eval()

    metric_dict = dict()

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
                returned_dict = model(data, train_classes_idx)
                loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets, train=True
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    returned_dict = model(data, eval_classes_idx)
                    loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                        returned_dict, targets, train=False
                    )

            # display the metrics
            with torch.no_grad():

                acc = (torch.argmax(returned_dict["outputs"].data,
                                    1) == targets).sum() / targets.size(0)

                mask = torch.zeros_like(returned_dict["outputs"].data)
                mask[:, acc_mask_idx] = 1
                acc_subset = (torch.argmax(returned_dict["outputs"].data * mask,
                                           1) == targets).sum() / targets.size(0)

                attended_concepts_count = torch.sum(
                    (returned_dict["attention_weights"].data - 1e-7) > 0,
                    dim=1
                ).type(torch.float)
                s10 = torch.quantile(attended_concepts_count, 0.10)
                s50 = torch.quantile(attended_concepts_count, 0.50)
                s90 = torch.quantile(attended_concepts_count, 0.90)

            def update_metric_dict(key, value, average=True):
                if average:
                    metric_dict[metric_prefix + key] = (
                        metric_dict.get(
                            metric_prefix + key, 0
                        ) * step + value
                    ) / (step + 1)
                else:
                    metric_dict[metric_prefix + key] = value

            update_metric_dict("acc", acc.item())
            update_metric_dict("acc_subset", acc_subset.item())
            update_metric_dict("loss", loss.item())
            update_metric_dict("loss_cpi", loss_cls_per_img.item())
            update_metric_dict("loss_ipc", loss_img_per_cls.item())
            update_metric_dict("loss_dvs", loss_diversity.item())
            update_metric_dict("loss_sps", loss_sparsity.item())
            update_metric_dict(
                "loss_sps_w", config.loss_sparsity_weight, average=False
            )
            update_metric_dict("s10", s10.item())
            update_metric_dict("s50", s50.item())
            update_metric_dict("s90", s90.item())

            pbar.set_postfix(metric_dict)
            pbar.update(1)

            step += 1
    return metric_dict


early_stopped = False
early_stop_counter = 0

current_metric = Recorder()
current_checkpoint_path = ""
best_metric = Recorder()
best_checkpoint_path = ""

for epoch in range(config.num_epochs):

    # train one epoch
    desc = f"Training epoch {epoch + 1}/{config.num_epochs}"
    train_dict = run_epoch(desc, model, train_loader,
                           train_acc_mask_idx,
                           train=True, metric_prefix="train_")

    # validation
    desc = f"Evaluate epoch {epoch + 1}/{config.num_epochs}"
    eval_dict = run_epoch(desc, model, eval_loader,
                          eval_acc_mask_idx,
                          train=False, metric_prefix="val_")

    desc = f"MajorVal epoch {epoch + 1}/{config.num_epochs}"
    eval_major_dict = run_epoch(desc, model, eval_major_loader,
                                eval_major_acc_mask_idx,
                                train=False,  metric_prefix="major_")

    desc = f"MinorVal epoch {epoch + 1}/{config.num_epochs}"
    eval_minor_dict = run_epoch(desc, model, eval_minor_loader,
                                eval_minor_acc_mask_idx,
                                train=False, metric_prefix="minor_")

    current_metric.update(
        train_dict | eval_dict | eval_major_dict | eval_minor_dict
    )
    current_metric.epoch = epoch + 1

    # adjust learing rate
    if epoch < config.warmup_epochs:
        warmup_scheduler.step()
    plateau_scheduler.step(current_metric.get(config.monitor_metric))

    # model checkpoint
    model_name_elements = [
        "epoch",
        f"{current_metric.epoch}",
        f"{current_metric.train_acc:.4f}",
        f"{current_metric.val_acc:.4f}",
        f"{current_metric.major_acc:.4f}",
        f"{current_metric.major_acc_subset:.4f}",
        f"{current_metric.minor_acc:.4f}",
        f"{current_metric.minor_acc_subset:.4f}",
        f"{current_metric.train_s50:.1f}",
        f"{current_metric.train_s90:.1f}",
        f"{current_metric.val_s50:.1f}",
        f"{current_metric.val_s90:.1f}",
        f"{current_metric.train_loss_dvs:.1f}"
    ]
    model_name = "_".join(model_name_elements) + ".pt"
    current_checkpoint_path = os.path.join(
        config.checkpoint_dir, model_name
    )

    if current_metric.get(config.monitor_metric) >= best_metric.get(config.monitor_metric, 0):
        early_stop_counter = 0
        best_metric.update(current_metric.to_dict())
        if best_checkpoint_path != "":
            os.remove(best_checkpoint_path)
        best_checkpoint_path = os.path.join(
            config.checkpoint_dir, "best_" + model_name
        )
        save_model(model, best_checkpoint_path)
    else:
        early_stop_counter += 1
        print(f"early_stop_counter: {early_stop_counter}\n")

    if epoch % config.save_interval == 0:
        save_model(model, current_checkpoint_path)

    if early_stop_counter >= config.early_stop_patience:
        early_stopped = True
        break

if not os.path.exists(current_checkpoint_path):
    save_model(model, current_checkpoint_path)

if early_stopped:
    print("Early stopped.")
else:
    print("Normal stopped.")

#####################################
# Save summary and detailed logs
#####################################

log_elements = dict()
log_elements.update(config.to_dict())
log_elements.update(
    {
        f"best_{key}": value
        for key, value in best_metric.to_dict().items()
    }
)
log_elements["best_checkpoint_path"] = best_checkpoint_path
log_elements.update(
    {
        f"last_{key}": value
        for key, value in current_metric.to_dict().items()
    }
)
log_elements["last_checkpoint_path"] = current_checkpoint_path

pprint(log_elements)

print(f"Summary log to be saved in {config.summary_log_path}")
print(f"Detailed log to be saved in {config.detailed_log_path}")

with open(config.summary_log_path, "a") as f:
    f.write(json.dumps(log_elements))
    f.write("\n")

print("END. Checkpoints and logs are saved.")
