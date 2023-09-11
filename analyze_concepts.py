import os
import shutil
import sys
import math
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import numpy as np  # 在 import torch 前
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_folders import PROVIDED_DATA_FOLDERS
from models import MODELS
from models_exp import MODELS_EXP
from utils import capped_lp_norm, orthogonality_l2_norm, PIController
from utils import load


##########################
# Basic settings
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

use_data_folder = "Sampled_ImageNet_200x1000_200x25_Seed_6"  # # 'n02391049' 134
# use_data_folder = "Sampled_ImageNet"  # # 'n02391049' 83
use_data_folder_info = PROVIDED_DATA_FOLDERS[use_data_folder]
num_classes = use_data_folder_info["num_classes"]

PROVIDED_MODELS = OrderedDict(**MODELS, **MODELS_EXP)
use_model = "BasicQuantResNet18V4NoSparse"
num_concepts = 250
num_attended_concepts = 5
norm_concepts = True
norm_summary = True
grad_factor = 1
loss_sparsity_weight = 0.0
loss_sparsity_adaptive = False
loss_diversity_weight = 1.0

batch_size = 125

load_checkpoint_path = "./checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309102206_on_gpu_7/best_epoch_34_0.8877_0.4262_0.7165_0.7189_0.1354_0.4568_250.0_250.0_250.0_250.0_0.0.pt"
checkpoint_desc = "250概念有正交约束"

##########################
# Dataset and DataLoader
##########################

# 训练和验证数据集
train_data = use_data_folder_info["train_folder_path"]
eval_data = use_data_folder_info["val_folder_path"]
eval_major_data = use_data_folder_info["major_val_folder_path"]
eval_minor_data = use_data_folder_info["minor_val_folder_path"]

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
    train_dataset, shuffle=False,
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
load(model, load_checkpoint_path)

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

    if (attention_weights is None) and (concept_similarity is None):
        loss_classification = criterion(outputs, targets)
        loss_sparsity = torch.tensor(0.0)
        loss_diversity = torch.tensor(0.0)
        loss = loss_classification + loss_sparsity_weight * \
            loss_sparsity + loss_diversity_weight * loss_diversity
        return loss, loss_classification, loss_sparsity, loss_diversity

    loss_classification = criterion(outputs, targets)
    loss_sparsity = capped_lp_norm(attention_weights)
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

    return loss, loss_classification, loss_sparsity, loss_diversity


##########################
# Analyze pipeline
##########################


def run_epoch(desc, model, dataloader, classes_idx):

    model.eval()

    metric_dict = {
        "loss": 0.0,
        "loss_classification": 0.0,
        "loss_sparsity": 0.0,
        "loss_sparsity_weight": loss_sparsity_weight,
        "loss_diversity": 0.0,
        "acc": 0.0,
        "acc_subset": 0.0,
        "s50": -1.0,
        "s90": -1.0
    }

    attention = []
    label = []

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

            with torch.no_grad():
                returned_dict = model(data)
                loss, loss_classification, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets, train=False
                )

            if returned_dict.get("attention_weights", None) is not None:
                attention.append(
                    returned_dict["attention_weights"].detach().cpu().numpy()
                )
                label.append(targets.cpu().numpy())

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
            metric_dict["loss_classification"] = (metric_dict["loss_classification"] *
                                                  step + loss_classification.item()) / (step + 1)
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
                    "loss_cls": metric_dict["loss_classification"],
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

    return attention, label


desc = f"Training"
train_attention, train_label = run_epoch(
    desc, model, train_loader, train_classes_idx
)

desc = f"Evaluate"
eval_attention, eval_label = run_epoch(
    desc, model, eval_loader, eval_classes_idx
)

desc = f"MajorVal"
eval_major_attention, eval_major_label = run_epoch(
    desc, model, eval_major_loader, eval_major_classes_idx
)

desc = f"MinorVal"
eval_minor_attention, eval_minor_label = run_epoch(
    desc, model, eval_minor_loader, eval_minor_classes_idx
)


analyze_label = np.concatenate(train_label, axis=0)
analyze_attention = np.concatenate(train_attention, axis=0)

label2name_dict = {value: key for key,
                   value in train_dataset.class_to_idx.items()}

label_concept_count_dict = dict()

for label, attention in zip(analyze_label, analyze_attention):
    if label not in label_concept_count_dict.keys():
        label_concept_count_dict[label] = np.zeros_like(
            attention, dtype=int
        )
    indices = np.where(attention > 0.05)
    label_concept_count_dict[label][indices] += 1

label_concept_dict = dict()

for label, count in label_concept_count_dict.items():
    concept_count_dict = {
        i: val for i,
        val in enumerate(count) if val > 0
    }
    label_concept_dict[label] = {
        k: v for k, v in sorted(
            concept_count_dict.items(),
            key=lambda item: (-item[1], item[0])
        )
    }

concept_label_dict = dict()
for label, concept_count_dict in label_concept_dict.items():
    for concept, count in concept_count_dict.items():
        if concept not in concept_label_dict.keys():
            concept_label_dict[concept] = dict()
        concept_label_dict[concept][label] = count
for concept, label_count_dict in concept_label_dict.items():
    concept_label_dict[concept] = {
        k: v for k, v in sorted(
            label_count_dict.items(),
            key=lambda item: (-item[1], item[0])
        )
    }

sorted_dict_items = sorted(label_concept_dict.items(), key=lambda x: x[0])
label_concept_dict = dict(sorted_dict_items)

sorted_dict_items = sorted(concept_label_dict.items(), key=lambda x: x[0])
concept_label_dict = dict(sorted_dict_items)


train_images = [sample[0] for sample in train_dataset.samples]


def get_top_k_filenames(filenames, arr, k):
    # 获取数组每列的前 k 个最大值的索引
    top_k_indices = np.argsort(-arr, axis=0)[:k]

    # 使用索引从文件名列表中获取对应的文件名
    result = []
    for col in range(top_k_indices.shape[1]):
        col_top_k_filenames = [filenames[i] for i in top_k_indices[:, col]]
        result.append(col_top_k_filenames)

    return result


m = 10
concept_images = get_top_k_filenames(train_images, analyze_attention, m * m)


def stitch_images(image_paths, m):
    # 获取所有图片的大小
    sizes = [Image.open(p).size for p in image_paths]

    # 计算平均长宽比
    avg_ratio = sum(width / height for width, height in sizes) / len(sizes)

    # 计算目标宽度和高度
    target_height = min(height for width, height in sizes)
    target_width = int(target_height * avg_ratio)

    # 调整每张图片的大小
    images = [Image.open(p) for p in image_paths]
    resized_images = [img.resize(
        (target_width, target_height), Image.LANCZOS
    ) for img in images]

    # 计算大图的大小
    n = math.ceil(len(image_paths) / m)
    stitched_image = Image.new('RGB', (m * target_width, n * target_height))

    # 遍历图片，将每个图片粘贴到大图上
    for index, img in enumerate(resized_images):
        # 计算当前图片在大图中的位置
        row = index // m
        col = index % m

        # 将图片粘贴到大图上
        stitched_image.paste(img, (col * target_width, row * target_height))

    return stitched_image


def copy_files_to_folder(file_list, target_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历文件列表，将每个文件复制到目标文件夹
    for file_path in file_list:
        if os.path.isfile(file_path):
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 构建目标文件路径
            target_path = os.path.join(target_folder, file_name)
            # 复制文件
            shutil.copy(file_path, target_path)
        else:
            print(f"Warning: {file_path} is not a valid file.")


for i, image_paths in enumerate(concept_images):
    output_dir = os.path.join(
        os.path.dirname(load_checkpoint_path),
        checkpoint_desc
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_image = stitch_images(image_paths, m)
    result_image.save(os.path.join(output_dir, f"concept_{i}.jpg"))
    copy_files_to_folder(image_paths, os.path.join(
        output_dir, f"concept_{i}_top_images"))
