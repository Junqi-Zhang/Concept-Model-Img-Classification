import os
import shutil
import sys
import math
from tqdm import tqdm
from pprint import pprint
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from modules.data_folders import PROVIDED_DATASETS
from modules.models import MODELS
from modules.models_exp import MODELS_EXP
from modules.utils import load_model, Recorder
from modules.losses import capped_lp_norm_hinge, orthogonality_l2_norm, PIController


##########################
# Basic settings
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = Recorder()

# dataset
config.dataset_name = "Sampled_ImageNet_500x1000_500x5_Seed_6"  # 'n02391049'  340, 'n02389026' 339.
# config.dataset_name = "Sampled_ImageNet_200x1000_200x25_Seed_6"  # 'n02391049' 134
# config.dataset_name = "Sampled_ImageNet"  # 'n02391049' 83
config.update(PROVIDED_DATASETS[config.dataset_name])
# model
config.use_model = "BasicQuantResNet18V4Smooth"
config.att_smoothing = 0.2
config.num_concepts = 500
config.num_attended_concepts = 100
config.norm_concepts = False
config.norm_summary = True
config.grad_factor = 1
# loss
config.loss_sparsity_weight = 0.02
config.loss_sparsity_adaptive = False
config.loss_diversity_weight = 1.0
# train
config.batch_size = 125
# device
config.dataloader_workers = 16
config.dataloader_pin_memory = True
# checkpoint
config.load_checkpoint_path = "./checkpoints/Sampled_ImageNet_500x1000_500x5_Seed_6/BasicQuantResNet18V4Smooth/202309170851_on_gpu_5/best_epoch_29_0.7879_0.3334_0.6603_0.6603_0.0058_0.2832_75.5_104.6_78.5_103.5_0.0.pt"
config.checkpoint_desc = "500概念_smooth0.2_sps0.02to100"

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
train_classes_idx = [idx for idx in train_dataset.class_to_idx.values()]
eval_dataset = ImageFolder(
    root=config.val_folder_path,
    transform=eval_transform
)
eval_classes_idx = [idx for idx in eval_dataset.class_to_idx.values()]

tmp_major_dataset = ImageFolder(root=config.major_val_folder_path)
tmp_minor_dataset = ImageFolder(root=config.minor_val_folder_path)

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
    root=config.major_val_folder_path,
    transform=eval_transform,
    target_transform=major_to_train
)
eval_major_classes_idx = [
    major_to_train(idx) for idx in eval_major_dataset.class_to_idx.values()
]
eval_minor_dataset = ImageFolder(
    root=config.minor_val_folder_path,
    transform=eval_transform,
    target_transform=minor_to_train
)
eval_minor_classes_idx = [
    minor_to_train(idx) for idx in eval_minor_dataset.class_to_idx.values()
]


# Create DataLoader instances

train_loader = DataLoader(
    train_dataset, shuffle=False,
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
        "norm_summary": config.norm_summary,
        "grad_factor": config.grad_factor,
        "smoothing": config.att_smoothing
    }
)

model = PROVIDED_MODELS[config.use_model](**model_parameters).to(device)

load_model(model, config.load_checkpoint_path)

criterion = nn.CrossEntropyLoss()
sparsity_controller = PIController(
    kp=0.001, ki=0.00001,
    target_metric=config.num_attended_concepts,
    initial_weight=config.loss_sparsity_weight
)


def compute_loss(returned_dict, targets, train=False):
    outputs = returned_dict["outputs"]
    attention_weights = returned_dict.get("attention_weights", None)
    concept_similarity = returned_dict.get("concept_similarity", None)

    def normalize_rows(input_tensor, epsilon=1e-10):
        input_tensor = input_tensor.to(torch.float)
        row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
        row_sums += epsilon
        normalized_tensor = input_tensor / row_sums
        return normalized_tensor

    loss_cls_per_img = criterion(outputs, targets)  # B * K
    loss_img_per_cls = criterion(
        outputs.t(), normalize_rows(F.one_hot(targets, config.num_classes).t())
    )  # K * B
    loss_classification = (loss_cls_per_img + loss_img_per_cls) / 2.0

    if (attention_weights is None) and (concept_similarity is None):
        loss_sparsity = torch.tensor(0.0)
        loss_diversity = torch.tensor(0.0)
        loss = loss_classification + config.loss_sparsity_weight * \
            loss_sparsity + config.loss_diversity_weight * loss_diversity
        return loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity

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


##########################
# Analyze pipeline
##########################


def run_epoch(desc, model, dataloader, classes_idx, metric_prefix=""):

    model.eval()

    metric_dict = dict()

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
                loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
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
                    attended_concepts_count = torch.sum(
                        (returned_dict.get("attention_weights").data - 1e-7) > 0,
                        dim=1
                    ).type(torch.float)
                    s10 = torch.quantile(attended_concepts_count, 0.10).item()
                    s50 = torch.quantile(attended_concepts_count, 0.50).item()
                    s90 = torch.quantile(attended_concepts_count, 0.90).item()
                else:
                    s10 = -1
                    s50 = -1
                    s90 = -1

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
            update_metric_dict("s10", s10)
            update_metric_dict("s50", s50)
            update_metric_dict("s90", s90)

            pbar.set_postfix(metric_dict)
            pbar.update(1)

            step += 1
    return attention, label


desc = f"Training"
train_attention, train_label = run_epoch(
    desc, model, train_loader, train_classes_idx, metric_prefix="train_"
)

desc = f"Evaluate"
eval_attention, eval_label = run_epoch(
    desc, model, eval_loader, eval_classes_idx, metric_prefix="val_"
)

desc = f"MajorVal"
eval_major_attention, eval_major_label = run_epoch(
    desc, model, eval_major_loader, eval_major_classes_idx, metric_prefix="major_"
)

desc = f"MinorVal"
eval_minor_attention, eval_minor_label = run_epoch(
    desc, model, eval_minor_loader, eval_minor_classes_idx, metric_prefix="minor_"
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
    indices = np.where(attention > 0.0)
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
        os.path.dirname(config.load_checkpoint_path),
        config.checkpoint_desc
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_image = stitch_images(image_paths, m)
    result_image.save(os.path.join(output_dir, f"concept_{i}.jpg"))
    copy_files_to_folder(image_paths, os.path.join(
        output_dir, f"concept_{i}_top_images"))
