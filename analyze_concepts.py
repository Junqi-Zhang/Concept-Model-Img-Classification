import os
import shutil
from tqdm import tqdm
from PIL import Image

import numpy as np  # 在 import torch 前
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data_folders import PROVIDED_DATA_FOLDERS
from models import PROVIDED_MODELS
from utils import load


##########################
# Basic settings
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

use_data_folder = "Sampled_ImageNet"
use_data_folder_info = PROVIDED_DATA_FOLDERS[use_data_folder]
num_classes = use_data_folder_info["num_classes"]

use_model = "BasicQuantResNet18V3"
num_concepts = 50
norm_concepts = False
norm_summary = False
grad_factor = 50
loss_sparsity_weight = 0.0
loss_diversity_weight = 1.0

batch_size = 125

load_checkpoint_path = "./checkpoints/Sampled_ImageNet/BasicQuantResNet18V3/202308291624_on_gpu_5/best_epoch_128_0.0222_0.9939_3.2961_0.6021_2.6051_0.6686_6.0602_0.3360.pt"
checkpoint_desc = "不单位化概念_无正交约束"

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


##########################
# Analyze pipeline
##########################


def run_epoch(desc, model, dataloader):

    model.eval()

    metric_dict = {
        "loss": 0.0,
        "loss_classification": 0.0,
        "loss_sparsity": 0.0,
        "loss_diversity": 0.0,
        "acc": 0.0
    }

    attention = []
    label = []

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

            with torch.no_grad():
                returned_dict = model(data)
                loss, loss_classification, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets
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

    return attention, label


def analyze_sparsity(attn):
    baseline = np.mean(attn)
    percentile_25 = np.percentile(attn, 25) / baseline
    percentile_50 = np.percentile(attn, 50) / baseline
    percentile_75 = np.percentile(attn, 75) / baseline
    percentile_90 = np.percentile(attn, 90) / baseline
    percentile_99 = np.percentile(attn, 99) / baseline
    print('25: %.5f, 50: %.5f, 75: %.5f, 90: %.5f, 99: %.5f' % (
        percentile_25, percentile_50, percentile_75, percentile_90, percentile_99))


desc = f"Training"
train_attention, train_label = run_epoch(desc, model, train_loader)

desc = f"Evaluate"
eval_attention, eval_label = run_epoch(desc, model, eval_loader)

desc = f"MajorVal"
eval_major_attention, eval_major_label = run_epoch(
    desc, model, eval_major_loader
)

desc = f"MinorVal"
eval_minor_attention, eval_minor_label = run_epoch(
    desc, model, eval_minor_loader
)


analyze_label = np.concatenate(train_label, axis=0)
analyze_attention = np.concatenate(train_attention, axis=0)

label2name_dict = {value: key for key,
                   value in train_dataset.class_to_idx.items()}

label_attention_dict = dict()

for label, attention in zip(analyze_label, analyze_attention):
    if label not in label_attention_dict.keys():
        label_attention_dict[label] = []
    label_attention_dict[label].append(attention)

label_concept_dict = dict()

for label, attention_list in label_attention_dict.items():
    indices = np.where(np.mean(attention_list, axis=0) > 0.1)
    label_concept_dict[label] = list(indices[0])

concept_label_dict = dict()
for label, concept_list in label_concept_dict.items():
    for concept in concept_list:
        if concept not in concept_label_dict.keys():
            concept_label_dict[concept] = []
        concept_label_dict[concept].append(label)
        concept_label_dict[concept].sort()

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
    # 计算每个小图的大小
    first_image = Image.open(image_paths[0])
    img_width, img_height = first_image.size

    # 创建一个新的大图，大小为 m * img_width x m * img_height
    stitched_image = Image.new('RGB', (m * img_width, m * img_height))

    # 遍历图片路径，将每个图片拼接到大图上
    for index, image_path in enumerate(image_paths):
        img = Image.open(image_path)

        # 计算当前图片在大图中的位置
        row = index // m
        col = index % m

        # 将图片粘贴到大图上
        stitched_image.paste(img, (col * img_width, row * img_height))

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
